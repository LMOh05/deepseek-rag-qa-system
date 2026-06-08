"""
评测模块 —— 量化评估 RAG 系统检索和生成质量。

指标：
- Recall@K: top-K 检索结果是否命中相关文档
- MRR: 第一个相关文档的平均倒数排名
- 自动计算 Recall@3、Recall@5、MRR
- 支持多配置对比和 Bad Case 分类
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.document_loader import load_documents
from src.chunker import chunk_documents, chunk_by_sentences
from src.embedder import get_embedding
from src.vector_store import VectorStore
from src.retriever import BM25Retriever, HybridRetriever


def load_test_questions():
    path = Path(__file__).parent.parent / "tests" / "test_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_recall_at_k(retrieved_docs: List, relevant_docs: List[str], k: int) -> bool:
    """判断 top-K 检索结果是否命中至少一个相关文档"""
    retrieved_sources = set()
    for doc in retrieved_docs[:k]:
        source = doc.metadata.get("filename", "")
        retrieved_sources.add(source)

    return bool(retrieved_sources & set(relevant_docs))


def compute_mrr(retrieved_docs: List, relevant_docs: List[str]) -> float:
    """计算 MRR (Mean Reciprocal Rank)"""
    for rank, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("filename", "")
        if source in relevant_docs:
            return 1.0 / rank
    return 0.0


def get_retrieved_sources(retrieved_docs: List, k: int = 3) -> List[str]:
    """提取检索结果的文档来源名"""
    seen = set()
    sources = []
    for doc in retrieved_docs[:k]:
        src = doc.metadata.get("filename", "?")
        if src not in seen:
            sources.append(src)
            seen.add(src)
    return sources


def evaluate_config(
    chunks: List,
    questions: List[Dict],
    config_name: str,
    retrieval_mode: str = "dense",
) -> Dict:
    """
    对一种配置执行完整评测。

    Args:
        chunks: 文档切片
        questions: 测试问题列表
        config_name: 配置名称
        retrieval_mode: dense/sparse/hybrid

    Returns:
        评测结果字典
    """
    print(f"\n{'='*60}")
    print(f"[Eval] {config_name} (mode={retrieval_mode})")
    print(f"{'='*60}")

    # 建索引
    embedding = get_embedding()
    vs = VectorStore()
    vs.rebuild(chunks)

    bm25 = BM25Retriever()
    bm25.index(chunks)
    hybrid = HybridRetriever(vs, bm25)

    results = {
        "config": config_name,
        "retrieval_mode": retrieval_mode,
        "total": len(questions),
        "recall_at_3": 0,
        "recall_at_5": 0,
        "mrr_sum": 0.0,
        "per_question": [],
        "bad_cases": [],
        "edge_cases_handled": 0,
    }

    recall3_count = 0
    recall5_count = 0
    mrr_total = 0.0
    edge_count = 0

    for q in questions:
        qid = q["id"]
        qtype = q["type"]
        question = q["question"]
        relevant = q["relevant_docs"]

        # 检索
        if retrieval_mode == "dense":
            retrieved = vs.similarity_search(question, k=config.RETRIEVAL_K)
        elif retrieval_mode == "sparse":
            hits = hybrid.retrieve_sparse(question, k=config.RETRIEVAL_K)
            retrieved = [doc for doc, _ in hits]
        elif retrieval_mode == "hybrid":
            retrieved = hybrid.retrieve_hybrid(question, top_k=config.RETRIEVAL_K)

        # 计算指标
        r3 = compute_recall_at_k(retrieved, relevant, 3)
        r5 = compute_recall_at_k(retrieved, relevant, 5)
        mrr = compute_mrr(retrieved, relevant)
        sources = get_retrieved_sources(retrieved, 3)

        if r3:
            recall3_count += 1
        if r5:
            recall5_count += 1
        mrr_total += mrr

        # 边缘 case 处理
        is_edge = (qtype == "边缘case" and not relevant)
        if is_edge:
            edge_count += 1

        # Bad case 记录（仅针对有相关文档但未命中的问题）
        if not r5 and relevant:
            results["bad_cases"].append({
                "id": qid,
                "question": question,
                "expected": relevant,
                "got": sources,
                "type": qtype,
                "reason": classify_bad_case(question, relevant, sources, chunks)
            })

        results["per_question"].append({
            "id": qid,
            "type": qtype,
            "question": question[:40],
            "recall@3": r3,
            "recall@5": r5,
            "mrr": round(mrr, 4),
            "top3_sources": sources,
            "expected": relevant,
            "is_edge_no_docs": is_edge,
        })

    total = len(questions)
    core_total = total - edge_count  # 排除无文档的边缘case

    results["total"] = total
    results["core_total"] = core_total
    results["recall_at_3"] = round(recall3_count / total, 4)
    results["recall_at_5"] = round(recall5_count / total, 4)
    # 核心指标（排除无文档的边缘case）
    results["core_recall_at_3"] = round(recall3_count / core_total, 4) if core_total else 0
    results["core_recall_at_5"] = round(recall5_count / core_total, 4) if core_total else 0
    results["mrr"] = round(mrr_total / total, 4)
    # 核心 MRR（仅包含有相关文档的问题）
    core_mrr = sum(pq["mrr"] for pq in results["per_question"] if not pq["is_edge_no_docs"])
    results["core_mrr"] = round(core_mrr / core_total, 4) if core_total else 0
    results["edge_cases_handled"] = edge_count

    # 按类型统计
    type_stats = defaultdict(lambda: {"total": 0, "recall3": 0, "recall5": 0})
    for pq in results["per_question"]:
        t = pq["type"]
        type_stats[t]["total"] += 1
        if pq["recall@3"]:
            type_stats[t]["recall3"] += 1
        if pq["recall@5"]:
            type_stats[t]["recall5"] += 1
    results["by_type"] = dict(type_stats)

    return results


def classify_bad_case(question: str, expected: List[str],
                      got: List[str], chunks: List) -> str:
    """对 Bad Case 进行归因分类"""
    if not got:
        return "检索完全失败"
    # 检查是否相关文档存在但未被检索
    if expected:
        expected_str = " ".join(expected)
        # 简单启发式：检查 top3 来源与期望是否有重叠
        if set(got) & set(expected):
            return "部分命中（排名靠后）"
        # 检查期望文档是否在切片中
        chunk_sources = {c.metadata.get("filename", "") for c in chunks}
        if set(expected) & chunk_sources:
            return "检索遗漏（文档存在但未召回）"
        return "文档缺失（切片中无此文档）"
    return "无相关文档（边缘case）"


def print_eval_report(all_results: List[Dict]):
    """打印完整评测报告"""
    test_data = load_test_questions()
    total = len(test_data["questions"])

    print("\n" + "=" * 90)
    print("RAG 系统效果评测报告")
    print("=" * 90)

    # 汇总对比表（含核心指标）
    print(f"\n{'='*90}")
    print("综合指标（含边缘case）:")
    print(f"{'配置':<25} {'Recall@3':<10} {'Recall@5':<10} {'MRR':<8} {'Bad':<6}")
    print("-" * 65)
    for r in all_results:
        bad_count = len(r["bad_cases"])
        print(f"  {r['config']:<23} {r['recall_at_3']:.2%}       {r['recall_at_5']:.2%}       {r['mrr']:.4f}   {bad_count}")

    print(f"\n核心指标（排除无文档边缘case，共{all_results[0]['core_total']}题）:")
    print(f"{'配置':<25} {'Recall@3':<10} {'Recall@5':<10} {'MRR':<8}")
    print("-" * 58)
    for r in all_results:
        print(f"  {r['config']:<23} {r['core_recall_at_3']:.2%}       {r['core_recall_at_5']:.2%}       {r['core_mrr']:.4f}")

    # 按问题类型统计（取第一个配置作为示例）
    if all_results:
        r = all_results[0]
        print(f"\n按问题类型统计（{r['config']}）:")
        print(f"{'类型':<15} {'总数':<6} {'Recall@3':<12} {'Recall@5':<12}")
        print("-" * 50)
        for t, stats in r["by_type"].items():
            r3 = stats["recall3"] / stats["total"] if stats["total"] else 0
            r5 = stats["recall5"] / stats["total"] if stats["total"] else 0
            print(f"  {t:<13} {stats['total']:<6} {r3:.0%}            {r5:.0%}")

    # Bad Case 汇总
    print(f"\n{'='*90}")
    print("Bad Case 归因分析")
    print(f"{'='*90}")

    # 合并所有配置的 bad case
    all_bad = {}
    for r in all_results:
        for bc in r["bad_cases"]:
            key = bc["id"]
            if key not in all_bad:
                all_bad[key] = bc
                all_bad[key]["configs"] = [r["config"]]
            else:
                all_bad[key]["configs"].append(r["config"])

    if all_bad:
        reason_counts = defaultdict(int)
        for bc in all_bad.values():
            reason_counts[bc["reason"]] += 1

        print(f"\n归因分布:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} 条")

        print(f"\n详细 Bad Case:")
        for qid, bc in sorted(all_bad.items()):
            print(f"\n  [{qid}] {bc['question'][:50]}")
            print(f"    类型: {bc['type']} | 期望: {bc['expected']} | 实际: {bc['got']}")
            print(f"    归因: {bc['reason']} | 影响配置: {bc['configs']}")
    else:
        print("\n  （无 Bad Case）")

    print(f"\n{'='*90}\n")


def main():
    print("=" * 90)
    print("RAG 系统效果评测")
    print(f"文档数: 3 | 测试集: 50 条问题")
    print(f"Embedding: {config.EMBEDDING_MODEL}")
    print("=" * 90)

    questions = load_test_questions()["questions"]
    docs = load_documents()

    all_results = []

    # 配置1: 基线 - 固定窗口 + 纯向量
    chunks1 = chunk_documents(docs, chunk_size=512, chunk_overlap=128)
    all_results.append(evaluate_config(chunks1, questions, "基线(固定窗口+向量)", "dense"))

    # 配置2: 语义分块 + 纯向量
    chunks2 = chunk_by_sentences(docs, max_chunk_size=512)
    all_results.append(evaluate_config(chunks2, questions, "语义分块+向量", "dense"))

    # 配置3: 语义分块 + 混合检索
    chunks3 = chunk_by_sentences(docs, max_chunk_size=512)
    all_results.append(evaluate_config(chunks3, questions, "语义分块+混合", "hybrid"))

    # 配置4: 语义分块 + 混合检索（最强配置，rerank 在阶段二的实验中待加）
    chunks4 = chunk_by_sentences(docs, max_chunk_size=512)
    all_results.append(evaluate_config(chunks4, questions, "语义分块+混合(最终)", "hybrid"))

    print_eval_report(all_results)

    # 保存结果到 JSON
    output_path = Path(__file__).parent.parent / "tests" / "eval_results.json"
    serializable = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "per_question"}
        serializable.append(r_copy)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"评测结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
