"""
混合检索对比实验 —— 阶段二核心实验之二。

对比三种检索模式对 top-3 命中率的影响：
1. 纯向量检索（dense）
2. 纯 BM25 检索（sparse）
3. 混合检索 RRF（hybrid）

并记录哪些 query 被 BM25 "救了回来"（向量没命中但 BM25/混合命中了）。
"""

import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedding
from src.vector_store import VectorStore
from src.retriever import BM25Retriever, HybridRetriever


def load_queries():
    queries_path = Path(__file__).parent.parent / "tests" / "test_queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def get_top_sources(hits):
    """提取检索结果的 top-3 来源文档名"""
    return [h.metadata.get("filename", "?") for h in hits[:3]]


def run_retrieval_experiment(chunks, queries):
    """
    对同一批切片执行三种检索模式对比。

    Returns:
        dict: 按模式组织的检索结果
    """
    # 1. 建 FAISS 索引
    embedding = get_embedding()
    vs = VectorStore()
    vs.rebuild(chunks)

    # 2. 建 BM25 索引
    bm25 = BM25Retriever()
    bm25.index(chunks)

    # 3. 混合检索器
    hybrid = HybridRetriever(vs, bm25)

    results = {"dense": {}, "sparse": {}, "hybrid": {}}

    for q in queries:
        qid = q["id"]

        # 向量检索
        dense_hits = vs.similarity_search(q["question"], k=5)
        results["dense"][qid] = {
            "question": q["question"],
            "expected": q["expected_doc"],
            "top3_sources": get_top_sources(dense_hits),
        }

        # BM25 检索
        sparse_hits = bm25.search(q["question"], k=5)
        sparse_docs = [doc for doc, _ in sparse_hits]
        results["sparse"][qid] = {
            "question": q["question"],
            "expected": q["expected_doc"],
            "top3_sources": get_top_sources(sparse_docs),
        }

        # 混合检索
        hybrid_hits = hybrid.retrieve_hybrid(q["question"], top_k=5)
        results["hybrid"][qid] = {
            "question": q["question"],
            "expected": q["expected_doc"],
            "top3_sources": get_top_sources(hybrid_hits),
        }

    return results


def print_report(results):
    """打印对比报告"""
    queries = results["dense"]

    print("\n" + "=" * 100)
    print("混合检索对比报告")
    print("=" * 100)

    # 逐题对比表
    header = f"{'ID':<4} {'问题':<20} {'期望':<16}"
    for mode in ["dense", "sparse", "hybrid"]:
        header += f" {'top1-'+mode:<18}"
    print(header)
    print("-" * 100)

    for qid in queries:
        qdata = queries[qid]
        short_q = qdata["question"][:18]
        expected = qdata["expected"]
        row = f"{qid:<4} {short_q:<20} {expected:<16}"
        for mode in ["dense", "sparse", "hybrid"]:
            top1 = results[mode][qid]["top3_sources"][0] if results[mode][qid]["top3_sources"] else "N/A"
            mark = "✓" if top1 == expected else "✗"
            row += f" {mark} {top1:<15}"
        print(row)

    # 命中率统计
    print("\n" + "=" * 100)
    print("top-1 命中率（top-1 是否命中期望文档）")
    print("-" * 100)
    total = len(queries)
    for mode in ["dense", "sparse", "hybrid"]:
        correct = sum(
            1 for qid in queries
            if results[mode][qid]["top3_sources"]
            and results[mode][qid]["top3_sources"][0] == results[mode][qid]["expected"]
        )
        print(f"  {mode}: {correct}/{total} = {correct/total*100:.0f}%")

    # BM25 救回分析
    print("\n" + "=" * 100)
    print("BM25 救回分析（向量 top1 未命中，但 BM25 或混合命中了）")
    print("-" * 100)
    rescued = []
    for qid in queries:
        dense_top1 = results["dense"][qid]["top3_sources"][0] if results["dense"][qid]["top3_sources"] else ""
        expected = results["dense"][qid]["expected"]
        if dense_top1 != results["dense"][qid]["expected"]:
            hyb_top1 = results["hybrid"][qid]["top3_sources"][0] if results["hybrid"][qid]["top3_sources"] else ""
            sp_top1 = results["sparse"][qid]["top3_sources"][0] if results["sparse"][qid]["top3_sources"] else ""
            rescued.append({
                "id": qid,
                "question": results["dense"][qid]["question"],
                "dense_hit": dense_top1,
                "sparse_hit": sp_top1,
                "hybrid_hit": hyb_top1,
            })

    if rescued:
        for r in rescued:
            print(f"  [{r['id']}] {r['question']}")
            print(f"       向量→{r['dense_hit']}  BM25→{r['sparse_hit']}  混合→{r['hybrid_hit']}")
    else:
        print("  （无）向量检索全部命中")

    print("=" * 100)


def main():
    print("=" * 80)
    print("混合检索对比实验")
    print(f"Embedding: {config.EMBEDDING_MODEL}")
    print(f"RRF k: {config.RRF_K}")
    print("=" * 80)

    queries = load_queries()
    print(f"测试问题: {len(queries)} 条")

    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"文档切片: {len(chunks)} 个")

    results = run_retrieval_experiment(chunks, queries)
    print_report(results)


if __name__ == "__main__":
    main()
