"""
切片策略对比实验 —— 阶段二核心实验之一。

对比两种策略对检索效果的影响：
1. 固定窗口 512/128（基线）
2. 固定窗口 256/64
3. 语义分块（按句子边界合并）

输出：每种策略的切片统计 + 检索命中情况
"""

import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.document_loader import load_documents
from src.chunker import chunk_documents, chunk_by_sentences
from src.embedder import get_embedding
from src.vector_store import VectorStore


def load_queries():
    """加载测试问题"""
    queries_path = Path(__file__).parent.parent / "tests" / "test_queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def run_experiment(chunks, queries, strategy_name):
    """
    对一组切片执行检索实验。

    Returns:
        dict: {query_id: {"question": str, "hits": [doc metadata], "time": float}}
    """
    # 建库
    embedding = get_embedding()
    vs = VectorStore()
    vs.rebuild(chunks)

    results = {}
    for q in queries:
        start = time.time()
        hits = vs.similarity_search(q["question"], k=5)
        elapsed = time.time() - start

        results[q["id"]] = {
            "question": q["question"],
            "expected_doc": q["expected_doc"],
            "hits": [
                {"source": h.metadata.get("filename", "?"), "preview": h.page_content[:80]}
                for h in hits
            ],
            "top1_source": hits[0].metadata.get("filename", "?") if hits else "N/A",
            "retrieval_time": f"{elapsed:.3f}s",
        }

    return results


def print_report(strategies_results):
    """打印对比报告"""
    print("\n" + "=" * 80)
    print("切片策略对比报告")
    print("=" * 80)

    # 表头
    print(f"\n{'ID':<4} {'问题':<20} {'期望文档':<16}", end="")
    for name in strategies_results:
        print(f" {name+' top1':<20}", end="")
    print()

    print("-" * 80)

    # 逐题对比
    queries = strategies_results[list(strategies_results.keys())[0]]
    for qid, qdata in queries.items():
        short_q = qdata["question"][:18]
        expected = qdata["expected_doc"]
        print(f"{qid:<4} {short_q:<20} {expected:<16}", end="")
        for name in strategies_results:
            top1 = strategies_results[name][qid]["top1_source"]
            match = "✓" if top1 == expected else "✗"
            print(f" {match} {top1:<16}", end="")
        print()

    # 命中率统计
    print("\n" + "=" * 80)
    print("命中率统计（top-1 命中期望文档）")
    print("-" * 80)
    total = len(queries)
    for name in strategies_results:
        correct = sum(
            1 for q in strategies_results[name].values()
            if q["top1_source"] == q["expected_doc"]
        )
        print(f"  {name}: {correct}/{total} = {correct/total*100:.0f}%")

    print("=" * 80)


def main():
    print("=" * 80)
    print("切片策略对比实验")
    print(f"Embedding: {config.EMBEDDING_MODEL}")
    print("=" * 80)

    queries = load_queries()
    print(f"测试问题数: {len(queries)}")

    # 加载原始文档
    docs = load_documents()
    print(f"文档段落数: {len(docs)}")

    strategies = {}

    # 策略1: 固定窗口 512/128（基线）
    print("\n[1/3] 策略1: 固定窗口 512/128...")
    chunks1 = chunk_documents(docs, chunk_size=512, chunk_overlap=128)
    print(f"  切片数: {len(chunks1)}")
    if chunks1:
        avg_len = sum(len(c.page_content) for c in chunks1) / len(chunks1)
        print(f"  平均长度: {avg_len:.0f} 字符")
    strategies["512/128"] = run_experiment(chunks1, queries, "512/128")

    # 策略2: 固定窗口 256/64
    print("\n[2/3] 策略2: 固定窗口 256/64...")
    chunks2 = chunk_documents(docs, chunk_size=256, chunk_overlap=64)
    print(f"  切片数: {len(chunks2)}")
    if chunks2:
        avg_len = sum(len(c.page_content) for c in chunks2) / len(chunks2)
        print(f"  平均长度: {avg_len:.0f} 字符")
    strategies["256/64"] = run_experiment(chunks2, queries, "256/64")

    # 策略3: 语义分块
    print("\n[3/3] 策略3: 语义分块（按句子合并到 ~512 字符）...")
    chunks3 = chunk_by_sentences(docs, max_chunk_size=512)
    print(f"  切片数: {len(chunks3)}")
    if chunks3:
        avg_len = sum(len(c.page_content) for c in chunks3) / len(chunks3)
        print(f"  平均长度: {avg_len:.0f} 字符")
    strategies["语义"] = run_experiment(chunks3, queries, "语义")

    # 输出对比报告
    print_report(strategies)


if __name__ == "__main__":
    main()
