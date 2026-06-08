"""
RAG 主流程编排 —— 串联文档加载 → 切片 → 入库 → 检索 → 生成。

使用方式：
    python -m src.rag_pipeline              # 交互式问答
    python -m src.rag_pipeline --rebuild    # 重建向量库
    python -m src.rag_pipeline --query "什么是机器学习"  # 单次查询
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedding
from src.vector_store import VectorStore
from src.generator import Generator
from src.retriever import BM25Retriever, HybridRetriever


class RAGPipeline:
    """RAG 全流程编排"""

    def __init__(self, rebuild: bool = False):
        self.embedding = get_embedding()
        self.vector_store = VectorStore()
        self.generator = Generator()
        self._chunks = None  # 缓存切片，供 BM25 复用

        # 加载文档 & 切片
        if rebuild or self.vector_store.count() == 0:
            self._build_index()
        else:
            print(f"[Vector Store] Loaded from disk ({self.vector_store.count()} vectors)")
            # 非重建时仍需加载文档给 BM25 用
            self._chunks = self._load_and_chunk()

        # 初始化 BM25（复用 _chunks，避免重复加载）
        self._init_bm25(self._chunks)

        # 混合检索器
        self.hybrid_retriever = HybridRetriever(
            self.vector_store, self.bm25
        )

    def _load_and_chunk(self):
        """加载文档并切片，返回 chunks（供 FAISS 和 BM25 共用）"""
        docs = load_documents()
        if not docs:
            raise ValueError("未找到任何文档！请在 data/docs/ 下放置 PDF/Word/TXT 文件")
        return chunk_documents(docs)

    def _build_index(self):
        """构建向量索引"""
        print("=" * 60)
        print("[Rebuild] Rebuilding index...")
        print("=" * 60)

        self._chunks = self._load_and_chunk()
        self.vector_store.rebuild(self._chunks)

    def _init_bm25(self, chunks):
        """初始化 BM25 检索器"""
        self.bm25 = BM25Retriever()
        if chunks:
            self.bm25.index(chunks)
        else:
            raise RuntimeError("BM25 初始化失败：未提供 chunks")

    def query(self, question: str, mode: str = "dense", top_k: int = None,
              show_prompt: bool = False) -> dict:
        """
        问答接口。

        Args:
            question: 用户问题
            mode: 检索模式 - "dense" / "sparse" / "hybrid" / "rerank" / "hybrid+rerank"
            top_k: 返回文档数
            show_prompt: 是否打印完整 prompt

        Returns:
            dict: {"answer", "sources", "retrieved_docs", ...}
        """
        top_k = top_k or config.TOP_K

        # 检索
        if mode == "dense":
            retrieved = self.vector_store.similarity_search(question, k=top_k)
        elif mode == "sparse":
            results = self.hybrid_retriever.retrieve_sparse(question, k=top_k)
            retrieved = [doc for doc, _ in results]
        elif mode == "hybrid":
            retrieved = self.hybrid_retriever.retrieve_hybrid(question, top_k=top_k)
        elif mode == "rerank":
            retrieved = self.hybrid_retriever.retrieve_with_rerank(question, top_k=top_k)
        elif mode == "hybrid+rerank":
            retrieved = self.hybrid_retriever.retrieve_hybrid_with_rerank(question, top_k=top_k)
        else:
            raise ValueError(f"未知检索模式: {mode}")

        # 生成
        result = self.generator.generate(question, retrieved)

        # 打印结果
        print(f"\n{'='*60}")
        print(f"[Q] {question}")
        print(f"[mode] {mode} | hits: {len(retrieved)}")
        print(f"{'='*60}")
        print(f"\n[A] {result['answer']}")
        print(f"\n[Sources] {', '.join(result['sources'])}")

        if show_prompt:
            print(f"\n{'='*60}")
            print("📋 完整 Prompt:")
            print(f"{'='*60}")
            print(result['prompt'])

        return {
            **result,
            "retrieved_docs": retrieved,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek RAG 知识问答系统")
    parser.add_argument("--rebuild", action="store_true", help="重建向量索引")
    parser.add_argument("--query", "-q", type=str, default=None, help="单次查询")
    parser.add_argument("--mode", "-m", type=str, default="dense",
                        choices=["dense", "sparse", "hybrid", "rerank", "hybrid+rerank"],
                        help="检索模式")
    parser.add_argument("--show-prompt", action="store_true", help="打印完整 prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("DeepSeek RAG 知识问答系统")
    print(f"   模型: {config.DEEPSEEK_MODEL}")
    print(f"   Embedding: {config.EMBEDDING_MODEL}")
    print("=" * 60)

    # 初始化
    rag = RAGPipeline(rebuild=args.rebuild)

    if args.query:
        # 单次查询
        rag.query(args.query, mode=args.mode, show_prompt=args.show_prompt)
    else:
        # 交互式问答
        print("\n输入问题开始对话，输入 'quit' 退出\n")
        while True:
            try:
                question = input("💬 你的问题: ").strip()
                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    print("👋 再见!")
                    break
                rag.query(question, mode=args.mode)
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break


if __name__ == "__main__":
    main()
