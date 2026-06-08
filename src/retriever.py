"""
检索模块 —— 向量检索 / BM25 / 混合检索 / 重排序。

这是阶段二的核心模块，分步实现：
1. 向量检索（已有，在 vector_store 中）
2. BM25 稀疏检索
3. 混合检索（RRF 融合）
4. 重排序（bge-reranker）
"""

from typing import List, Tuple
from langchain_core.documents import Document


class BM25Retriever:
    """BM25 稀疏检索器 —— 关键词匹配，补齐向量检索的短板"""

    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        self._bm25 = None
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []

        self.BM25Okapi = BM25Okapi
        self._tokenize = lambda text: list(jieba.cut(text))

    def index(self, documents: List[Document]):
        """构建 BM25 索引"""
        self._documents = documents
        self._tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = self.BM25Okapi(self._tokenized_corpus)
        print(f"[BM25] Index built: {len(documents)} docs")

    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """BM25 检索，返回 (Document, score)"""
        if self._bm25 is None:
            raise ValueError("BM25 索引为空，请先调用 index()")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 取 top-k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:k]:
            if score > 0:
                results.append((self._documents[idx], float(score)))

        return results


def reciprocal_rank_fusion(
    dense_results: List[Document],
    sparse_results: List[Tuple[Document, float]],
    k: int = 60,
    top_k: int = 5,
) -> List[Document]:
    """
    RRF (Reciprocal Rank Fusion) 融合两路检索结果。

    为什么用 RRF 而不是简单加权？
    - RRF 不依赖原始分数的绝对大小，只关注排名
    - 向量相似度和 BM25 分数的量纲不同，直接加权需要调参
    - RRF 的 k 参数（默认60）提供了平滑效果

    Args:
        dense_results: 向量检索结果
        sparse_results: BM25 检索结果 (doc, score)
        k: RRF 平滑因子
        top_k: 最终返回数量

    Returns:
        融合重排后的 Document 列表
    """
    # 用 id(doc) 跟踪每个文档
    doc_map = {}  # page_content hash → Document
    scores = {}   # page_content hash → RRF score

    # 向量检索排名
    for rank, doc in enumerate(dense_results, 1):
        key = doc.page_content[:200]  # 用前200字符作为 key
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank)

    # BM25 检索排名
    for rank, (doc, _) in enumerate(sparse_results, 1):
        key = doc.page_content[:200]
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank)

    # 按 RRF 分数排序
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys[:top_k]]


class HybridRetriever:
    """混合检索器：向量 + BM25 + 可选 Rerank"""

    def __init__(self, vector_store, bm25_retriever: BM25Retriever = None):
        from src.vector_store import VectorStore

        self.vector_store = vector_store
        self.bm25 = bm25_retriever or BM25Retriever()
        self._reranker = None

    def _get_reranker(self):
        """懒加载 reranker 模型"""
        if self._reranker is None:
            from FlagEmbedding import FlagReranker
            from src.config import config
            print(f"[Reranker] Loading: {config.RERANK_MODEL}")
            self._reranker = FlagReranker(
                config.RERANK_MODEL,
                use_fp16=False,  # CPU 上用 fp32
                device="cpu",
            )
        return self._reranker

    def retrieve_dense(self, query: str, k: int = 20) -> List[Document]:
        """纯向量检索"""
        return self.vector_store.similarity_search(query, k=k)

    def retrieve_sparse(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """纯 BM25 检索"""
        return self.bm25.search(query, k=k)

    def retrieve_hybrid(self, query: str, dense_k: int = 20, sparse_k: int = 20,
                        top_k: int = 5) -> List[Document]:
        """混合检索：向量 + BM25 → RRF 融合"""
        dense_results = self.retrieve_dense(query, k=dense_k)
        sparse_results = self.retrieve_sparse(query, k=sparse_k)
        return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)

    def retrieve_with_rerank(self, query: str, dense_k: int = 20,
                             top_k: int = 5) -> List[Document]:
        """向量检索 + 重排序"""
        candidates = self.retrieve_dense(query, k=dense_k)

        if len(candidates) <= top_k:
            return candidates

        # 构建 (query, doc) 对
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self._get_reranker().compute_score(pairs)

        # 按 rerank 分数排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [candidates[idx] for idx, _ in indexed_scores[:top_k]]

    def retrieve_hybrid_with_rerank(self, query: str, dense_k: int = 20,
                                    sparse_k: int = 20, top_k: int = 5) -> List[Document]:
        """混合检索 + 重排序：全链路最强配置"""
        # 先混合检索拿候选
        candidates = self.retrieve_hybrid(query, dense_k, sparse_k, top_k=20)

        if len(candidates) <= top_k:
            return candidates

        # Rerank
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self._get_reranker().compute_score(pairs)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [candidates[idx] for idx, _ in indexed_scores[:top_k]]


if __name__ == "__main__":
    print("测试检索模块需要先加载文档和向量库")
    print("请运行 src/rag_pipeline.py 进行完整测试")
