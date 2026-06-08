"""
Embedding 模块 —— 封装 sentence-transformers，提供本地向量化能力。

选型记录:
- text2vec-base-chinese: 1024维, C-MTEB中文榜单表现优秀
- m3e-base: 768维, 社区活跃但维度较低
- 选择 text2vec 的理由:
  1. 更高维度 → 理论上更强的语义区分能力
  2. 中文检索场景 (Retrieval) 指标优于 m3e
  3. 本地免费, 不依赖外部 API
"""

from typing import List
from langchain_core.embeddings import Embeddings


class ChineseEmbeddings(Embeddings):
    """LangChain 兼容的中文 Embedding 封装"""

    def __init__(self, model_name: str = None, device: str = None):
        from src.config import config
        from sentence_transformers import SentenceTransformer

        model_name = model_name or config.EMBEDDING_MODEL
        device = device or config.EMBEDDING_DEVICE

        print(f"[Embedding] Loading: {model_name} (device={device})")
        self._model = SentenceTransformer(model_name, device=device)
        # 兼容新旧版本 API
        try:
            self._dim = self._model.get_embedding_dimension()
        except AttributeError:
            self._dim = self._model.get_sentence_embedding_dimension()
        print(f"   向量维度: {self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档"""
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化，余弦相似度 = 内积
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """向量化查询"""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()


# 全局单例，避免重复加载模型
_embedding_instance = None


def get_embedding() -> ChineseEmbeddings:
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = ChineseEmbeddings()
    return _embedding_instance


if __name__ == "__main__":
    emb = get_embedding()
    # 快速测试
    test_texts = ["人工智能是计算机科学的一个分支", "今天天气真好"]
    vecs = emb.embed_documents(test_texts)
    print(f"文本数: {len(vecs)}, 向量维度: {len(vecs[0])}")

    # 测试语义相似度（余弦相似度，归一化后 = 内积）
    query_vec = emb.embed_query("什么是AI")
    for text, vec in zip(test_texts, vecs):
        sim = sum(a * b for a, b in zip(query_vec, vec))
        print(f"  query vs '{text[:20]}...' → 相似度 {sim:.4f}")
