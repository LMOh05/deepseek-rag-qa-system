"""
向量存储模块 —— 基于 FAISS 的本地向量库。

为什么选 FAISS 而不是 Chroma?
- FAISS: 纯内存/本地文件，不依赖 sqlite3，Windows 兼容性好
- Chroma: 需要 sqlite3 >= 3.35，Windows Python 3.10 自带的不满足
- 这个项目数据量小，FAISS 的本地读写完全够用
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class VectorStore:
    """FAISS 向量库封装"""

    def __init__(self, persist_dir: str = None, embedding=None):
        from src.config import config
        from src.embedder import get_embedding

        self.persist_dir = str(persist_dir or config.FAISS_DIR)
        self.embedding = embedding or get_embedding()
        self._store: Optional[FAISS] = None

    @property
    def store(self) -> Optional[FAISS]:
        if self._store is None:
            if os.path.exists(self.persist_dir) and os.path.isfile(
                os.path.join(self.persist_dir, "index.faiss")
            ):
                self._store = FAISS.load_local(
                    self.persist_dir,
                    self.embedding,
                    allow_dangerous_deserialization=True,
                )
        return self._store

    def add_documents(self, documents: List[Document]) -> None:
        """将文档切片入库"""
        print(f"[FAISS] Writing {len(documents)} chunks...")

        if self._store is None:
            self._store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding,
            )
        else:
            self._store.add_documents(documents)

        self._store.save_local(self.persist_dir)
        print(f"   path: {self.persist_dir}")
        print(f"   done!")

    def similarity_search(
        self, query: str, k: int = 5, with_scores: bool = False
    ):
        """
        向量相似度检索。

        Returns:
            List[Document] 或 List[Tuple[Document, float]]
        """
        if self._store is None:
            raise ValueError("Vector store is empty. Call add_documents() first.")

        if with_scores:
            return self._store.similarity_search_with_score(query, k=k)
        return self._store.similarity_search(query, k=k)

    def rebuild(self, documents: List[Document]) -> None:
        """清空并重建向量库"""
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        self._store = None
        self.add_documents(documents)

    def count(self) -> int:
        """返回库中切片数量"""
        if self._store is None:
            return 0
        return self._store.index.ntotal


if __name__ == "__main__":
    from src.document_loader import load_documents
    from src.chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)

    vs = VectorStore()
    vs.add_documents(chunks)

    results = vs.similarity_search("什么是机器学习", k=3, with_scores=True)
    for doc, score in results:
        print(f"\nscore: {score:.4f} | source: {doc.metadata.get('filename', '?')}")
        print(doc.page_content[:150])
