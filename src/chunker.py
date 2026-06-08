"""
切片策略模块 —— 将文档切成适合检索的片段。

初始实现：固定窗口切片（chunk_size=512, overlap=128）
阶段二会扩展：语义分块、递归分割等策略对比。
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: List[str] = None,
) -> List[Document]:
    """
    使用递归字符分割器将文档列表切分成小片段。

    Args:
        documents: 原始文档列表
        chunk_size: 切片大小（tokens），默认从 config 读取
        chunk_overlap: 重叠大小（tokens），默认从 config 读取
        separators: 自定义分隔符优先级，中文场景优先按句子边界切

    Returns:
        切片后的 Document 列表
    """
    from src.config import config

    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    if separators is None:
        # 中文友好的分隔符优先级：段落 → 句子 → 短句 → 字
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # 合并过小的相邻切片（同一文件内），避免无意义的碎片
    MIN_CHUNK_CHARS = 200
    merged = []
    for chunk in chunks:
        if (merged
                and len(chunk.page_content) < MIN_CHUNK_CHARS
                and chunk.metadata.get("filename") == merged[-1].metadata.get("filename")):
            # 合并到前一个切片
            merged[-1].page_content += "\n" + chunk.page_content
        else:
            merged.append(chunk)

    # 打印切片统计
    print(f"\n[Chunk] Done:")
    print(f"   原始文档段落: {len(documents)}")
    print(f"   切片后片段:   {len(merged)}")
    if merged:
        avg_len = sum(len(c.page_content) for c in merged) / len(merged)
        min_len = min(len(c.page_content) for c in merged)
        max_len = max(len(c.page_content) for c in merged)
        print(f"   平均长度: {avg_len:.0f} 字符 | 最短: {min_len} | 最长: {max_len}")

    return merged


def chunk_by_sentences(
    documents: List[Document],
    max_chunk_size: int = 512,
) -> List[Document]:
    """
    语义分块：按句子边界合并，尽量保持语义完整性。
    阶段二实验用。
    """
    import re

    # 中文句子分割正则
    sentence_pattern = re.compile(r'[^。！？；\n]+[。！？；\n]?')

    chunks = []
    for doc in documents:
        sentences = sentence_pattern.findall(doc.page_content)
        if not sentences:
            chunks.append(doc)
            continue

        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) <= max_chunk_size:
                current_chunk += sent
            else:
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk,
                        metadata=doc.metadata.copy()
                    ))
                current_chunk = sent

        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk,
                metadata=doc.metadata.copy()
            ))

    print(f"\n[Semantic Chunk] Done: {len(documents)} -> {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    from src.document_loader import load_documents
    docs = load_documents()
    chunks = chunk_documents(docs)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- 切片 {i+1} | {chunk.metadata.get('filename', '?')} ---")
        print(chunk.page_content[:200])
