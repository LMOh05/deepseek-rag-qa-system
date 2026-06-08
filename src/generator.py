"""
生成模块 —— 封装 DeepSeek Chat API，拼接检索结果生成答案。
"""

from typing import List
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Prompt 模板
# ============================================================

# 默认模板：简洁、要求引用来源
DEFAULT_SYSTEM_PROMPT = """你是一个基于文档知识的问答助手。请严格依据以下提供的参考资料回答问题。

规则：
1. 只使用下方【参考资料】中的信息回答，不要编造
2. 如果资料不足以回答，请明确说"根据现有资料无法回答"
3. 回答末尾列出你引用的【资料来源】，格式：[文件名-页码]
4. 回答简洁、结构化，避免冗余

【参考资料】
{context}

【用户问题】
{question}

请回答："""

# 详细版模板（阶段二 Prompt 优化用）
DETAILED_SYSTEM_PROMPT = """你是一个专业的知识问答助手，擅长从文档资料中提取和整合信息。

请按以下步骤思考并回答：
1. 分析用户问题的核心意图
2. 从下方【参考资料】中找出所有相关内容
3. 按逻辑顺序组织回答
4. 确保每个观点都有出处

规则：
- 严格依据参考资料，不得添加外部知识
- 如果资料之间存在矛盾，请指出
- 回答末尾列出所有引用的来源

【参考资料】
{context}

【用户问题】
{question}

请详细回答："""

# 思维链版模板（阶段二 Prompt 优化用）
COT_SYSTEM_PROMPT = """你是一个严谨的知识问答助手。请先逐步推理，再给出最终答案。

步骤：
第一步-理解问题：用自己的话重述用户的问题
第二步-检索信息：从【参考资料】中找出相关片段，标注位置
第三步-推理整合：将信息串联起来，形成完整答案
第四步-输出答案：给出结构化的最终回答

【参考资料】
{context}

【用户问题】
{question}

请按照以上四步回答："""


class Generator:
    """DeepSeek 生成器"""

    def __init__(self, model: str = None, temperature: float = None,
                 system_prompt: str = None):
        from src.config import config

        self.model = model or config.DEEPSEEK_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._llm = ChatDeepSeek(
            model=self.model,
            temperature=self.temperature,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.DEEPSEEK_API_KEY,
            api_base=config.DEEPSEEK_BASE_URL,
        )

    def _build_prompt(self, question: str, docs: List[Document]) -> str:
        """拼接检索到的文档片段为 context"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "未知")
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[片段{i}] 来源: {source} (页码:{page})\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        return self.system_prompt.format(context=context, question=question)

    def generate(self, question: str, retrieved_docs: List[Document]) -> dict:
        """
        基于检索结果生成答案。

        Returns:
            dict: {"answer": str, "sources": List[str], "prompt": str}
        """
        prompt_text = self._build_prompt(question, retrieved_docs)

        response = self._llm.invoke(prompt_text)

        # 提取来源
        sources = list(set(
            f"{d.metadata.get('filename', '?')}-p{d.metadata.get('page', '?')}"
            for d in retrieved_docs
        ))

        return {
            "answer": response.content,
            "sources": sources,
            "prompt": prompt_text,
            "model": self.model,
        }

    def stream_generate(self, question: str, retrieved_docs: List[Document]):
        """流式生成（打字机效果）"""
        prompt_text = self._build_prompt(question, retrieved_docs)
        for chunk in self._llm.stream(prompt_text):
            yield chunk.content


if __name__ == "__main__":
    # 简单测试
    from src.document_loader import load_documents
    from src.chunker import chunk_documents
    from src.vector_store import VectorStore

    print("加载文档...")
    docs = load_documents()
    chunks = chunk_documents(docs)

    vs = VectorStore()
    vs.add_documents(chunks)

    gen = Generator()

    question = "什么是机器学习"
    print(f"\n问题: {question}")
    retrieved = vs.similarity_search(question, k=3)
    result = gen.generate(question, retrieved)

    print(f"\n答案:\n{result['answer']}")
    print(f"\n来源: {result['sources']}")
