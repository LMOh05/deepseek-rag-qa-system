"""
Agent 模块 —— 在 RAG 管道之上叠加自主决策层。

Agent 自主决定调用哪个工具：
- search_rag_docs: 本地文档检索（机器学习/RAG/NLP 等技术问题）
- search_web: 互联网搜索（最新信息，可选安装 duckduckgo-search）
- 直接回答: LLM 内置知识回答（常识/闲聊）

使用：
    python src/agent.py --query "什么是机器学习"
    python src/agent.py  # 交互式
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.rag_pipeline import RAGPipeline
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# 全局 RAG 实例
# ============================================================
_rag: RAGPipeline = None

def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


# ============================================================
# 工具定义
# ============================================================
@tool
def search_rag_docs(query: str) -> str:
    """在本地技术文档中搜索。适用于机器学习、RAG、NLP、深度学习等技术概念问题。"""
    try:
        rag = get_rag()
        result = rag.query(query, mode="hybrid", top_k=3)
        docs_text = ""
        for i, doc in enumerate(result["retrieved_docs"], 1):
            src = doc.metadata.get("filename", "?")
            docs_text += f"\n[文档{i} - {src}]\n{doc.page_content[:500]}"
        return (
            f"检索到的文档：{docs_text}\n\n"
            f"基于文档的回答：{result['answer']}\n"
            f"来源: {', '.join(result['sources'])}"
        )
    except Exception as e:
        return f"RAG 检索失败: {e}"


@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息。适用于新闻、实时数据、文档中没有的内容。"""
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return "未找到相关结果。"
        lines = [f"[{i+1}] {r['title']}\n{r['body'][:300]}\n{r['href']}" for i, r in enumerate(results)]
        return "网络搜索结果：\n\n" + "\n\n".join(lines)
    except ImportError:
        return "Web 搜索未安装(pip install duckduckgo-search)。建议用 search_rag_docs 查文档。"
    except Exception as e:
        return f"搜索出错: {e}"


# ============================================================
# Agent 核心逻辑（Tool-calling，不依赖 langchain.agents）
# ============================================================
AGENT_SYSTEM = """你是一个智能知识助手，可以调用工具来回答问题。

你的工具：
1. search_rag_docs(query) - 搜索本地技术文档（问机器学习/RAG/NLP 时优先用这个）
2. search_web(query) - 搜索互联网最新信息

决策规则：
- 用户问技术概念 → 调用 search_rag_docs
- 用户问最新消息/文档没有的话题 → 调用 search_web
- 问候/闲聊/常识 → 直接回答，不调工具
- 复杂问题可以先后调用多个工具

始终标注信息来源。如果信息来自文档，注明文件名。"""


class RAGAgent:
    """RAG + Agent 智能助手"""

    def __init__(self):
        # 绕过系统代理（和 config.py 中 SSL 修复配合）
        import os
        os.environ.setdefault("no_proxy", "*")
        os.environ.setdefault("NO_PROXY", "*")

        self._tools = {"search_rag_docs": search_rag_docs, "search_web": search_web}
        self._llm = ChatDeepSeek(
            model=config.DEEPSEEK_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.DEEPSEEK_API_KEY,
            api_base=config.DEEPSEEK_BASE_URL,
        )
        self._llm_with_tools = self._llm.bind_tools(list(self._tools.values()))
        self._history = []

    def query(self, question: str, verbose: bool = True) -> dict:
        """Agent 问答入口"""
        messages = [SystemMessage(content=AGENT_SYSTEM)]

        # 加入历史（最近 6 轮）
        for role, content in self._history[-6:]:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(SystemMessage(content=content))

        messages.append(HumanMessage(content=question))

        # 第一轮：LLM 决定是否调工具
        response = self._llm_with_tools.invoke(messages)
        tools_used = []

        # 如果 LLM 决定调工具
        if hasattr(response, "tool_calls") and response.tool_calls:
            # 保留 LLM 的 tool_call 响应
            messages.append(response)

            tool_results = []
            for call in response.tool_calls:
                name = call["name"]
                args = call.get("args", {})
                query_arg = args.get("query", args.get("__arg1", str(args)))

                if verbose:
                    print(f"  [Tool] {name}: {str(query_arg)[:60]}...")

                tool_fn = self._tools.get(name)
                if tool_fn:
                    result = tool_fn.invoke(args)
                    tools_used.append({"tool": name, "input": str(query_arg)[:100]})
                    tool_results.append(f"[工具 {name} 结果]\n{result}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

            # 第二轮：LLM 基于工具结果生成最终回答（用不带 tools 的 LLM）
            final_response = self._llm.invoke(messages)
            answer = final_response.content
        else:
            answer = response.content

        # 保存历史
        self._history.append(("user", question))
        self._history.append(("assistant", answer))

        return {
            "answer": answer,
            "tools_used": tools_used,
        }

    def clear_history(self):
        self._history = []


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Agent 智能知识助手")
    parser.add_argument("--query", "-q", type=str, help="单次查询")
    parser.add_argument("--no-verbose", action="store_true", help="隐藏工具调用日志")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Agent 智能知识助手")
    print(f"   LLM: {config.DEEPSEEK_MODEL}")
    print(f"   工具: RAG文档检索 | Web搜索 | 直接回答")
    print("=" * 60)

    agent = RAGAgent()

    if args.query:
        result = agent.query(args.query, verbose=not args.no_verbose)
        print(f"\n{result['answer']}")
        if result["tools_used"]:
            print(f"\n--- 使用了工具: {[t['tool'] for t in result['tools_used']]}")
    else:
        print("\n问题用 'quit' 退出, 'clear' 清空历史\n")
        while True:
            try:
                question = input("You: ").strip()
                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    break
                if question.lower() == "clear":
                    agent.clear_history()
                    print("[History cleared]")
                    continue
                result = agent.query(question, verbose=not args.no_verbose)
                print(f"\nAgent: {result['answer']}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
