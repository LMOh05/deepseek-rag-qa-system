# DeepSeek RAG + Agent 智能知识问答系统

## 项目简介

基于 DeepSeek v4-flash 的 **RAG + Agent** 智能知识问答系统。支持私域文档的智能问答、Agent 自主决策多工具调用、以及 LoRA 微调优化回答质量。

> 四阶段全流程：基础搭建 → 检索优化 → 量化评测 → LoRA 微调 → Agent 模块。

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | DeepSeek v4-flash | 1M上下文，\$0.14/1M tokens |
| Agent | LangChain Agent | Tool-calling，自主决策工具调用 |
| 框架 | LangChain 1.3 | RAG + Agent 流程编排 |
| Embedding | text2vec-base-chinese | 768维，本地免费 |
| 向量库 | FAISS | 本地持久化，Windows兼容好 |
| 稀疏检索 | BM25 (rank-bm25 + jieba) | 关键词匹配 |
| 重排序 | bge-reranker-base | 提升检索精度 |
| 微调 | Qwen2.5-1.5B + LoRA | Colab T4，loss=1.80，已落地 |

## 快速开始

### 1. 环境准备

```bash
cd D:\python\deepseek-rag-qa-system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key。

### 3. 准备文档

将 PDF/Word/TXT 文档放入 `data/docs/` 目录。

### 4. 运行

```bash
# RAG 管道（基础问答）
python -m src.rag_pipeline

# Agent 模式（自主决策工具调用）
python src/agent.py
python src/agent.py --query "什么是机器学习"

# 检索模式
python -m src.rag_pipeline --mode hybrid --query "RAG的工作流程"
python -m src.rag_pipeline --mode hybrid+rerank --query "什么是词嵌入"

# 运行实验
python experiments/chunk_strategy_compare.py
python experiments/hybrid_search_compare.py

# 评测
python src/evaluator.py
```

### 5. Agent 工具

| 工具 | 功能 | 触发条件 |
|------|------|---------|
| search_rag_docs | 搜索本地技术文档 | 问机器学习/RAG/NLP等技术概念 |
| search_web | 互联网搜索 | 问最新信息（需安装 duckduckgo-search） |
| answer_from_knowledge | LLM 内置知识回答 | 常识问题、闲聊 |

## 项目结构

```
├── data/docs/                # 原始文档
├── src/
│   ├── config.py             # 配置中心（SSL/HF/UTF-8 自动修复）
│   ├── document_loader.py    # PDF/Word/TXT 解析
│   ├── chunker.py            # 切片策略（固定窗口+语义分块）
│   ├── embedder.py           # Embedding（text2vec-base-chinese）
│   ├── vector_store.py       # FAISS 向量库
│   ├── retriever.py          # 检索（向量/BM25/混合RRF/Rerank）
│   ├── generator.py          # DeepSeek 生成（3套Prompt）
│   ├── rag_pipeline.py       # RAG 主流程 CLI
│   ├── agent.py              # Agent 层（工具调用+自主决策）
│   └── evaluator.py          # 评测（Recall@K/MRR）
├── experiments/              # 对比实验脚本
├── notebooks/                # Colab LoRA 微调 Notebook
├── models/                   # LoRA 微调权重
│   └── qwen25-rag-lora3/     # 训练好的 LoRA adapter
├── prompts/                  # Prompt 模板
├── tests/                    # 50条测试集 + 评测结果
├── 实验验证文件.md            # 完整实验报告
└── 项目计划.md               # 实施计划
```

## 评测结果

| 配置 | Recall@3 | Recall@5 | MRR |
|------|----------|----------|-----|
| 语义分块+混合检索 ✅ | **100.00%** | **100.00%** | **0.9767** |

> 50 条测试集，核心 43 题，0 Bad Case，边缘 case 全部正确处理。

## 微调效果

| 测试 | DeepSeek v4-flash | Qwen2.5 原版 | Qwen2.5 + LoRA |
|------|:--:|:--:|:--:|
| 基于资料回答 | ✅ | ❌ | ✅ |
| 信息不足拒绝 | ✅ | ❌ | ✅ |
| 多信息综合 | ✅ | ❌ | ✅ |

## License

MIT
