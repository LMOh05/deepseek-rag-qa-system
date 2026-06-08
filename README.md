# DeepSeek RAG 私域知识问答系统

## 项目简介

基于 DeepSeek v4-flash API 的 RAG（检索增强生成）知识问答系统，支持私域文档（PDF/Word/TXT）的智能问答。

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | DeepSeek v4-flash | 1M上下文，$0.14/1M tokens |
| 框架 | LangChain 1.3 | RAG 流程编排 |
| Embedding | text2vec-base-chinese | 768维，本地免费 |
| 向量库 | FAISS | 本地持久化，Windows兼容好 |
| 稀疏检索 | BM25 (rank-bm25) | 关键词匹配 |
| 重排序 | bge-reranker-base | 提升检索精度 |

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key。

### 3. 准备文档

将你的 PDF/Word/TXT 文档放入 `data/docs/` 目录。

### 4. 运行

```bash
# 首次运行（重建索引）
python -m src.rag_pipeline --rebuild

# 后续运行（加载已有索引）
python -m src.rag_pipeline

# 单次查询
python -m src.rag_pipeline --query "什么是机器学习"

# 使用混合检索
python -m src.rag_pipeline --mode hybrid --query "什么是机器学习"
```

### 5. 检索模式

| 模式 | 命令参数 | 说明 |
|------|----------|------|
| 纯向量 | `--mode dense` | 余弦相似度检索 |
| 纯BM25 | `--mode sparse` | 关键词匹配 |
| 混合检索 | `--mode hybrid` | 向量+BM25 RRF融合 |
| 向量+Rerank | `--mode rerank` | 粗检+重排序 |
| 混合+Rerank | `--mode hybrid+rerank` | 最强配置 |

## 项目结构

```
├── data/docs/           # 文档存放
├── src/
│   ├── config.py        # 配置中心（自动修复SSL/HF镜像）
│   ├── document_loader.py # 文档解析（PDF/Word/TXT）
│   ├── chunker.py       # 切片策略（固定窗口/语义分块）
│   ├── embedder.py      # Embedding（text2vec-base-chinese）
│   ├── vector_store.py  # FAISS 向量库
│   ├── retriever.py     # 检索（向量/BM25/混合/Rerank）
│   ├── generator.py     # DeepSeek 生成（3套Prompt模板）
│   └── rag_pipeline.py  # 主流程编排
├── experiments/         # 对比实验
├── prompts/             # Prompt 模板
├── tests/               # 测试集
├── notebooks/           # Jupyter 笔记
├── faiss_index/         # 向量索引（自动生成）
├── requirements.txt
└── .env                 # API Key（不提交git）
```

## 踩坑记录

1. **LangChain 1.x 导入路径大改**：`langchain.schema` → `langchain_core.documents`
2. **Windows SSL 证书问题**：conda 设置的 `SSL_CERT_FILE` 路径不存在，需用 `certifi` 覆盖
3. **HuggingFace 国内访问**：需设 `HF_ENDPOINT=https://hf-mirror.com`
4. **Chroma → FAISS**：Windows Python 3.10 的 sqlite3 版本过低，换 FAISS 解决
5. **GBK 编码**：`sys.stdout.reconfigure(encoding='utf-8')` 解决中文乱码

## License

MIT
