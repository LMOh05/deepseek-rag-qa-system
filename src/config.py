"""
配置中心 —— 所有可调参数集中管理，避免魔法数字散落各处。
"""

import os
from pathlib import Path

# ============================================================
# 环境修复（在一切 import 之前执行）
# ============================================================

# 1. 修复 SSL 证书路径（conda 设的可能不存在）
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# 2. HuggingFace 国内镜像（下载模型用）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 3. 终端 UTF-8 编码（避免 Windows GBK 报错）
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    """RAG 系统全局配置"""

    # ========== 路径 ==========
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "docs"
    CHROMA_DIR = PROJECT_ROOT / "chroma_db"
    PROMPTS_DIR = PROJECT_ROOT / "prompts"

    # ========== DeepSeek API ==========
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 2048

    # ========== Embedding ==========
    # 为什么选 text2vec-base-chinese?
    # 1. 专为中文优化，C-MTEB 榜单表现好
    # 2. 本地免费，无需调 API
    # 3. 1024 维 vs m3e 768 维：更高维度理论上表达能力更强
    # 4. 模型体积小 (~400MB)，CPU 上推理速度可接受
    EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
    EMBEDDING_DEVICE = "cpu"  # 无 GPU，CPU 推理

    # ========== 切片策略 ==========
    # 为什么 chunk_size=512?
    # - DeepSeek 上下文窗口大，但检索粒度不宜过粗
    # - 512 tokens ≈ 600-800 中文字，足够承载一个完整段落
    # - 太小会丢失上下文，太大会稀释语义
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128  # 25% 重叠，保证跨切片信息不丢失

    # ========== 检索 ==========
    TOP_K = 5  # 最终返回的文档片段数
    RETRIEVAL_K = 20  # 初检数量（给 rerank 留空间）

    # ========== 混合检索 RRF ==========
    RRF_K = 60  # RRF 平滑因子，经典默认值

    # ========== 重排序 ==========
    # 用 base 版而非 large 版，CPU 可跑、速度快
    RERANK_MODEL = "BAAI/bge-reranker-base"


# 单例
config = Config()
