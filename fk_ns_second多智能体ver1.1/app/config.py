"""项目配置 - 从 .env 文件加载 API 密钥和模型配置"""

import os
from dotenv import load_dotenv

# 加载 .env
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# 清除可能干扰的环境变量
os.environ.pop("OPENAI_API_KEY", None)

# ---------- API 密钥 ----------
ZHIPU_API_KEY = os.getenv("SF_API_KEY", "")
ZHIPU_BASE_URL = os.getenv("SF_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
# 确保 base_url 以 / 结尾
if not ZHIPU_BASE_URL.endswith("/"):
    ZHIPU_BASE_URL += "/"

VISION_MODEL = os.getenv("LLM_MODEL1", "glm-4.6v")
TEXT_MODEL = os.getenv("LLM_MODEL2", "glm-4.5-air")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

EMBEDDING_API_KEY = os.getenv(
    "EMBEDDING_API_KEY",
    "sk-ubocxkwnpwlhjnyhkjtwrsfssaswbelgicevuobatiqdjthh",
)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
VECTOR_INDEX_LIMIT = int(os.getenv("VECTOR_INDEX_LIMIT", "1000"))

# ---------- 路径 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "fake_news.db")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "static", "vector_store", "news_embeddings.json")
SOURCE_CREDIBILITY_KB_PATH = os.path.join(PROJECT_ROOT, "static", "vector_store", "source_credibility_kb.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
