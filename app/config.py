import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _clean_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

TELEGRAM_TOKEN = _clean_env("TELEGRAM_TOKEN")
TELEGRAM_USER_ID = _clean_env("TELEGRAM_USER_ID")
API_KEY = os.getenv("API_KEY", "")  # optional: protects the /chat endpoint

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
