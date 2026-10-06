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

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")
MAX_STEPS = 6  # safety limit so the loop can never run forever

TELEGRAM_TOKEN = _clean_env("TELEGRAM_TOKEN")
TELEGRAM_USER_ID = _clean_env("TELEGRAM_USER_ID")
API_KEY = os.getenv("API_KEY", "")  # optional: protects the /chat endpoint

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

SYSTEM_PROMPT = (
    "You are PocketAgent, a helpful personal assistant. "
    "Use your tools when they help. Keep answers short and clear. "
    "When the user tells you a lasting fact or preference about themselves, "
    "call remember_fact. Use notes only for ideas and reminders, and tasks for to-dos. "
    "When answering from documents, say which file the answer came from; "
    "if the documents do not contain the answer, say so and do not guess."
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()  # gemini | ollama | auto
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")  # unload the model from RAM after 10 idle minutes
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "off").lower()  # off = skip reasoning (much faster)