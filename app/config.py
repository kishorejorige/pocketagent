import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_STEPS = 6  # safety limit so the loop can never run forever

SYSTEM_PROMPT = (
    "You are PocketAgent, a helpful personal assistant. "
    "Use your tools when they help. Keep answers short and clear."
)
