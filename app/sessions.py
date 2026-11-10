import logging
import threading

from app.agent import ChatMessage, TextPart, run_agent

MAX_TURNS = 20  # keep the last 20 user messages (and everything after them)
_histories: dict[str, list[ChatMessage]] = {}
_lock = threading.Lock()  # one request at a time is fine for a personal agent
logger = logging.getLogger(__name__)


def _trim(history: list[ChatMessage]) -> None:
    starts = [
        i for i, c in enumerate(history)
        if c.role == "user" and any(p.text for p in c.parts)
    ]
    if len(starts) > MAX_TURNS:
        del history[: starts[-MAX_TURNS]]


def chat(session_id: str, text: str) -> str:
    with _lock:
        history = _histories.setdefault(session_id, [])
        start = len(history)
        history.append(ChatMessage(role="user", parts=[TextPart(text=text)]))
        try:
            reply = run_agent(history)
            _trim(history)

        except Exception:
            del history[start:]  # drop the half-finished turn
            logger.exception("Local document search failed")
            return "I couldn't search the documents right now."

        return reply or "(no reply)"