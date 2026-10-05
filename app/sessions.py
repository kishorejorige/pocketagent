import threading

from google.genai import errors, types

from app.agent import run_agent

MAX_TURNS = 20  # keep the last 20 user messages (and everything after them)
_histories: dict[str, list] = {}
_lock = threading.Lock()  # one request at a time is fine for a personal agent


def _trim(history: list) -> None:
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
        history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        try:
            reply = run_agent(history)
        except errors.APIError as e:
            del history[start:]  # drop the half-finished turn
            return f"Gemini is unavailable right now ({e.code}). Try again in a minute."
        _trim(history)
        return reply or "(no reply)"