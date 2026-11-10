from app.agent import ChatMessage, TextPart
from app import sessions


def test_trim_keeps_recent_turns():
    h = []
    for i in range(30):
        h.append(ChatMessage(role="user", parts=[TextPart(text=f"q{i}")]))
        h.append(ChatMessage(role="model", parts=[TextPart(text=f"a{i}")]))
    sessions._trim(h)
    assert h[0].parts[0].text == "q10"
    assert len(h) == 40


def test_chat_reports_local_search_failure(monkeypatch):
    sessions._histories.clear()

    def fail(history):
        raise OSError("index unavailable")

    monkeypatch.setattr(sessions, "run_agent", fail)

    assert sessions.chat("failed-search", "hello") == (
        "I couldn't search the documents right now."
    )
    assert sessions._histories["failed-search"] == []