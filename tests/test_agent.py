from google.genai import types

from app import agent
from app.memory import rag


def _history(*questions):
    return [
        types.Content(role="user", parts=[types.Part(text=question)])
        for question in questions
    ]


def test_agent_answers_from_local_documents_and_saves_history(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    monkeypatch.setattr(rag, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(rag, "INDEX_FILE", tmp_path / "rag_index.json")
    (docs_dir / "faq.txt").write_text(
        "Opening hours: Monday to Friday 7:30 AM to 6:00 PM.",
        encoding="utf-8",
    )
    history = _history("When does the cafe open?")

    reply = agent.run_agent(history)

    assert "faq.txt" in reply
    assert "7:30 AM to 6:00 PM" in reply
    assert history[-1].role == "model"
    assert history[-1].parts[0].text == reply


def test_agent_uses_previous_question_for_follow_up(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    monkeypatch.setattr(rag, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(rag, "INDEX_FILE", tmp_path / "rag_index.json")
    (docs_dir / "faq.txt").write_text(
        "Opening hours: Monday to Friday 7:30 AM to 6:00 PM. "
        "Saturday 8:00 AM to 4:00 PM.",
        encoding="utf-8",
    )
    history = _history("When are opening hours?", "And Saturdays?")

    reply = agent.run_agent(history)

    assert "Saturday 8:00 AM to 4:00 PM" in reply


def test_agent_lists_documents_without_an_external_model(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.txt").write_text("Local guide", encoding="utf-8")
    monkeypatch.setattr(rag, "DOCS_DIR", docs_dir)
    history = _history("What documents do you have?")

    reply = agent.run_agent(history)

    assert "guide.txt" in reply


def test_agent_without_user_message_returns_local_guidance():
    assert "local documents" in agent.run_agent([])


def test_agent_routes_question_to_local_rag(monkeypatch):
    history = _history("question from the user")
    called = []

    def local_answer(question):
        called.append(question)
        return "local answer"

    monkeypatch.setattr(agent.rag, "answer", local_answer)

    assert agent.run_agent(history) == "local answer"
    assert called == ["question from the user"]
