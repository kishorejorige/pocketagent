import os

os.environ.setdefault("GEMINI_API_KEY", "test")

from app.memory import store


def test_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.save_fact("Client is Mike")
    store.save_fact("Client is Mike")  # duplicates are ignored
    assert store.list_facts() == [(1, "Client is Mike")]
    assert store.delete_fact(1) is True
    assert store.list_facts() == []