"""Tools are plain functions, so test them without calling Gemini. Run: pytest"""
import os

os.environ.setdefault("GEMINI_API_KEY", "test")

from app.tools import tasks


def test_task_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "TASKS_FILE", tmp_path / "tasks.json")
    assert "added" in tasks.add_task("buy milk")
    assert "buy milk" in tasks.list_tasks()
    assert "done" in tasks.complete_task(1)
    assert "[x]" in tasks.list_tasks()
