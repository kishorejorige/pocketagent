import json

from app.config import DATA_DIR

TASKS_FILE = DATA_DIR / "tasks.json"


def _load() -> list:
    if not TASKS_FILE.exists():
        return []
    return json.loads(TASKS_FILE.read_text(encoding="utf-8"))


def _save(tasks: list) -> None:
    TASKS_FILE.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def add_task(title: str) -> str:
    """Add a task to the user's to-do list.

    Args:
        title: Short description of the task.
    """
    tasks = _load()
    task_id = max((t["id"] for t in tasks), default=0) + 1
    tasks.append({"id": task_id, "title": title, "done": False})
    _save(tasks)
    return f"Task {task_id} added: {title}"


def list_tasks() -> str:
    """List all tasks on the user's to-do list with their status."""
    tasks = _load()
    if not tasks:
        return "No tasks yet."
    return "\n".join(
        f"{t['id']}. [{'x' if t['done'] else ' '}] {t['title']}" for t in tasks
    )


def complete_task(task_id: int) -> str:
    """Mark a task as done.

    Args:
        task_id: The number of the task to complete.
    """
    tasks = _load()
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
            _save(tasks)
            return f"Task {task_id} marked done."
    return f"No task with id {task_id}."
