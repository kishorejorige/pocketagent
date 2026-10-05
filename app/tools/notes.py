from datetime import datetime

from app.config import DATA_DIR

NOTES_FILE = DATA_DIR / "notes.txt"


def add_note(text: str) -> str:
    """Save a note or idea for the user.

    Args:
        text: The note text to save.
    """
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='minutes')} | {text}\n")
    return "Note saved."


def list_notes(limit: int = 10) -> str:
    """List the user's most recent saved notes.

    Args:
        limit: How many recent notes to return.
    """
    if not NOTES_FILE.exists():
        return "No notes yet."
    lines = NOTES_FILE.read_text(encoding="utf-8").strip().splitlines()
    return "\n".join(lines[-limit:]) or "No notes yet."
