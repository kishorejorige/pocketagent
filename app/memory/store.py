import sqlite3
from contextlib import closing

from app.config import DATA_DIR

DB_PATH = DATA_DIR / "pocketagent.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL UNIQUE, "
        "created TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


def save_fact(text: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("INSERT OR IGNORE INTO facts (text) VALUES (?)", (text,))


def list_facts() -> list[tuple[int, str]]:
    with closing(_connect()) as conn:
        return conn.execute("SELECT id, text FROM facts ORDER BY id").fetchall()


def delete_fact(fact_id: int) -> bool:
    with closing(_connect()) as conn, conn:
        cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0