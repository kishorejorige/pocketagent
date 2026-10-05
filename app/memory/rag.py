"""Local RAG: chunk documents, embed them with Ollama, search by cosine similarity.
Run `python -m app.memory.rag` to (re)build the index by hand."""
import json
import math
import urllib.request

from app import config
from app.config import DATA_DIR

DOCS_DIR = DATA_DIR / "docs"
INDEX_FILE = DATA_DIR / "rag_index.json"
EXTENSIONS = {".txt", ".md"}


def embed(texts: list[str], prefix: str) -> list[list[float]]:
    """Get embeddings from Ollama. nomic-embed-text expects a task prefix."""
    body = json.dumps(
        {"model": config.EMBED_MODEL, "input": [prefix + t for t in texts]}
    ).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["embeddings"]


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks of at most `size` characters."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:  # a single very long paragraph
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = p
    if cur:
        chunks.append(cur)
    return chunks


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _load() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return {}


def build_index() -> str:
    """Index new or changed files and drop deleted ones. Cheap when nothing changed."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    index = _load()
    files = {
        f.name: f for f in DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    }
    changed = 0
    for name, path in files.items():
        mtime = path.stat().st_mtime
        if index.get(name, {}).get("mtime") == mtime:
            continue
        chunks = chunk_text(path.read_text(encoding="utf-8", errors="ignore"))
        vecs: list[list[float]] = []
        for i in range(0, len(chunks), 16):
            vecs += embed(chunks[i:i + 16], "search_document: ")
        index[name] = {
            "mtime": mtime,
            "chunks": [
                {"text": t, "vec": [round(x, 5) for x in v]}
                for t, v in zip(chunks, vecs)
            ],
        }
        changed += 1
    for name in list(index):
        if name not in files:
            del index[name]
            changed += 1
    if changed:
        INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    total = sum(len(d["chunks"]) for d in index.values())
    return f"{len(index)} documents, {total} chunks ({changed} updated)"


def search(query: str, k: int = 4) -> list[tuple[float, str, str]]:
    index = _load()
    items = [(name, c) for name, d in index.items() for c in d["chunks"]]
    if not items:
        return []
    q = embed([query], "search_query: ")[0]
    scored = [(_cos(q, c["vec"]), name, c["text"]) for name, c in items]
    return sorted(scored, reverse=True)[:k]


if __name__ == "__main__":
    print(build_index())