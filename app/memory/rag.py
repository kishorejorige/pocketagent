"""Chunk documents, embed them with Gemini or Ollama, and search by cosine similarity.
Run `python -m app.memory.rag` to (re)build the index by hand."""
import json
import math
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types

from app import config
from app.config import DATA_DIR

DOCS_DIR = DATA_DIR / "docs"
INDEX_FILE = DATA_DIR / "rag_index.json"
INDEX_VERSION = 4
EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 16
EMBED_BATCH_DELAY = 1
EMBED_RETRIES = 5
RETRYABLE_CODES = (429, 500, 503)
EMBEDDING_ERRORS = (errors.APIError, httpx.HTTPError, OSError, ValueError, KeyError)


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    provider: str
    model: str
    vector_size: int


def load_text(path: Path) -> str:
    """Load text content from a .txt, .md, .pdf, or .docx file."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(path))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(t.strip() for t in text_parts if t.strip())
        if not full_text:
            return "no text found, it may be a scanned PDF; OCR is not supported"
        return full_text
    elif ext == ".docx":
        import docx
        doc = docx.Document(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _validate_vectors(vectors: list[list[float]], count: int) -> list[list[float]]:
    if len(vectors) != count or not vectors or not vectors[0]:
        raise ValueError("Embedding provider returned an incomplete response.")
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("Embedding provider returned inconsistent vector sizes.")
    return vectors


def _gemini_batch(texts: list[str], prefix: str) -> list[list[float]]:
    task_type = "RETRIEVAL_QUERY" if prefix == "search_query: " else "RETRIEVAL_DOCUMENT"
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    delay = 2
    for attempt in range(EMBED_RETRIES):
        try:
            response = client.models.embed_content(
                model=config.GEMINI_EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors = [
                list(item.values or []) for item in (response.embeddings or [])
            ]
            return _validate_vectors(vectors, len(texts))
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == EMBED_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Gemini embedding retries were exhausted.")


def _ollama_batch(texts: list[str], prefix: str) -> list[list[float]]:
    body = json.dumps(
        {"model": config.EMBED_MODEL, "input": [prefix + t for t in texts]}
    ).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        vectors = json.load(r)["embeddings"]
    return _validate_vectors(vectors, len(texts))


def _embed_batches(
    texts: list[str], prefix: str, provider: str
) -> EmbeddingResult:
    model = config.GEMINI_EMBED_MODEL if provider == "gemini" else config.EMBED_MODEL
    vectors: list[list[float]] = []
    vector_size = 0
    for offset in range(0, len(texts), EMBED_BATCH_SIZE):
        if offset and provider == "gemini":
            time.sleep(EMBED_BATCH_DELAY)
        batch = texts[offset:offset + EMBED_BATCH_SIZE]
        batch_vectors = (
            _gemini_batch(batch, prefix)
            if provider == "gemini"
            else _ollama_batch(batch, prefix)
        )
        if vector_size and any(len(vector) != vector_size for vector in batch_vectors):
            raise ValueError("Embedding provider returned inconsistent vector sizes.")
        vector_size = len(batch_vectors[0])
        vectors.extend(batch_vectors)
    return EmbeddingResult(vectors, provider, model, vector_size)


def embed(
    texts: list[str], prefix: str, provider: str | None = None
) -> EmbeddingResult:
    """Create a consistent set of vectors, preferring Gemini in auto mode."""
    selected = provider or config.EMBEDDING_PROVIDER
    if selected not in {"auto", "gemini", "ollama"}:
        raise ValueError("EMBEDDING_PROVIDER must be gemini, ollama, or auto.")
    if selected == "auto":
        try:
            return _embed_batches(texts, prefix, "gemini")
        except EMBEDDING_ERRORS as gemini_error:
            try:
                return _embed_batches(texts, prefix, "ollama")
            except EMBEDDING_ERRORS as ollama_error:
                raise RuntimeError(
                    "No embedding provider is available. Set GEMINI_API_KEY and check "
                    "GEMINI_EMBED_MODEL, or start Ollama and check OLLAMA_URL. "
                    f"(Gemini: {type(gemini_error).__name__}; "
                    f"Ollama: {type(ollama_error).__name__})"
                ) from ollama_error
    try:
        return _embed_batches(texts, prefix, selected)
    except EMBEDDING_ERRORS as e:
        raise RuntimeError(
            f"{selected.title()} embeddings failed ({type(e).__name__}). Set "
            "GEMINI_API_KEY or start Ollama, or change EMBEDDING_PROVIDER."
        ) from e


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


def _section_prefix(heading_path: list[str], table_header: str = "") -> str:
    parts = []
    if heading_path:
        parts.append(f"Section path: {' > '.join(heading_path)}")
    if table_header:
        parts.append(f"Table header: {table_header}")
    return "\n".join(parts) + ("\n" if parts else "")


def _prefix_chunks(
    text: str,
    heading_path: list[str],
    table_header: str = "",
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    prefix = _section_prefix(heading_path, table_header)
    if len(prefix) >= size:
        prefix = prefix[:size - 1] + "\n"
    content_size = size - len(prefix)
    content_chunks = chunk_text(
        text,
        size=content_size,
        overlap=min(overlap, max(0, content_size - 1)),
    )
    return [prefix + chunk for chunk in content_chunks]


def _markdown_chunks(text: str) -> list[str]:
    heading_path: list[str] = []
    blocks: list[str] = []
    chunks: list[str] = []

    def flush() -> None:
        if blocks:
            chunks.extend(_prefix_chunks("\n\n".join(blocks), heading_path))
            blocks.clear()

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            heading_path[:] = heading_path[:level - 1]
            heading_path.append(heading.group(2).strip())
        elif line.strip():
            blocks.append(line.rstrip())
        else:
            flush()
    flush()
    return chunks


def _docx_chunks(path: Path) -> list[str]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    heading_path: list[str] = []
    blocks: list[str] = []
    chunks: list[str] = []

    def flush() -> None:
        if blocks:
            chunks.extend(_prefix_chunks("\n\n".join(blocks), heading_path))
            blocks.clear()

    for element in document.element.body.iterchildren():
        if element.tag.endswith("}p"):
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()
            if not text:
                continue
            match = re.fullmatch(r"Heading\s+(\d+)", paragraph.style.name or "")
            if match:
                flush()
                level = int(match.group(1))
                heading_path[:] = heading_path[:level - 1]
                heading_path.append(text)
            else:
                blocks.append(text)
        elif element.tag.endswith("}tbl"):
            flush()
            table = Table(element, document)
            if not table.rows:
                continue
            header = " | ".join(cell.text.strip() for cell in table.rows[0].cells)
            for row in table.rows[1:]:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                chunks.extend(
                    _prefix_chunks(
                        row_text,
                        heading_path,
                        table_header=header,
                    )
                )
    flush()
    return chunks


def chunk_document(path: Path) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".md":
        return _markdown_chunks(load_text(path))
    if ext == ".docx":
        return _docx_chunks(path)
    return _prefix_chunks(load_text(path), [])


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _load() -> dict:
    if INDEX_FILE.exists():
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("documents"), dict):
                return data
        except Exception:
            pass
    return {}


def _metadata_matches(data: dict, provider: str | None = None) -> bool:
    metadata = data.get("embedding")
    if not isinstance(metadata, dict):
        return False
    actual_provider = metadata.get("provider")
    if actual_provider not in {"gemini", "ollama"}:
        return False
    if provider and actual_provider != provider:
        return False
    if (
        config.EMBEDDING_PROVIDER != "auto"
        and actual_provider != config.EMBEDDING_PROVIDER
    ):
        return False
    expected_model = (
        config.GEMINI_EMBED_MODEL if actual_provider == "gemini" else config.EMBED_MODEL
    )
    if metadata.get("model") != expected_model:
        return False
    vector_size = metadata.get("vector_size")
    if type(vector_size) is not int or vector_size < 0:
        return False
    if data.get("_version") != INDEX_VERSION:
        return False
    for document in data["documents"].values():
        if not isinstance(document, dict) or not isinstance(document.get("chunks"), list):
            return False
        for chunk in document["chunks"]:
            vector = chunk.get("vec") if isinstance(chunk, dict) else None
            if (
                not isinstance(vector, list)
                or len(vector) != vector_size
                or any(not isinstance(value, (int, float)) for value in vector)
            ):
                return False
    return True


def _index_files(
    paths: list[Path], provider: str | None = None
) -> tuple[dict, dict | None]:
    prepared = []
    all_chunks = []
    for path in paths:
        try:
            chunks = chunk_document(path)
        except Exception as e:
            print(f"Warning: Failed to index file {path.name}: {e}")
            continue
        prepared.append((path, chunks))
        all_chunks.extend(chunks)

    result = embed(all_chunks, "search_document: ", provider) if all_chunks else None
    metadata = (
        {
            "provider": result.provider,
            "model": result.model,
            "vector_size": result.vector_size,
        }
        if result
        else None
    )
    documents = {}
    offset = 0
    for path, chunks in prepared:
        vectors = result.vectors[offset:offset + len(chunks)] if result else []
        offset += len(chunks)
        documents[path.name] = {
            "mtime": path.stat().st_mtime,
            "chunks": [
                {"text": text, "vec": [round(value, 5) for value in vector]}
                for text, vector in zip(chunks, vectors)
            ],
        }
    return documents, metadata


def _configured_metadata(provider: str | None = None) -> dict:
    actual_provider = provider or config.EMBEDDING_PROVIDER
    if actual_provider == "auto":
        actual_provider = "gemini"
    model = config.GEMINI_EMBED_MODEL if actual_provider == "gemini" else config.EMBED_MODEL
    return {"provider": actual_provider, "model": model, "vector_size": 0}


def build_index(force_provider: str | None = None) -> str:
    """Index new or changed files and drop deleted ones. Cheap when nothing changes."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    files = {
        f.name: f for f in DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    }
    valid_index = _metadata_matches(data, force_provider)
    old_index = data.get("documents", {}) if valid_index else {}
    force_rebuild = force_provider is not None or not valid_index
    changed_files = [
        path for name, path in files.items()
        if force_rebuild or old_index.get(name, {}).get("mtime") != path.stat().st_mtime
    ]
    deleted = set(old_index) - set(files)
    if not force_rebuild and not changed_files and not deleted:
        total = sum(len(doc["chunks"]) for doc in old_index.values())
        return f"{len(old_index)} documents, {total} chunks (0 updated)"

    updated, new_metadata = _index_files(changed_files, force_provider)

    if not force_rebuild and new_metadata:
        old_metadata = data["embedding"]
        if new_metadata != {
            "provider": old_metadata["provider"],
            "model": old_metadata["model"],
            "vector_size": old_metadata["vector_size"],
        }:
            force_rebuild = True
            updated, new_metadata = _index_files(
                list(files.values()), new_metadata["provider"]
            )

    index = updated if force_rebuild else {**old_index, **updated}
    for name in deleted:
        index.pop(name, None)
    metadata = new_metadata or (
        data.get("embedding") if not force_rebuild else _configured_metadata(force_provider)
    )
    if force_rebuild or updated or deleted:
        INDEX_FILE.write_text(
            json.dumps(
                {"_version": INDEX_VERSION, "embedding": metadata, "documents": index}
            ),
            encoding="utf-8"
        )
    total = sum(len(d["chunks"]) for d in index.values())
    return (
        f"{len(index)} documents, {total} chunks "
        f"({len(changed_files) + len(deleted)} updated)"
    )


def search(query: str, k: int | None = None) -> list[tuple[float, str, str]]:
    data = _load()
    if not _metadata_matches(data):
        build_index()
        data = _load()
    index = data.get("documents", {})
    items = [(name, c) for name, d in index.items() for c in d["chunks"]]
    if not items:
        return []
    query_embedding = embed([query], "search_query: ")
    metadata = data.get("embedding", {})
    if (
        metadata.get("provider") != query_embedding.provider
        or metadata.get("model") != query_embedding.model
    ):
        raise RuntimeError(
            "The document index uses a different embedding provider or model than this "
            "query. Rebuild the index with a matching EMBEDDING_PROVIDER and model."
        )
    if metadata.get("vector_size") != query_embedding.vector_size:
        build_index(force_provider=query_embedding.provider)
        data = _load()
        metadata = data.get("embedding", {})
        index = data.get("documents", {})
        if metadata.get("vector_size") != query_embedding.vector_size:
            raise RuntimeError(
                "The document index and query embeddings have different vector sizes. "
                "Rebuild the index with the configured embedding model."
            )
        items = [(name, c) for name, d in index.items() for c in d["chunks"]]
    scored = sorted(
        (
            (_cos(query_embedding.vectors[0], c["vec"]), name, c["text"])
            for name, c in items
        ),
        reverse=True,
    )
    unique = []
    seen = set()
    for result in scored:
        key = (result[1], result[2])
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique[:config.RAG_TOP_K if k is None else k]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        raise SystemExit('Usage: python -m app.memory.rag "your query"')
    print(build_index())
    for score, name, text in search(query):
        print(f"\n[{score:.3f}] {name}\n{text}")
