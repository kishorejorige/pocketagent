"""Index documents locally and search them with TF-IDF cosine similarity.
Run `python -m app.memory.rag` to (re)build the index by hand."""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from app import config
from app.config import DATA_DIR

DOCS_DIR = DATA_DIR / "docs"
INDEX_FILE = DATA_DIR / "rag_index.json"
INDEX_VERSION = 5
EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_ANSWER_SCORE = 0.08
NO_ANSWER = (
    "I couldn't find that information in the local documents. Try rephrasing the "
    "question or add a relevant .txt, .md, .pdf, or .docx file to data/docs."
)

_STOP_WORDS = {
    "a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "before", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "get", "give", "has", "have", "how",
    "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "please", "say", "should", "so", "tell", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your",
}
_QUERY_SYNONYMS = {
    "cost": {"price"},
    "money": {"refund"},
    "back": {"refund"},
    "return": {"refund"},
    "schedule": {"hour", "open"},
    "time": {"hour", "open"},
    "park": {"parking"},
    "car": {"parking"},
    "event": {"catering"},
    "cater": {"catering"},
}


def load_text(path: Path) -> str:
    """Load text content from a .txt, .md, .pdf, or .docx file."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8-sig", errors="ignore")
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


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "edly", "ed", "able", "ly", "es", "s"):
        if len(token) - len(suffix) >= 3 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token


def _tokens(text: str, expand_query: bool = False) -> list[str]:
    tokens = [
        _stem(token.lower())
        for token in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    ]
    tokens = [token for token in tokens if token not in _STOP_WORDS and len(token) > 1]
    if expand_query:
        tokens.extend(
            _stem(synonym)
            for token in tuple(tokens)
            for synonym in _QUERY_SYNONYMS.get(token, ())
        )
    return tokens


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(weight * b.get(term, 0.0) for term, weight in a.items())
    norm_a = math.sqrt(sum(weight * weight for weight in a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _tfidf_vectors(documents: list[str], query: str) -> tuple[list[dict[str, float]], dict[str, float]]:
    document_terms = [Counter(_tokens(text)) for text in documents]
    query_terms = Counter(_tokens(query, expand_query=True))
    document_frequency = Counter(
        term for terms in document_terms for term in terms
    )
    count = len(document_terms)

    def vectorize(terms: Counter) -> dict[str, float]:
        return {
            term: (1 + math.log(frequency))
            * (math.log((count + 1) / (document_frequency[term] + 1)) + 1)
            for term, frequency in terms.items()
        }

    return [vectorize(terms) for terms in document_terms], vectorize(query_terms)


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


def _load() -> dict:
    if INDEX_FILE.exists():
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("documents"), dict):
                return data
        except Exception:
            pass
    return {}


def _metadata_matches(data: dict) -> bool:
    if (
        data.get("_version") != INDEX_VERSION
        or data.get("algorithm") != "local-tfidf-v1"
    ):
        return False
    for document in data.get("documents", {}).values():
        if not isinstance(document, dict) or not isinstance(document.get("chunks"), list):
            return False
        if any(
            not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str)
            for chunk in document["chunks"]
        ):
            return False
    return True


def _index_files(paths: list[Path]) -> dict:
    documents = {}
    for path in paths:
        try:
            chunks = chunk_document(path)
        except Exception as e:
            print(f"Warning: Failed to index file {path.name}: {e}")
            continue
        documents[path.name] = {
            "mtime": path.stat().st_mtime,
            "chunks": [{"text": text} for text in chunks],
        }
    return documents


def build_index() -> str:
    """Index new or changed files and drop deleted ones. Cheap when nothing changes."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    files = {
        f.name: f for f in DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    }
    valid_index = _metadata_matches(data)
    old_index = data.get("documents", {}) if valid_index else {}
    force_rebuild = not valid_index
    changed_files = [
        path for name, path in files.items()
        if force_rebuild or old_index.get(name, {}).get("mtime") != path.stat().st_mtime
    ]
    deleted = set(old_index) - set(files)
    if not force_rebuild and not changed_files and not deleted:
        total = sum(len(doc["chunks"]) for doc in old_index.values())
        return f"{len(old_index)} documents, {total} chunks (0 updated)"

    updated = _index_files(changed_files)

    index = updated if force_rebuild else {**old_index, **updated}
    for name in deleted:
        index.pop(name, None)
    if force_rebuild or updated or deleted:
        INDEX_FILE.write_text(
            json.dumps(
                {
                    "_version": INDEX_VERSION,
                    "algorithm": "local-tfidf-v1",
                    "documents": index,
                }
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
    if not items or not _tokens(query, expand_query=True):
        return []
    vectors, query_vector = _tfidf_vectors([chunk["text"] for _, chunk in items], query)
    scored = sorted(
        (
            (_cos(vector, query_vector), name, chunk["text"])
            for (name, chunk), vector in zip(items, vectors)
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


def answer(query: str) -> str:
    """Return relevant document passages with their source, without external models."""
    results = search(query)
    candidates = [
        (score, name, paragraph.strip())
        for score, name, text in results
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    if not candidates:
        return NO_ANSWER

    paragraph_vectors, query_vector = _tfidf_vectors(
        [paragraph for _, _, paragraph in candidates], query
    )
    scored = sorted(
        (
            (0.4 * chunk_score + 0.6 * _cos(vector, query_vector), name, paragraph)
            for (chunk_score, name, paragraph), vector
            in zip(candidates, paragraph_vectors)
        ),
        reverse=True,
    )
    relevant = [result for result in scored if result[0] >= MIN_ANSWER_SCORE]
    if not relevant:
        return NO_ANSWER
    best_score = relevant[0][0]
    relevant = [
        result for result in relevant
        if result[0] >= max(MIN_ANSWER_SCORE, best_score * 0.55)
    ][:3]
    return "\n\n".join(
        f"According to {name}:\n{text}" for _, name, text in relevant
    )


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        raise SystemExit('Usage: python -m app.memory.rag "your query"')
    print(build_index())
    for score, name, text in search(query):
        print(f"\n[{score:.3f}] {name}\n{text}")
