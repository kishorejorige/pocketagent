import os

os.environ.setdefault("GEMINI_API_KEY", "test")

from app.memory import rag


def test_chunking():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
    chunks = rag.chunk_text(text, size=900)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_cosine():
    assert abs(rag._cos([1, 0], [1, 0]) - 1) < 1e-9
    assert rag._cos([1, 0], [0, 1]) == 0