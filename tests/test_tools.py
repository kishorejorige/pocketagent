from app.memory import rag
from app.tools.docs import list_docs


def test_list_docs(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.pdf").write_bytes(b"12345")

    monkeypatch.setattr(rag, "DOCS_DIR", docs_dir)
    res = list_docs()
    assert "- guide.pdf" in res
