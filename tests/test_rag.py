import io
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import docx
import pytest
from google.genai.errors import APIError

os.environ.setdefault("GEMINI_API_KEY", "test")

from app import config
from app.memory import rag
from app.tools.docs import search_docs


@pytest.fixture(autouse=True)
def prevent_network(monkeypatch):
    client = MagicMock()
    client.models.embed_content.side_effect = lambda **kwargs: SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[1.0, 0.0]) for _ in kwargs["contents"]
        ]
    )
    monkeypatch.setattr(rag.genai, "Client", MagicMock(return_value=client))

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        return io.BytesIO(
            json.dumps(
                {"embeddings": [[1.0, 0.0] for _ in body["input"]]}
            ).encode("utf-8")
        )

    monkeypatch.setattr(rag.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rag.time, "sleep", lambda delay: None)


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    monkeypatch.setattr(rag, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(rag, "INDEX_FILE", tmp_path / "rag_index.json")
    return docs_dir


def test_chunking():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
    chunks = rag.chunk_text(text, size=900)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_cosine():
    assert abs(rag._cos([1, 0], [1, 0]) - 1) < 1e-9
    assert rag._cos([1, 0], [0, 1]) == 0


def test_load_text_txt_and_md(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello plain text", encoding="utf-8")
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Hello markdown", encoding="utf-8")

    assert rag.load_text(txt_file) == "Hello plain text"
    assert rag.load_text(md_file) == "# Hello markdown"


def test_load_text_docx(tmp_path):
    doc_path = tmp_path / "test.docx"
    doc = docx.Document()
    doc.add_paragraph("First paragraph text.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Header 1"
    table.cell(0, 1).text = "Value 1"
    doc.save(str(doc_path))

    extracted = rag.load_text(doc_path)
    assert "First paragraph text." in extracted
    assert "Header 1 | Value 1" in extracted


def test_markdown_chunks_include_heading_path(tmp_path):
    markdown_file = tmp_path / "prices.md"
    markdown_file.write_text(
        "# Suggested price menu\n\n"
        "General market rates.\n\n"
        "## US clients (USD)\n\n"
        "Starter price: $500.\n\n"
        "### FAQ chatbot\n\n"
        "Starter tier: $600.",
        encoding="utf-8",
    )

    chunks = rag.chunk_document(markdown_file)

    assert any("Section path: Suggested price menu" in chunk for chunk in chunks)
    assert any(
        "Section path: Suggested price menu > US clients (USD)" in chunk
        and "Starter price: $500." in chunk
        for chunk in chunks
    )
    assert any(
        "Section path: Suggested price menu > US clients (USD) > FAQ chatbot" in chunk
        and "Starter tier: $600." in chunk
        for chunk in chunks
    )
    assert all(len(chunk) <= rag.CHUNK_SIZE for chunk in chunks)


def test_docx_table_rows_include_heading_and_repeated_header(tmp_path):
    doc_path = tmp_path / "prices.docx"
    document = docx.Document()
    document.add_heading("Suggested price menu", level=1)
    document.add_heading("US clients (USD)", level=2)
    table = document.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Service"
    table.cell(0, 1).text = "Starter price"
    table.cell(1, 0).text = "FAQ chatbot"
    table.cell(1, 1).text = "$500"
    table.cell(2, 0).text = "Support chatbot"
    table.cell(2, 1).text = "$750"
    document.add_heading("India clients (INR)", level=2)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Service"
    table.cell(0, 1).text = "Starter price"
    table.cell(1, 0).text = "FAQ chatbot"
    table.cell(1, 1).text = "₹20,000"
    document.save(str(doc_path))

    chunks = rag.chunk_document(doc_path)

    us_chunks = [chunk for chunk in chunks if "US clients (USD)" in chunk]
    us_chunk = next(chunk for chunk in us_chunks if "FAQ chatbot" in chunk)
    second_us_chunk = next(chunk for chunk in us_chunks if "Support chatbot" in chunk)
    india_chunk = next(chunk for chunk in chunks if "India clients (INR)" in chunk)
    assert "Suggested price menu > US clients (USD)" in us_chunk
    assert "Table header: Service | Starter price" in us_chunk
    assert "FAQ chatbot | $500" in us_chunk
    assert "Suggested price menu > US clients (USD)" in second_us_chunk
    assert "Table header: Service | Starter price" in second_us_chunk
    assert "Support chatbot | $750" in second_us_chunk
    assert "Suggested price menu > India clients (INR)" in india_chunk
    assert "Table header: Service | Starter price" in india_chunk
    assert "FAQ chatbot | ₹20,000" in india_chunk
    assert all(len(chunk) <= rag.CHUNK_SIZE for chunk in chunks)


def test_load_text_pdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "PDF Page 1 Content"
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1]

    monkeypatch.setattr("pypdf.PdfReader", lambda path: mock_reader)
    assert rag.load_text(pdf_path) == "PDF Page 1 Content"

    mock_page1.extract_text.return_value = ""
    assert "no text found, it may be a scanned PDF" in rag.load_text(pdf_path)


def test_auto_uses_gemini_first_when_key_is_configured(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "auto")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "configured-test-key")
    gemini = MagicMock(return_value=[[1.0, 0.0]])
    ollama = MagicMock(side_effect=AssertionError("Ollama should not be called"))
    monkeypatch.setattr(rag, "_gemini_batch", gemini)
    monkeypatch.setattr(rag, "_ollama_batch", ollama)

    result = rag.embed(["hello"], "search_query: ")

    assert result.provider == "gemini"
    assert result.model == config.GEMINI_EMBED_MODEL
    assert result.vector_size == 2
    gemini.assert_called_once()
    ollama.assert_not_called()


def test_auto_falls_back_to_ollama(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "auto")
    monkeypatch.setattr(
        rag, "_gemini_batch", MagicMock(side_effect=OSError("Gemini unavailable"))
    )
    ollama = MagicMock(return_value=[[0.0, 1.0]])
    monkeypatch.setattr(rag, "_ollama_batch", ollama)

    result = rag.embed(["hello"], "search_query: ")

    assert result.provider == "ollama"
    assert result.model == config.EMBED_MODEL
    ollama.assert_called_once()


def test_gemini_sdk_embedding_call_is_batched_and_uses_configured_model(monkeypatch):
    client = MagicMock()
    client.models.embed_content.return_value.embeddings = [
        SimpleNamespace(values=[1.0, 0.0]),
        SimpleNamespace(values=[0.0, 1.0]),
    ]
    monkeypatch.setattr(rag.genai, "Client", MagicMock(return_value=client))
    monkeypatch.setattr(config, "GEMINI_EMBED_MODEL", "gemini-test")

    vectors = rag._gemini_batch(["first", "second"], "search_document: ")

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    client.models.embed_content.assert_called_once()
    kwargs = client.models.embed_content.call_args.kwargs
    assert kwargs["model"] == "gemini-test"
    assert kwargs["contents"] == ["first", "second"]
    assert kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_ollama_embedding_call_is_mocked_and_uses_configured_model(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(
            json.dumps({"embeddings": [[1.0, 0.0]]}).encode("utf-8")
        )

    monkeypatch.setattr(rag.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(config, "EMBED_MODEL", "nomic-test")

    vectors = rag._ollama_batch(["text"], "search_query: ")

    assert vectors == [[1.0, 0.0]]
    request, timeout = requests[0]
    assert json.loads(request.data) == {
        "model": "nomic-test",
        "input": ["search_query: text"],
    }
    assert timeout == 120


def test_gemini_retries_retryable_api_errors(monkeypatch):
    client = MagicMock()
    client.models.embed_content.side_effect = [
        APIError(503, {}),
        SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])]),
    ]
    monkeypatch.setattr(rag.genai, "Client", MagicMock(return_value=client))
    delays = []
    monkeypatch.setattr(rag.time, "sleep", delays.append)

    assert rag._gemini_batch(["text"], "search_query: ") == [[1.0, 0.0]]
    assert delays == [2]
    assert client.models.embed_content.call_count == 2


def test_index_rebuilds_when_provider_model_or_vector_size_changes(
    isolated_index, monkeypatch
):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_EMBED_MODEL", "gemini-test")
    gemini = MagicMock(return_value=[[1.0, 0.0]])
    ollama = MagicMock(return_value=[[0.0, 1.0]])
    monkeypatch.setattr(rag, "_gemini_batch", gemini)
    monkeypatch.setattr(rag, "_ollama_batch", ollama)
    (isolated_index / "one.txt").write_text("First document", encoding="utf-8")
    (isolated_index / "two.txt").write_text("Second document", encoding="utf-8")

    rag.build_index()
    assert gemini.call_count == 1

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "ollama")
    rag.build_index()
    assert ollama.call_count == 1

    monkeypatch.setattr(config, "EMBED_MODEL", "nomic-embed-text-new")
    rag.build_index()
    assert ollama.call_count == 2

    dimension_change = MagicMock(
        side_effect=lambda texts, prefix: [[0, 1, 0] for _ in texts]
    )
    monkeypatch.setattr(rag, "_ollama_batch", dimension_change)
    (isolated_index / "one.txt").write_text("Updated document", encoding="utf-8")
    rag.build_index()
    saved = rag._load()
    assert saved["embedding"]["vector_size"] == 3
    assert dimension_change.call_count == 2
    assert all(
        len(chunk["vec"]) == 3
        for doc in saved["documents"].values()
        for chunk in doc["chunks"]
    )


def test_old_index_without_embedding_metadata_is_rebuilt(
    isolated_index, monkeypatch
):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "gemini")
    (isolated_index / "good.txt").write_text("Valid text content", encoding="utf-8")
    index_file = rag.INDEX_FILE
    index_file.write_text(
        json.dumps(
            {
                "_version": rag.INDEX_VERSION - 1,
                "documents": {
                    "good.txt": {
                        "mtime": (isolated_index / "good.txt").stat().st_mtime,
                        "chunks": [{"text": "old", "vec": [0.0]}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    gemini = MagicMock(return_value=[[1.0, 0.0]])
    monkeypatch.setattr(rag, "_gemini_batch", gemini)

    rag.build_index()

    saved = rag._load()
    assert saved["embedding"]["provider"] == "gemini"
    assert saved["documents"]["good.txt"]["chunks"][0]["vec"] == [1.0, 0.0]
    gemini.assert_called_once()


def test_index_rebuilds_after_index_version_bump(isolated_index, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "gemini")
    path = isolated_index / "doc.txt"
    path.write_text("Current document content", encoding="utf-8")
    rag.INDEX_FILE.write_text(
        json.dumps(
            {
                "_version": rag.INDEX_VERSION - 1,
                "embedding": {
                    "provider": "gemini",
                    "model": config.GEMINI_EMBED_MODEL,
                    "vector_size": 2,
                },
                "documents": {
                    "doc.txt": {
                        "mtime": path.stat().st_mtime,
                        "chunks": [{"text": "old chunk", "vec": [0.0, 1.0]}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    gemini = MagicMock(return_value=[[1.0, 0.0]])
    monkeypatch.setattr(rag, "_gemini_batch", gemini)

    rag.build_index()

    saved = rag._load()
    assert saved["_version"] == rag.INDEX_VERSION
    assert saved["documents"]["doc.txt"]["chunks"][0]["text"] == "Current document content"
    gemini.assert_called_once()


def test_search_returns_best_matching_chunk(isolated_index, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "gemini")

    def fake_gemini(texts, prefix):
        if prefix == "search_query: ":
            return [[1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0] for text in texts]
        return [[1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(rag, "_gemini_batch", fake_gemini)
    (isolated_index / "fruit.txt").write_text(
        "Apple pie recipe\n\nOrange juice recipe", encoding="utf-8"
    )

    rag.build_index()
    results = rag.search("apple")

    assert results[0][1] == "fruit.txt"
    assert "Apple pie" in results[0][2]


def test_search_returns_configured_top_k_and_deduplicates(isolated_index, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(config, "RAG_TOP_K", 5)
    for i in range(7):
        (isolated_index / f"doc{i}.md").write_text(
            f"# Section {i}\n\nUnique content {i}", encoding="utf-8"
        )
    monkeypatch.setattr(
        rag, "_gemini_batch", lambda texts, prefix: [[1.0, 0.0] for _ in texts]
    )

    rag.build_index()
    results = rag.search("query")

    assert len(results) == 5
    assert len({(name, text) for _, name, text in results}) == 5


def test_system_prompt_requires_document_search_for_relevant_questions():
    prompt = config.SYSTEM_PROMPT
    assert "opening hours" in prompt
    assert "call search_docs first" in prompt
    assert "copy numbers exactly" in prompt
    assert "Name each source document" in prompt
    assert "which document each fact comes from" in prompt


def test_auto_provider_change_never_compares_mismatched_index(
    isolated_index, monkeypatch
):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "auto")
    (isolated_index / "doc.txt").write_text("Document content", encoding="utf-8")
    rag.build_index()
    monkeypatch.setattr(
        rag, "_gemini_batch", MagicMock(side_effect=OSError("offline"))
    )

    with pytest.raises(RuntimeError, match="different embedding provider or model"):
        rag.search("query")


def test_search_docs_returns_friendly_error_if_both_providers_fail(
    isolated_index, monkeypatch
):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "auto")
    (isolated_index / "doc.txt").write_text("Document content", encoding="utf-8")
    monkeypatch.setattr(
        rag, "_gemini_batch", MagicMock(side_effect=OSError("Gemini offline"))
    )
    monkeypatch.setattr(
        rag, "_ollama_batch", MagicMock(side_effect=OSError("Ollama offline"))
    )

    response = search_docs("query")

    assert "No embedding provider is available" in response
    assert "GEMINI_API_KEY" in response
    assert "start Ollama" in response


def test_gemini_embedding_batches_are_limited_and_delayed(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "gemini")
    batch_sizes = []
    delays = []

    def fake_gemini(texts, prefix):
        batch_sizes.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(rag, "_gemini_batch", fake_gemini)
    monkeypatch.setattr(rag.time, "sleep", delays.append)

    result = rag.embed([f"text {i}" for i in range(35)], "search_document: ")

    assert len(result.vectors) == 35
    assert batch_sizes == [16, 16, 3]
    assert delays == [rag.EMBED_BATCH_DELAY, rag.EMBED_BATCH_DELAY]


def test_index_rebuild_and_corrupt_file_survival(isolated_index, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "gemini")
    good_file = isolated_index / "good.txt"
    good_file.write_text("Valid text content", encoding="utf-8")

    corrupt_file = isolated_index / "corrupt.pdf"
    corrupt_file.write_bytes(b"corrupt non-pdf data")

    res = rag.build_index()
    assert "1 documents" in res

    results = rag.search("Valid")
    assert len(results) == 1
    assert results[0][1] == "good.txt"

    good_file.unlink()
    res2 = rag.build_index()
    assert "0 documents" in res2
