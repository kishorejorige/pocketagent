import docx
import pytest

from app.memory import rag
from app.tools.docs import search_docs


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
    assert all(len(chunk) <= 900 for chunk in chunks)


def test_cosine_similarity():
    assert rag._cos({"a": 1.0}, {"a": 1.0}) == 1.0
    assert rag._cos({"a": 1.0}, {"b": 1.0}) == 0.0


def test_load_text_txt_and_markdown(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("\ufeffHello plain text", encoding="utf-8")
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Hello markdown", encoding="utf-8")

    assert rag.load_text(txt_file) == "Hello plain text"
    assert rag.load_text(md_file) == "# Hello markdown"


def test_load_text_docx(tmp_path):
    doc_path = tmp_path / "test.docx"
    document = docx.Document()
    document.add_paragraph("First paragraph text.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Header 1"
    table.cell(0, 1).text = "Value 1"
    document.save(str(doc_path))

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
    table.cell(1, 1).text = "INR 20,000"
    document.save(str(doc_path))

    chunks = rag.chunk_document(doc_path)
    us_chunk = next(chunk for chunk in chunks if "US clients (USD)" in chunk and "FAQ chatbot" in chunk)
    second_us_chunk = next(
        chunk for chunk in chunks
        if "US clients (USD)" in chunk and "Support chatbot" in chunk
    )
    india_chunk = next(chunk for chunk in chunks if "India clients (INR)" in chunk)
    assert "Suggested price menu > US clients (USD)" in us_chunk
    assert "Table header: Service | Starter price" in us_chunk
    assert "FAQ chatbot | $500" in us_chunk
    assert "FAQ chatbot | $500" not in second_us_chunk
    assert "Support chatbot | $750" in second_us_chunk
    assert "Suggested price menu > India clients (INR)" in india_chunk
    assert "FAQ chatbot | INR 20,000" in india_chunk
    assert all(len(chunk) <= rag.CHUNK_SIZE for chunk in chunks)


def test_load_text_pdf(tmp_path, monkeypatch):
    pytest.importorskip("pypdf")
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    class FakePage:
        def extract_text(self):
            return "PDF Page Content"

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", lambda path: FakeReader())
    assert rag.load_text(pdf_path) == "PDF Page Content"


def test_search_ranks_documents_with_local_tfidf(isolated_index):
    (isolated_index / "cafe.txt").write_text(
        "Opening hours: Monday to Friday 7:30 AM to 6:00 PM. "
        "We are closed on Sundays.\n\n"
        "Refund policy: Unopened items can be refunded within 7 days with a receipt.",
        encoding="utf-8",
    )
    (isolated_index / "parking.txt").write_text(
        "Free parking is available behind the building for two hours.",
        encoding="utf-8",
    )

    result = rag.build_index()
    matches = rag.search("What time does the cafe open?")
    assert "2 documents" in result
    assert matches[0][1] == "cafe.txt"
    assert "Opening hours" in matches[0][2]


def test_query_normalization_and_synonyms_find_relevant_answer(isolated_index):
    (isolated_index / "faq.txt").write_text(
        "Refund policy: Unopened items can be refunded within 7 days with a receipt.\n\n"
        "Free parking is available behind the cafe for up to 2 hours.",
        encoding="utf-8",
    )

    refund_answer = rag.answer("Can I get my money back?")
    parking_answer = rag.answer("Can I park my car?")
    assert "Refund policy" in refund_answer
    assert "Parking" not in refund_answer
    assert "faq.txt" in parking_answer
    assert "parking" in parking_answer.lower()
    assert "Refund policy" not in parking_answer


def test_answer_does_not_guess_when_no_document_matches(isolated_index):
    (isolated_index / "faq.txt").write_text(
        "We offer coffee, tea, and fresh pastries.", encoding="utf-8"
    )
    assert "couldn't find that information" in rag.answer("How do I repair a bicycle?")


def test_search_deduplicates_and_respects_top_k(isolated_index, monkeypatch):
    monkeypatch.setattr(rag.config, "RAG_TOP_K", 2)
    for number in range(4):
        (isolated_index / f"doc{number}.txt").write_text(
            f"Unique subject {number} with shared cafe policy.", encoding="utf-8"
        )

    results = rag.search("shared cafe policy")
    assert len(results) == 2
    assert len({(name, text) for _, name, text in results}) == 2
    assert all(score > 0 for score, _, _ in results)


def test_index_updates_changed_and_deleted_files(isolated_index):
    first = isolated_index / "first.txt"
    first.write_text("First version contains coffee.", encoding="utf-8")
    assert "1 documents" in rag.build_index()
    assert "0 updated" in rag.build_index()

    first.write_text("Updated version contains tea.", encoding="utf-8")
    assert "1 updated" in rag.build_index()
    assert "tea" in rag.search("tea")[0][2]

    first.unlink()
    assert "0 documents" in rag.build_index()
    assert rag.search("tea") == []


def test_legacy_index_is_rebuilt_locally(isolated_index):
    path = isolated_index / "faq.txt"
    path.write_text("The cafe opens at 7:30 AM.", encoding="utf-8")
    rag.INDEX_FILE.write_text(
        '{"_version": 4, "embedding": {"provider": "legacy"}, "documents": {}}',
        encoding="utf-8",
    )

    rag.build_index()

    index = rag._load()
    assert index["_version"] == rag.INDEX_VERSION
    assert index["algorithm"] == "local-tfidf-v1"
    assert "vec" not in index["documents"]["faq.txt"]["chunks"][0]


def test_search_docs_returns_source_passage(isolated_index):
    (isolated_index / "cafe.txt").write_text(
        "Parking: Free parking is available behind the cafe.", encoding="utf-8"
    )

    response = search_docs("Where can I park?")

    assert "[cafe.txt]" in response
    assert "Free parking is available" in response


def test_index_survives_corrupt_files(isolated_index, capsys):
    (isolated_index / "good.txt").write_text("Valid text content", encoding="utf-8")
    (isolated_index / "corrupt.pdf").write_bytes(b"corrupt non-pdf data")

    result = rag.build_index()

    assert "1 documents" in result
    assert "Warning: Failed to index file corrupt.pdf" in capsys.readouterr().out
    assert rag.search("Valid text")[0][1] == "good.txt"
