from app.memory import rag


def search_docs(query: str) -> str:
    """Return matching excerpts from local documents with their filenames."""
    try:
        rag.build_index()
        results = rag.search(query)
    except Exception as error:
        return f"Document search failed: {error}"
    if not results:
        return "No documents found. Add .txt, .md, .pdf, or .docx files to data/docs."
    excerpts = "\n\n".join(f"[{name}]\n{text}" for _, name, text in results)
    return f"DOCUMENT EXCERPTS:\n{excerpts}"


def list_docs() -> str:
    """List supported local documents and their file sizes."""
    if not rag.DOCS_DIR.exists():
        return "No documents found."
    files = [
        path for path in sorted(rag.DOCS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in rag.EXTENSIONS
    ]
    if not files:
        return "No documents found."

    def format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    return "\n".join(
        f"- {path.name} ({format_size(path.stat().st_size)})" for path in files
    )
