from app.memory import rag


def search_docs(query: str) -> str:
    """Search the user's own documents (price lists, FAQs, client notes, policies
    stored in data/docs). Use this whenever the user asks about information that
    could be in their documents, and answer only from the results.

    Args:
        query: What to look for, written as a short question or keywords.
    """
    try:
        rag.build_index()  # picks up new or edited files
        results = rag.search(query)
    except Exception as e:
        return f"Document search failed. Is Ollama running? ({e})"
    if not results:
        return "No documents found. Add .txt or .md files to data/docs."
    return "\n\n".join(f"[{name}] (score {s:.2f})\n{text}" for s, name, text in results)