from app.memory import store


def remember_fact(fact: str) -> str:
    """Save a lasting fact or preference about the user (name, clients, habits,
    preferences) so it is remembered in future sessions. Do not use for to-dos or ideas.

    Args:
        fact: One short sentence, for example "The user's client is called Mike".
    """
    store.save_fact(fact)
    return "Remembered."


def recall_facts() -> str:
    """List everything remembered about the user, with ids."""
    facts = store.list_facts()
    if not facts:
        return "Nothing remembered yet."
    return "\n".join(f"{i}. {text}" for i, text in facts)


def forget_fact(fact_id: int) -> str:
    """Delete a remembered fact by its id.

    Args:
        fact_id: The id shown by recall_facts.
    """
    return "Forgotten." if store.delete_fact(fact_id) else f"No fact with id {fact_id}."