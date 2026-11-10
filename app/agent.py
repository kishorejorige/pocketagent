"""PocketAgent's offline document-Q&A loop. Run terminal chat with: python -m app.agent"""
import re
from dataclasses import dataclass

from app.memory import rag
from app.tools.docs import list_docs


@dataclass
class TextPart:
    text: str


@dataclass
class ChatMessage:
    role: str
    parts: list[TextPart]


def _user_texts(contents: list[ChatMessage]) -> list[str]:
    return [
        text
        for content in contents
        if content.role == "user"
        for part in (content.parts or [])
        if (text := part.text)
    ]


def _answer(question: str, previous_question: str | None = None) -> str:
    if re.search(r"\b(document|documents|file|files)\b", question, re.IGNORECASE) and re.search(
        r"\b(list|show|have|available|what)\b", question, re.IGNORECASE
    ):
        return list_docs()

    follow_up = re.match(
        r"^\s*(and\b|what about\b|how about\b|also\b|what if\b)", question, re.IGNORECASE
    )
    if previous_question and (follow_up or len(question.split()) <= 4):
        question = f"{previous_question}\n{question}"
    return rag.answer(question)


def run_agent(contents: list[ChatMessage]) -> str:
    """Answer the latest user turn using only local documents and NLP."""
    questions = _user_texts(contents)
    if not questions:
        return "Send a question about the information in your local documents."

    reply = _answer(
        questions[-1],
        previous_question=questions[-2] if len(questions) > 1 else None,
    )
    contents.append(ChatMessage(role="model", parts=[TextPart(text=reply)]))
    return reply


def main() -> None:
    history: list[ChatMessage] = []
    print("PocketAgent ready (offline document search). Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        history.append(ChatMessage(role="user", parts=[TextPart(text=user_input)]))
        print("Agent:", run_agent(history))


if __name__ == "__main__":
    main()
