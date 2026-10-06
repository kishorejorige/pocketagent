from google.genai import types
from ollama._utils import convert_function_to_tool

from app.llm_local import coerce_args, to_ollama_messages


def _history():
    return [
        types.Content(role="user", parts=[types.Part(text="Add a task: buy domain")]),
        types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="add_task", args={"title": "buy domain"})],
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name="add_task", response={"result": "Task 1 added"})],
        ),
        types.Content(role="model", parts=[types.Part(text="Done.")]),
    ]


def test_history_converts_to_ollama_messages():
    msgs = to_ollama_messages(_history(), "SYS")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "add_task", "arguments": {"title": "buy domain"}}
    assert msgs[3]["tool_name"] == "add_task" and msgs[3]["content"] == "Task 1 added"
    assert msgs[4]["content"] == "Done."


def test_coerce_args():
    def complete_task(task_id: int, note: str = "") -> str:
        return ""

    assert coerce_args(complete_task, {"task_id": "3"}) == {"task_id": 3}
    assert coerce_args(complete_task, {"task_id": 2.0}) == {"task_id": 2}
    assert coerce_args(complete_task, {"task_id": "abc"}) == {"task_id": "abc"}

def test_strip_thinking():
    from app.llm_local import strip_thinking

    assert strip_thinking("Okay, let me see...\n</think>\n\nTask 9 done.") == "Task 9 done."
    assert strip_thinking("<think>hmm</think>Hello") == "Hello"
    assert strip_thinking("Plain answer") == "Plain answer"


def test_tool_schema_from_docstring():
    def list_notes(limit: int = 10) -> str:
        """List the user's most recent saved notes.

        Args:
            limit: How many recent notes to return.
        """
        return ""

    t = convert_function_to_tool(list_notes).model_dump(exclude_none=True)
    assert t["function"]["name"] == "list_notes"
    assert "limit" in t["function"]["parameters"]["properties"]