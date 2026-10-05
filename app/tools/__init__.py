"""Register tools here. To add a tool: write a function with type hints and a
docstring, import it, and add it to ALL_TOOLS. Gemini builds the schema from it."""
from .clock import get_current_time
from .github import check_github_repo
from .memory_tools import forget_fact, recall_facts, remember_fact
from .notes import add_note, list_notes
from .tasks import add_task, complete_task, list_tasks

ALL_TOOLS = [
    get_current_time,
    add_note,
    list_notes,
    add_task,
    list_tasks,
    complete_task,
    check_github_repo,
    remember_fact,
    recall_facts,
    forget_fact,
]

TOOLS = {fn.__name__: fn for fn in ALL_TOOLS}