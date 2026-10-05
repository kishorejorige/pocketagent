"""PocketAgent core loop. Run the terminal chat with:  python -m app.agent"""
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from app import config
from app.llm_local import chat as local_chat
from app.llm_local import coerce_args
from app.memory import store
from app.tools import ALL_TOOLS, TOOLS

client = genai.Client(api_key=config.GEMINI_API_KEY)


@dataclass
class LLMResult:
    text: str
    calls: list  # objects with .name and .args
    content: types.Content  # what gets appended to the history
    backend: str  # "gemini" or "local"


def build_system_prompt() -> str:
    """Rebuilt every call so newly saved facts show up immediately."""
    prompt = config.SYSTEM_PROMPT
    facts = store.list_facts()
    if facts:
        prompt += "\n\nKnown facts about the user:\n" + "\n".join(f"- {t}" for _, t in facts)
    return prompt


def call_gemini(contents: list) -> LLMResult:
    """Retry on 429/500/503, and switch to the backup Gemini model after 2 failures."""
    gen_config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(),
        tools=ALL_TOOLS,  # the SDK turns each function's hints + docstring into a schema
        # We run the loop ourselves, so turn off automatic tool calling
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    delay = 2
    for attempt in range(5):
        model = config.MODEL if attempt < 2 else config.FALLBACK_MODEL
        try:
            r = client.models.generate_content(
                model=model, contents=contents, config=gen_config
            )
            calls = list(r.function_calls or [])
            return LLMResult(
                text="" if calls else (r.text or ""),
                calls=calls,
                content=r.candidates[0].content,
                backend="gemini",
            )
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == 4:
                raise
            nxt = config.MODEL if attempt + 1 < 2 else config.FALLBACK_MODEL
            note = f", switching to {nxt}" if nxt != model else ""
            print(f"  [retry] {model} busy ({e.code}), waiting {delay}s{note}...")
            time.sleep(delay)
            delay *= 2


def call_local(contents: list) -> LLMResult:
    text, calls, content = local_chat(contents, build_system_prompt(), ALL_TOOLS)
    return LLMResult(text=text, calls=calls, content=content, backend="local")


def run_agent(contents: list) -> str:
    """Run one user turn. `contents` is the conversation history (modified in place)."""
    use_local = config.LLM_PROVIDER == "ollama"
    for _ in range(config.MAX_STEPS):
        if use_local:
            result = call_local(contents)
        else:
            try:
                result = call_gemini(contents)
            except Exception as e:
                if config.LLM_PROVIDER != "auto":
                    raise
                print(f"  [fallback] Gemini failed ({type(e).__name__}); using local {config.OLLAMA_MODEL}")
                use_local = True  # stay local for the rest of this turn
                result = call_local(contents)
        contents.append(result.content)

        if not result.calls:  # no tool requested -> final answer
            return result.text

        result_parts = []
        for call in result.calls:
            args = dict(call.args or {})
            print(f"  [tool] {call.name}({args})")
            try:
                fn = TOOLS[call.name]
                tool_result = fn(**coerce_args(fn, args))
            except Exception as e:
                tool_result = f"Error: {e}"
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": tool_result}
                )
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return "Stopped: too many steps."


def main() -> None:
    history: list = []
    print(f"PocketAgent ready (provider: {config.LLM_PROVIDER}). Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        start = len(history)
        history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
        try:
            print("Agent:", run_agent(history))
        except Exception as e:
            del history[start:]  # drop the half-finished turn
            print(f"Agent: no AI backend is available right now ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()