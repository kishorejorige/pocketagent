"""PocketAgent core loop. Run the terminal chat with:  python -m app.agent"""
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from google import genai
from google.genai import errors, types

from app import config
from app.llm_local import chat as local_chat
from app.llm_local import coerce_args
from app.memory import store
from app.tools import ALL_TOOLS, TOOLS

client = genai.Client(api_key=config.GEMINI_API_KEY)
_model_cooldowns: dict[str, float] = {}
_cooldown_lock = threading.Lock()


class GeminiUnavailable(Exception):
    pass


def _model_available(model: str) -> bool:
    now = time.monotonic()
    with _cooldown_lock:
        until = _model_cooldowns.get(model)
        if until is None:
            return True
        if until > now:
            return False
        del _model_cooldowns[model]
    print(f"  [retry] {model} cooldown ended; retrying...")
    return True


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
    """Retry transient errors and skip rate-limited models while they cool down."""
    gen_config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(),
        tools=ALL_TOOLS,  # the SDK turns each function's hints + docstring into a schema
        # We run the loop ourselves, so turn off automatic tool calling
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    delay = 2
    primary = config.MODEL
    fallback = config.FALLBACK_MODEL
    models = list(dict.fromkeys((primary, fallback)))
    available = [model for model in models if _model_available(model)]
    if not available:
        raise GeminiUnavailable(
            "Both Gemini models are cooling down. Please try again in a few minutes."
        )

    model = available[0]
    primary_failures = 0
    for attempt in range(5):
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
            if e.code == 429:
                retry_after = _retry_after_seconds(e)
                cooldown = (
                    retry_after
                    if retry_after is not None
                    else config.GEMINI_COOLDOWN_SECONDS
                )
                with _cooldown_lock:
                    _model_cooldowns[model] = time.monotonic() + cooldown
                print(f"  [cooldown] {model} rate limited; cooling down for {cooldown:g}s")
                next_models = [candidate for candidate in models if candidate != model]
                next_model = next(
                    (
                        candidate for candidate in next_models
                        if _model_available(candidate)
                    ),
                    None,
                )
                if next_model is None:
                    raise GeminiUnavailable(
                        "Both Gemini models are unavailable or rate-limited. "
                        "Please try again later."
                    ) from e
                print(f"  [retry] {model} rate limited (429), switching to {next_model}...")
                model = next_model
                continue

            if e.code not in (500, 503):
                next_model = next(
                    (
                        candidate for candidate in models
                        if candidate != model
                        and _model_available(candidate)
                    ),
                    None,
                )
                if next_model is None:
                    raise GeminiUnavailable(
                        "Both Gemini models failed. Please try again later."
                    ) from e
                print(f"  [retry] {model} failed ({e.code}), switching to {next_model}...")
                model = next_model
                continue

            if model == primary:
                primary_failures += 1
            next_model = model
            if model == primary and primary_failures >= 2 and fallback in available:
                next_model = fallback
            if attempt == 4:
                raise GeminiUnavailable(
                    "Both Gemini models are unavailable right now. Please try again later."
                ) from e
            note = f", switching to {next_model}" if next_model != model else ""
            print(f"  [retry] {model} busy ({e.code}), waiting {delay}s{note}...")
            time.sleep(delay)
            delay *= 2
            model = next_model
        except Exception as e:
            next_model = next(
                (
                    candidate for candidate in models
                    if candidate != model
                    and _model_available(candidate)
                ),
                None,
            )
            if next_model is None:
                raise GeminiUnavailable(
                    "Both Gemini models are unavailable right now. Please try again later."
                ) from e
            print(f"  [retry] {model} failed ({type(e).__name__}), switching to {next_model}...")
            model = next_model
    raise GeminiUnavailable("Both Gemini models are unavailable right now. Please try again later.")


def _retry_after_seconds(error: errors.APIError) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


LOCAL_RULES = (
    "\n\nIMPORTANT: To add, complete, save or look up anything you MUST call a tool. "
    "Never say an action is done unless you called the tool in this turn. "
    "If unsure, call the tool."
)


def call_local(contents: list) -> LLMResult:
    text, calls, content = local_chat(contents, build_system_prompt() + LOCAL_RULES, ALL_TOOLS)
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
                    if isinstance(e, GeminiUnavailable):
                        return str(e)
                    raise
                print(f"  [fallback] Gemini failed ({type(e).__name__}); using local {config.OLLAMA_MODEL}")
                use_local = True  # stay local for the rest of this turn
                try:
                    result = call_local(contents)
                except Exception:
                    if isinstance(e, GeminiUnavailable):
                        return (
                            "Gemini models are unavailable and local Ollama could not "
                            "answer. Please try again later."
                        )
                    raise
        contents.append(result.content)

        if not result.calls:  # no tool requested -> final answer
            return f"[local model] {result.text}" if result.backend == "local" else result.text

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