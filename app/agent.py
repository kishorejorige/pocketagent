"""PocketAgent core loop. Run the terminal chat with:  python -m app.agent"""
import time

from google import genai
from google.genai import errors, types

from app import config
from app.memory import store
from app.tools import ALL_TOOLS, TOOLS

client = genai.Client(api_key=config.GEMINI_API_KEY)


def build_config() -> types.GenerateContentConfig:
    """Rebuilt every call so newly saved facts show up immediately."""
    facts = store.list_facts()
    prompt = config.SYSTEM_PROMPT
    if facts:
        prompt += "\n\nKnown facts about the user:\n" + "\n".join(f"- {t}" for _, t in facts)
    return types.GenerateContentConfig(
        system_instruction=prompt,
        tools=ALL_TOOLS,  # the SDK turns each function's hints + docstring into a schema
        # We run the loop ourselves, so turn off automatic tool calling
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def call_model(contents: list):
    """Call Gemini. Retry on 429/500/503, and switch to the backup model after 2 failures."""
    delay = 2
    for attempt in range(5):
        model = config.MODEL if attempt < 2 else config.FALLBACK_MODEL
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=build_config()
            )
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == 4:
                raise
            nxt = config.MODEL if attempt + 1 < 2 else config.FALLBACK_MODEL
            note = f", switching to {nxt}" if nxt != model else ""
            print(f"  [retry] {model} busy ({e.code}), waiting {delay}s{note}...")
            time.sleep(delay)
            delay *= 2


def run_agent(contents: list) -> str:
    """Run one user turn. `contents` is the conversation history (modified in place)."""
    for _ in range(config.MAX_STEPS):
        response = call_model(contents)
        contents.append(response.candidates[0].content)

        calls = response.function_calls
        if not calls:  # no tool requested -> final answer
            return response.text

        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            print(f"  [tool] {call.name}({args})")
            try:
                result = TOOLS[call.name](**args)
            except Exception as e:
                result = f"Error: {e}"
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return "Stopped: too many steps."


def main() -> None:
    history: list = []
    print("PocketAgent ready. Type 'quit' to exit.")
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
        except errors.APIError as e:
            del history[start:]  # drop the half-finished turn
            print(f"Agent: Gemini is unavailable right now ({e.code}). Try again in a minute.")


if __name__ == "__main__":
    main()