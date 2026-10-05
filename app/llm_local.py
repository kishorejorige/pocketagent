"""Local chat backend (Ollama). Our history is stored in Gemini format; this module
converts it to Ollama messages and converts Ollama's reply back."""
import typing
from types import SimpleNamespace

import ollama
from google.genai import types

from app import config


def to_ollama_messages(contents: list, system_prompt: str) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for c in contents:
        parts = c.parts or []
        if c.role == "model":
            text = "".join(p.text for p in parts if p.text and not p.thought)
            calls = [
                {
                    "function": {
                        "name": p.function_call.name,
                        "arguments": dict(p.function_call.args or {}),
                    }
                }
                for p in parts
                if p.function_call
            ]
            msg: dict = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        else:
            for p in parts:
                if p.function_response:
                    resp = p.function_response.response or {}
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_name": p.function_response.name,
                            "content": str(resp.get("result", resp)),
                        }
                    )
            text = "".join(p.text for p in parts if p.text)
            if text:
                msgs.append({"role": "user", "content": text})
    return msgs


def coerce_args(fn, args: dict) -> dict:
    """Small models often send "3" instead of 3. Cast arguments to the function's type hints."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        return args
    out = {}
    for key, value in args.items():
        want = hints.get(key)
        try:
            if want is int and not isinstance(value, int):
                value = int(float(value))
            elif want is float and not isinstance(value, (int, float)):
                value = float(value)
            elif want is str and not isinstance(value, str):
                value = str(value)
        except (TypeError, ValueError):
            pass
        out[key] = value
    return out


def chat(contents: list, system_prompt: str, tools: list):
    """One model call. Returns (text, calls, content) in the shape the agent loop uses."""
    client = ollama.Client(host=config.OLLAMA_URL, timeout=config.OLLAMA_TIMEOUT)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=to_ollama_messages(contents, system_prompt),
        tools=tools,
        options={"num_ctx": config.OLLAMA_NUM_CTX},
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    )
    msg = resp.message
    text = msg.content or ""
    calls = [
        SimpleNamespace(name=tc.function.name, args=dict(tc.function.arguments or {}))
        for tc in (msg.tool_calls or [])
    ]
    parts = []
    if text:
        parts.append(types.Part(text=text))
    for c in calls:
        parts.append(types.Part.from_function_call(name=c.name, args=c.args))
    if not parts:
        text = "(the local model returned an empty answer)"
        parts.append(types.Part(text=text))
    return text, calls, types.Content(role="model", parts=parts)