import os

os.environ.setdefault("GEMINI_API_KEY", "test")

from google.genai import types

from app import sessions


def test_trim_keeps_recent_turns():
    h = []
    for i in range(30):
        h.append(types.Content(role="user", parts=[types.Part(text=f"q{i}")]))
        h.append(types.Content(role="model", parts=[types.Part(text=f"a{i}")]))
    sessions._trim(h)
    assert h[0].parts[0].text == "q10"
    assert len(h) == 40