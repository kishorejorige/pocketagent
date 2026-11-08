import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors, types

os.environ.setdefault("GEMINI_API_KEY", "test")

from app import agent, config


@pytest.fixture(autouse=True)
def clear_cooldowns(monkeypatch):
    with agent._cooldown_lock:
        agent._model_cooldowns.clear()
    monkeypatch.setattr(config, "MODEL", "primary")
    monkeypatch.setattr(config, "FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(config, "GEMINI_COOLDOWN_SECONDS", 600)
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(agent.time, "sleep", lambda delay: None)


def _success():
    return SimpleNamespace(
        function_calls=None,
        text="ok",
        candidates=[
            SimpleNamespace(
                content=types.Content(role="model", parts=[types.Part(text="ok")])
            )
        ],
    )


def _api_error(code, headers=None):
    response = SimpleNamespace(headers=headers or {})
    return errors.APIError(code, {}, response=response)


def test_429_switches_to_fallback_immediately(monkeypatch):
    calls = []
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _api_error(429),
        _success(),
    ]
    monkeypatch.setattr(agent, "client", client)
    sleeps = []
    monkeypatch.setattr(agent.time, "sleep", sleeps.append)

    result = agent.call_gemini([])

    assert result.text == "ok"
    assert [call.kwargs["model"] for call in client.models.generate_content.call_args_list] == [
        "primary", "fallback"
    ]
    assert sleeps == []
    assert "primary" in agent._model_cooldowns


def test_cooldown_skips_primary_until_expiration(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: now[0])
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _api_error(429),
        _success(),
        _success(),
        _success(),
    ]
    monkeypatch.setattr(agent, "client", client)

    agent.call_gemini([])
    agent.call_gemini([])
    assert [call.kwargs["model"] for call in client.models.generate_content.call_args_list] == [
        "primary", "fallback", "fallback"
    ]

    now[0] += config.GEMINI_COOLDOWN_SECONDS + 1
    agent.call_gemini([])
    assert client.models.generate_content.call_args_list[-1].kwargs["model"] == "primary"


def test_retry_after_sets_model_cooldown_without_sleep(monkeypatch):
    now = [25.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: now[0])
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _api_error(429, {"Retry-After": "42"}),
        _success(),
    ]
    monkeypatch.setattr(agent, "client", client)
    sleeps = []
    monkeypatch.setattr(agent.time, "sleep", sleeps.append)

    agent.call_gemini([])

    assert agent._model_cooldowns["primary"] == 67.0
    assert sleeps == []


def test_503_retries_with_exponential_backoff(monkeypatch):
    client = MagicMock()
    client.models.generate_content.side_effect = [_api_error(503), _success()]
    monkeypatch.setattr(agent, "client", client)
    sleeps = []
    monkeypatch.setattr(agent.time, "sleep", sleeps.append)

    assert agent.call_gemini([]).text == "ok"
    assert sleeps == [2]
    assert [call.kwargs["model"] for call in client.models.generate_content.call_args_list] == [
        "primary", "primary"
    ]


def test_both_models_failing_returns_friendly_message(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_COOLDOWN_SECONDS", 600)
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _api_error(429),
        _api_error(429),
    ]
    monkeypatch.setattr(agent, "client", client)

    reply = agent.run_agent([])

    assert "Both Gemini models are unavailable or rate-limited" in reply
    assert client.models.generate_content.call_count == 2
