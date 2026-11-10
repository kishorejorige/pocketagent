import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app import api, config, telegram_bot
from app.rate_limit import SlidingWindowRateLimiter


def test_sliding_window_allows_limit_then_blocks_and_resets():
    now = [0.0]
    limiter = SlidingWindowRateLimiter(clock=lambda: now[0])

    assert limiter.allow("user", 2)
    now[0] = 1.0
    assert limiter.allow("user", 2)
    assert not limiter.allow("user", 2)

    now[0] = 61.0
    assert limiter.allow("user", 2)
    assert limiter.allow("user", 2)


def test_zero_limit_disables_rate_limiting():
    limiter = SlidingWindowRateLimiter(clock=lambda: 0.0)
    assert limiter.allow("user", 0)
    assert limiter.allow("user", 0)


def test_api_rate_limit_returns_429_without_chat_call(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")
    allow_message = MagicMock(return_value=False)
    monkeypatch.setattr(api, "allow_message", allow_message)
    chat = MagicMock()
    monkeypatch.setattr(api, "chat", chat)

    response = TestClient(api.app).post(
        "/chat", json={"session_id": "limited-session", "message": "hello"}
    )

    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]
    allow_message.assert_called_once_with("api:limited-session")
    chat.assert_not_called()


def test_telegram_rate_limit_replies_without_chat_call(monkeypatch):
    async def run():
        monkeypatch.setattr(config, "TELEGRAM_USER_ID", "12345")
        allow_message = MagicMock(return_value=False)
        monkeypatch.setattr(telegram_bot, "allow_message", allow_message)
        chat = MagicMock()
        monkeypatch.setattr(telegram_bot, "chat", chat)
        monkeypatch.setattr(
            telegram_bot.asyncio, "to_thread", AsyncMock()
        )

        update = MagicMock()
        update.effective_user.id = 12345
        update.effective_message = AsyncMock()
        update.effective_message.text = "hello"
        update.effective_chat.id = 99
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await telegram_bot.handle(update, context)

        update.effective_message.reply_text.assert_awaited_once_with(
            "Rate limit reached. Try again in a minute."
        )
        allow_message.assert_called_once_with("telegram:12345")
        context.bot.send_chat_action.assert_not_awaited()
        telegram_bot.asyncio.to_thread.assert_not_awaited()
        chat.assert_not_called()

    asyncio.run(run())
