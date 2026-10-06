import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

os.environ["TELEGRAM_USER_ID"] = "12345"

from app import config
from app.telegram_bot import (
    ALLOWED_EXTENSIONS, MAX_FILE_SIZE, _allowed, handle_document, sanitize_filename
)

config.TELEGRAM_USER_ID = "12345"


def test_sanitize_filename():
    assert sanitize_filename(r"..\..\secret.pdf") == "secret.pdf"
    assert sanitize_filename("funny#$%name!.txt") == "funny___name_.txt"

    long_name = "a" * 150 + ".docx"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 90
    assert sanitized.endswith(".docx")


def test_telegram_document_handler_user_check():
    async def _run():
        update = MagicMock()
        update.effective_user.id = 99999  # Disallowed user ID
        context = MagicMock()

        await handle_document(update, context)
        update.effective_message.reply_text.assert_not_called()

    asyncio.run(_run())


def test_telegram_document_handler_oversize():
    async def _run():
        update = MagicMock()
        update.effective_user.id = 12345
        message = AsyncMock()
        document = MagicMock()
        document.file_size = MAX_FILE_SIZE + 100
        message.document = document
        update.effective_message = message
        context = MagicMock()

        await handle_document(update, context)
        message.reply_text.assert_called_once_with("File too large. Maximum size is 10 MB.")

    asyncio.run(_run())


def test_telegram_document_handler_invalid_ext():
    async def _run():
        update = MagicMock()
        update.effective_user.id = 12345
        message = AsyncMock()
        document = MagicMock()
        document.file_size = 1000
        document.file_name = "malicious.exe"
        message.document = document
        update.effective_message = message
        context = MagicMock()

        await handle_document(update, context)
        message.reply_text.assert_called_once_with(
            "Invalid file format. Only .pdf, .docx, .txt, and .md are supported."
        )

    asyncio.run(_run())
