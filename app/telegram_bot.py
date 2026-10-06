import asyncio
import logging
import os
import re
from pathlib import Path

logging.getLogger("httpx").setLevel(logging.WARNING)

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters,
)

from app import config
from app.memory import rag
from app.sessions import chat

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def sanitize_filename(filename: str) -> str:
    name = os.path.basename(filename).strip()
    name = re.sub(r"[\\/\x00]", "", name)
    name = re.sub(r"^\.+", "", name)
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        base, ext = parts[0], parts[1].lower()
    else:
        base, ext = name, ""
    base = re.sub(r"[^A-Za-z0-9_-]", "_", base)
    if not base:
        base = "file"
    base = base[:80]
    return f"{base}.{ext}" if ext else base


def _allowed(update: Update) -> bool:
    user = update.effective_user
    allowed = user is not None and str(user.id) == config.TELEGRAM_USER_ID
    logger.info(
        "Telegram update user_id=%s allowed=%s",
        user.id if user is not None else None,
        allowed,
    )
    return allowed


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if _allowed(update) and message is not None:
        await message.reply_text("PocketAgent ready. Send me a message.")


async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = update.effective_user
    chat_instance = update.effective_chat
    if (
        not _allowed(update)
        or message is None
        or message.text is None
        or user is None
        or chat_instance is None
    ):
        return
    await context.bot.send_chat_action(chat_instance.id, "typing")
    reply = await asyncio.to_thread(chat, f"tg-{user.id}", message.text)
    await message.reply_text(reply[:4000])  # Telegram's limit is 4096 chars


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not _allowed(update) or message is None or message.document is None:
        return

    doc = message.document
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        await message.reply_text("File too large. Maximum size is 10 MB.")
        return

    filename = sanitize_filename(doc.file_name or "document.txt")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        await message.reply_text("Invalid file format. Only .pdf, .docx, .txt, and .md are supported.")
        return

    dest = rag.DOCS_DIR / filename
    existed = dest.exists()
    action = "Updated" if existed else "Saved"

    try:
        telegram_file = await context.bot.get_file(doc.file_id)
        rag.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        await telegram_file.download_to_drive(custom_path=dest)
    except Exception as e:
        await message.reply_text(f"Failed to download document: {e}")
        return

    try:
        index_msg = await asyncio.to_thread(rag.build_index)
        await message.reply_text(f"{action} {filename}: {index_msg}")
    except Exception as e:
        await message.reply_text(
            f"{action} {filename}, but indexing failed. It will retry on your next question. ({e})"
        )


async def log_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    if error is None:
        logger.error("Telegram application reported an error without an exception")
        return
    logger.error(
        "Unhandled Telegram handler error",
        exc_info=(type(error), error, error.__traceback__),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_USER_ID:
        raise SystemExit("Set TELEGRAM_TOKEN and TELEGRAM_USER_ID in .env first.")
    app = Application.builder().token(config.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_error_handler(log_error)
    print("Telegram bot running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()