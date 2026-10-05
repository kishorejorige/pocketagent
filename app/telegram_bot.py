import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters,
)

from app import config
from app.sessions import chat

logger = logging.getLogger(__name__)


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
    app.add_error_handler(log_error)
    print("Telegram bot running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()