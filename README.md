# PocketAgent

A personal AI assistant built with Python and Gemini. PocketAgent uses a manual
tool-calling loop, SQLite-backed memory, and local tools for notes, tasks, time,
and public GitHub repositories.

## Requirements

- Python 3.13
- A Gemini API key

## Setup

In PowerShell, from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`. The model names below are optional; these
defaults are used when they are omitted:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODEL=gemini-3.1-flash-lite
```

Keep `.env` private. It is excluded from Git.

## Run PocketAgent

Terminal chat:

```powershell
python -m app.agent
```

Telegram bot (optional): create a bot with Telegram's BotFather, then add these
values to `.env`:

```dotenv
TELEGRAM_TOKEN=your-telegram-bot-token
TELEGRAM_USER_ID=your-numeric-telegram-user-id
```

Start the bot:

```powershell
python -m app.telegram_bot
```

The bot responds only to the configured Telegram user ID and only to text
messages. Run one polling instance per token; another process using the same
token can cause Telegram polling conflicts. `/start` checks that the authorized
account can reach the bot.

FastAPI service:

```powershell
python -m uvicorn app.api:app --reload
```

The API provides `GET /health` and `POST /chat`. Send JSON such as
`{"message":"What time is it?","session_id":"local"}` to `/chat`. The
`session_id` is optional. To require an API key, set `API_KEY` in `.env` and
send it in the `x-api-key` header.

Run the tests with:

```powershell
python -m pytest
```

## Add a tool

Write a function with type hints and a docstring in `app/tools/`, import it, and
add it to `ALL_TOOLS` in `app/tools/__init__.py`. The agent uses the function
definitions to expose tools to Gemini.
