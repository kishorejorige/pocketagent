# PocketAgent

A personal AI assistant built from scratch in Python and Gemini, with no agent
frameworks. Chat with it in the terminal, on Telegram, or through a FastAPI
endpoint. It calls tools, remembers facts you ask it to save, and answers
questions from your own documents.

<!-- Add a screenshot after saving it as docs/telegram-demo.png, then uncomment:
![PocketAgent on Telegram](docs/telegram-demo.png)
-->

## Features

- **Manual tool-calling loop:** the agent decides when to call a tool, runs it,
  and feeds the result back to the model.
- **Tools:** current time, notes, tasks, public GitHub repository checks, saved
  facts (remember, recall, forget), and document search and listing.
- **Document Q&A (RAG):** ask questions about `.txt`, `.md`, `.pdf` and `.docx`
  files placed in `data/docs`.
- **Saved facts in SQLite:** facts you ask it to remember are stored in
  `data/pocketagent.db` and survive restarts.
- **Three interfaces:** terminal chat, a Telegram bot restricted to your user ID,
  and a FastAPI service with `GET /health` and `POST /chat` (optional API key).
- **Resilient:** retries on temporary Gemini errors (429, 500, 503), then tries
  a fallback model. In `auto` mode it can switch to a local Ollama model.
- **Deployable:** runs in Docker with a restart policy, and is covered by tests.

## How it works

```text
 Terminal  |  Telegram bot  |  FastAPI /chat
                   |
             sessions (chat history in memory)
                   |
   agent loop  -->  tools: time, notes, tasks, GitHub, saved facts (SQLite),
                    documents (RAG)
                   |
   Gemini (primary -> fallback model)  -->  optional local Ollama
```

## Requirements

- Python 3.13
- A Gemini API key
- Docker Desktop (optional, for running the bot in the background)
- Ollama (optional; for local chat models or as an embedding fallback)

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

### Model provider

Set `LLM_PROVIDER` in `.env` to choose the backend:

- `auto` (default): tries Gemini first, and if the Gemini call fails, switches to
  local Ollama for the rest of that turn.
- `gemini`: Gemini only.
- `ollama`: local Ollama only.

Set `EMBEDDING_PROVIDER` to choose the document embedding backend:

- `auto` (default): Gemini first, then Ollama if Gemini embeddings fail.
- `gemini`: Gemini embeddings only.
- `ollama`: Ollama embeddings only.

The default Gemini embedding model is `gemini-embedding-001`. Ollama must be
running and reachable at `OLLAMA_URL` only when selected or used as a fallback.

## Run PocketAgent

### Terminal chat

```powershell
python -m app.agent
```

### Telegram bot

Create a bot with Telegram's BotFather, then add these values to `.env`:

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

### FastAPI service

```powershell
python -m uvicorn app.api:app --reload
```

The API provides `GET /health` and `POST /chat`. Send JSON such as
`{"message":"What time is it?","session_id":"local"}` to `/chat`. The
`session_id` is optional. To require an API key, set `API_KEY` in `.env` and
send it in the `x-api-key` header.

### Docker (runs the Telegram bot in the background)

```powershell
docker compose up -d --build
docker compose logs -f
docker compose down
```

The container restarts automatically unless you stop it. Docker Compose
bind-mounts `./data` to `/app/data`, so the SQLite database and your documents
are kept across container restarts as long as that folder stays on the host.
Conversation history is held in memory and resets when the process restarts.
Stop any local copy of the bot first so two processes do not poll the same
token.

## Ask questions about your documents

Put `.txt`, `.md`, `.pdf` or `.docx` files directly in `data/docs`, then ask the
agent about them, for example "What documents do you have?" or "What does the
refund policy say?". Gemini embeddings work with a Gemini API key alone;
Ollama embeddings are an optional alternative. Scanned PDFs are not OCR-processed.

## Add a tool

Write a function with type hints and a docstring in `app/tools/`, import it, and
add it to `ALL_TOOLS` in `app/tools/__init__.py`. The agent uses the function
definitions to expose tools to Gemini.

## Tests

```powershell
python -m pytest
```

## Tech

Python 3.13, Gemini API, FastAPI, SQLite, python-telegram-bot, Ollama, Docker,
pytest.