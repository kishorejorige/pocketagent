# PocketAgent

PocketAgent is a local document Q&A bot. Chat with it in the terminal, on
Telegram, or through a FastAPI endpoint. It searches your own documents and
returns matching passages with their source filenames. Chat and document search
do not call an external language-model API.

## Features

- **Offline document Q&A:** search `.txt`, `.md`, `.pdf`, and `.docx` files in
  `data/docs`.
- **Local NLP ranking:** token normalization, stop-word filtering, query
  synonym expansion, and corpus-trained TF-IDF/cosine similarity rank relevant
  passages without a model download or external service.
- **Source-grounded replies:** matching passages are quoted with their document
  filename; if no good match is found, the bot says so instead of guessing.
- **Three interfaces:** terminal chat, a Telegram bot restricted to your user
  ID, and a FastAPI service with `GET /health` and `POST /chat` (optional API
  key).
- **Telegram document uploads:** authorized users can upload supported
  documents, which are indexed automatically.
- **Deployable:** runs in Docker with a restart policy and is covered by tests.

## How it works

```text
 Terminal  |  Telegram bot  |  FastAPI /chat
                   |
             sessions (chat history in memory)
                   |
          local NLP document search
         (TF-IDF + cosine similarity)
                   |
     data/docs -> extracted text chunks -> ranked passages
```

The index is rebuilt when documents change and stored in `data/rag_index.json`.
No external AI service, model download, or API key is needed for document
indexing and answers.
This statistical retrieval approach is lightweight and private, but unlike a
generative model it does not invent or paraphrase information; it returns source
text and can miss questions phrased very differently from the documents.

## Requirements

- Python 3.13
- Docker Desktop (optional, for running the bot in the background)

## Setup

In PowerShell, from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The first setup installs dependencies and creates `.env`. The launcher requires
this file to exist for every mode. Set `TELEGRAM_TOKEN` and `TELEGRAM_USER_ID`
in it to run the Telegram bot; those settings are not needed for terminal chat
or the API. Keep `.env` private and do not commit it.
`RAG_TOP_K` sets the number of candidate passages considered (default 5);
answers include up to three relevant passages.
`RATE_LIMIT_PER_MINUTE` limits each Telegram user and API session to 10
messages per minute by default; set it to `0` to disable the limit.
Set `API_KEY` to protect API requests with the `x-api-key` header. The API
listens on localhost by default; do not expose it directly to the internet
without configuring authentication and HTTPS through a trusted proxy.

## Run PocketAgent

Use the PowerShell launcher from the project directory:

```powershell
.\run.ps1                 # Start the Telegram bot (default)
.\run.ps1 -Mode terminal  # Start terminal chat
.\run.ps1 -Mode api       # Start the FastAPI server at http://127.0.0.1:8000
```

The launcher uses `.venv` when available and otherwise uses `python` from your
PATH. It expects dependencies to be installed using the setup steps above. If
PowerShell blocks local scripts, run it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Mode terminal
```

### Terminal chat

```powershell
.\run.ps1 -Mode terminal
```

### Telegram bot

Create a bot with Telegram's BotFather, then set the following values in `.env`:

```dotenv
TELEGRAM_TOKEN=your-telegram-bot-token
TELEGRAM_USER_ID=your-numeric-telegram-user-id
```

Start the bot:

```powershell
.\run.ps1
```

The bot responds only to the configured Telegram user ID and only to text
messages and supported document uploads. Run one polling instance per token;
another process using the same token can cause Telegram polling conflicts.
`/start` checks that the authorized account can reach the bot.

### FastAPI service

```powershell
.\run.ps1 -Mode api
```

The API provides `GET /health` and `POST /chat`. The `session_id` is optional.
For example, from PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"message":"What does the refund policy say?","session_id":"local"}'
```

To require an API key, set `API_KEY` in `.env` and include it in the
`x-api-key` header on the `/chat` request.

### Docker

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f bot
docker compose down
```

Docker Compose starts only the Telegram bot. To run the API locally, use
`.\run.ps1 -Mode api`. Docker restarts the bot automatically unless you stop it.
Compose bind-mounts `./data` to `/app/data`, so your documents and search index
persist across container restarts. Conversation history is held in memory and
resets when the bot restarts.

Only one process can poll a Telegram bot token at a time. Stop any other local
or remote bot instance using the same token before starting the Compose bot.

## Ask questions about your documents

Put `.txt`, `.md`, `.pdf`, or `.docx` files directly in `data/docs`, then ask a
question about them, for example "What does the refund policy say?" or "When is
the cafe open?". Answers include the source filename. Scanned PDFs are not
OCR-processed. Questions that cannot be matched to the local documents receive
a clear not-found reply.

## Tests

```powershell
python -m pytest
```

## Tech

Python 3.13, local TF-IDF document retrieval, FastAPI, python-telegram-bot,
Docker, pytest.
