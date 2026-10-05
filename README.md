# PocketAgent

A personal AI assistant built from scratch in Python with Gemini. No agent frameworks:
just an LLM, a tool-calling loop, and your own functions.

## Run
```
pip install -r requirements.txt
cp .env.example .env   # add your Gemini API key
python -m app.agent
```

## Add a tool
Write a function with type hints and a docstring in `app/tools/`, then add it to
`ALL_TOOLS` in `app/tools/__init__.py`.
