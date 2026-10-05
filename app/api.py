from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app import config
from app.sessions import chat

app = FastAPI(title="PocketAgent")


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(body: ChatIn, x_api_key: str | None = Header(default=None)):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"reply": chat(body.session_id, body.message)}