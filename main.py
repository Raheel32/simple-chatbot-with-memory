"""
FastAPI Chatbot with Memory + Authentication
----------------------------------------------
Same chatbot logic as before, now with two layers of protection so
a session's history can't be read or hijacked by someone else:

1. API KEY - every request needs a valid `X-API-Key` header, or it's
   rejected outright. Anonymous requests are never allowed.

2. SESSION OWNERSHIP - session_ids are no longer free-form strings
   you make up on the client. You must first call /session/start,
   which generates a long random, unguessable session_id and records
   which API key "owns" it. Every later request for that session_id
   must present the SAME API key, or it's rejected — so even if
   someone saw/guessed a session_id, they still couldn't use it
   without also having the matching key.

Ownership records live in SQLite (same chat_memory.db file), so
they survive restarts just like the chat history does.

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test UI.
For endpoints marked with a lock icon there, click "Authorize" and
paste one of the keys from your .env file.
"""

import os
import sqlite3
import secrets
from contextlib import contextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()  # reads .env into environment variables

# ---- Config ----
MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers concise."
WINDOW_SIZE = 5  # exchanges remembered per session
DB_FILE = "chat_memory.db"
DB_PATH = f"sqlite:///{DB_FILE}"

# Comma-separated list of valid keys, e.g. API_KEYS=peter-dev-key,other-key
VALID_API_KEYS = {
    k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()
}
if not VALID_API_KEYS:
    raise RuntimeError(
        "No API keys configured. Create a .env file with API_KEYS=your-secret-key"
    )

app = FastAPI(title="Simple Chatbot with Memory")
llm = ChatOllama(model=MODEL_NAME, temperature=0.7)
memory_cache: dict[str, ConversationBufferWindowMemory] = {}


# ---------- Session ownership storage (plain sqlite3, separate table) ----------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                   session_id TEXT PRIMARY KEY,
                   api_key TEXT NOT NULL,
                   created_at TEXT DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_session(api_key: str) -> str:
    session_id = secrets.token_urlsafe(32)  # unguessable: 32 random bytes
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, api_key) VALUES (?, ?)",
            (session_id, api_key),
        )
    return session_id


def session_belongs_to(session_id: str, api_key: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT api_key FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row is not None and row[0] == api_key


def delete_session_record(session_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


# ---------- Auth dependency ----------

def require_api_key(x_api_key: str = Header(..., description="Your API key")) -> str:
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ---------- Chat memory helpers ----------

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in memory_cache:
        history = SQLChatMessageHistory(session_id=session_id, connection=DB_PATH)
        memory_cache[session_id] = ConversationBufferWindowMemory(
            k=WINDOW_SIZE, return_messages=True, chat_memory=history
        )
    return memory_cache[session_id]


# ---------- Schemas ----------

class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


# ---------- Routes ----------

@app.post("/session/start", response_model=SessionResponse)
def start_session(api_key: str = Depends(require_api_key)):
    """Call this first. Returns a fresh, unguessable session_id tied
    to your API key. Use that session_id (with the same key) for
    every /chat call in this conversation."""
    session_id = create_session(api_key)
    return SessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, api_key: str = Depends(require_api_key)):
    if not session_belongs_to(req.session_id, api_key):
        # Same error whether the session doesn't exist or belongs to
        # someone else — don't reveal which, that itself is info a
        # spoofing attempt could use.
        raise HTTPException(status_code=403, detail="Invalid session")

    memory = get_memory(req.session_id)
    memory.chat_memory.add_user_message(req.message)

    windowed_history = memory.load_memory_variables({})["history"]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + windowed_history

    response = llm.invoke(messages)
    memory.chat_memory.add_ai_message(response.content)

    return ChatResponse(reply=response.content)


@app.post("/reset/{session_id}")
def reset(session_id: str, api_key: str = Depends(require_api_key)):
    if not session_belongs_to(session_id, api_key):
        raise HTTPException(status_code=403, detail="Invalid session")

    memory_cache.pop(session_id, None)
    SQLChatMessageHistory(session_id=session_id, connection=DB_PATH).clear()
    delete_session_record(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/")
def health():
    return {"status": "ok"}