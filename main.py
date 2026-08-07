"""
FastAPI Chatbot with Memory
----------------------------
Same chatbot logic as app.py, served as a web API. Each client gets
their own memory by passing a `session_id` — the server keeps a
separate ConversationBufferWindowMemory per session, so multiple
users can chat at once without mixing up each other's history.

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test UI.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_classic.memory import ConversationBufferWindowMemory

# ---- Config ----
MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers concise."
WINDOW_SIZE = 5  # exchanges remembered per session

app = FastAPI(title="Simple Chatbot with Memory")
llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

# Session store: one memory object per session_id.
# NOTE: this lives in RAM, so it resets if the server restarts,
# and won't work across multiple server processes/instances.
# That's fine for local dev/testing; for production you'd swap
# this for a database or Redis-backed memory instead.
sessions: dict[str, ConversationBufferWindowMemory] = {}


class ChatRequest(BaseModel):
    session_id: str   # any string that identifies one conversation, e.g. a user id
    message: str


class ChatResponse(BaseModel):
    reply: str


def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Fetch this session's memory, creating a fresh one if it's new."""
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferWindowMemory(
            k=WINDOW_SIZE, return_messages=True
        )
    return sessions[session_id]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    memory = get_memory(req.session_id)

    memory.chat_memory.add_user_message(req.message)

    windowed_history = memory.load_memory_variables({})["history"]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + windowed_history

    response = llm.invoke(messages)

    memory.chat_memory.add_ai_message(response.content)

    return ChatResponse(reply=response.content)


@app.post("/reset/{session_id}")
def reset(session_id: str):
    """Clear one session's memory (or no-op if it didn't exist)."""
    sessions.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}


@app.get("/")
def health():
    return {"status": "ok"}