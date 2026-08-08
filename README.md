# Simple Chatbot with Memory

A minimal command-line chatbot that remembers the conversation using
LangChain + a local Ollama model (no API key or cost).

## How the "memory" works
LLMs are stateless — they don't remember anything by themselves.
"Memory" just means: **we keep track of past messages and re-send
them on every turn.**

This version uses LangChain's `ConversationBufferWindowMemory`,
which only keeps the last `WINDOW_SIZE` exchanges (set to 5 by
default in `app.py`) instead of the entire conversation. Once you
go past that, the oldest exchange is automatically dropped — so
the amount of text sent to the model (and your token usage) stays
flat no matter how long the chat runs, instead of growing forever.
Type `reset` in the chat to clear memory and start fresh, or change
`WINDOW_SIZE` in `app.py` to remember more/fewer exchanges.

## Setup
1. Install Ollama: https://ollama.com/download
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```
4. Run it:
   ```bash
   python app.py
   ```

## Example
```
You: My name is Peter
Bot: Nice to meet you, Peter!
You: What's my name?
Bot: Your name is Peter.
```

## Running as a web API (main.py)
The same chatbot logic is also available as a FastAPI web service in
`main.py`, so it can be called from a browser, frontend, or another
app instead of the terminal.

1. Make sure Ollama is running and dependencies are installed (same
   setup as above — `requirements.txt` now also includes `fastapi`
   and `uvicorn`).
2. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
3. Open **http://127.0.0.1:8000/docs** — FastAPI's built-in interactive
   UI where you can try the endpoints directly in the browser.

### Endpoints
- `POST /session/start` — **call this first.** Requires `X-API-Key`
  header. Returns a fresh, unguessable `session_id`:
  ```json
  { "session_id": "kR7f...(long random string)" }
  ```
- `POST /chat` — requires `X-API-Key` header + a `session_id` from
  `/session/start`:
  ```json
  { "session_id": "kR7f...", "message": "My name is Peter" }
  ```
  returns
  ```json
  { "reply": "Nice to meet you, Peter!" }
  ```
- `POST /reset/{session_id}` — requires `X-API-Key`; clears and
  permanently deletes that session
- `GET /` — health check (no auth needed)

### Setting up your API key
1. Copy `.env.example` to a new file named `.env`.
2. Replace the placeholder with a real random string (or several,
   comma-separated, if you want multiple valid keys):
   ```
   API_KEYS=my-super-secret-dev-key
   ```
3. `.env` is gitignored — it will never be pushed to GitHub.

### Why authentication now, on top of session_id?
Before this change, anyone who saw or guessed a `session_id` (e.g. a
short, predictable string) could read or continue someone else's
conversation just by sending it. Two things fix that:
- `session_id` is now a long random token from `/session/start` —
  effectively unguessable.
- Every request also needs the **same API key** that created the
  session. So even in the unlikely case a session_id leaked, it's
  useless without the matching key too.

In `/docs`, click the **Authorize** button (top right) and paste
your key once — it'll be sent automatically on every request you
try from that page.

### Why `session_id`?
A web API can serve many users at once, so there's no single shared
"conversation" like there was in the CLI version. Each `session_id`
gets its own memory window — send the same `session_id` on every
request from a given user/chat window so the bot remembers them, and
a different `session_id` per user so their histories don't mix.

**Note:** session memory is persisted to a local SQLite file
(`chat_memory.db`, created automatically on first run) via
LangChain's `SQLChatMessageHistory`. Restarting the server does
NOT clear conversations — send the same `session_id` again and
the bot picks up right where it left off. Use `POST /reset/{session_id}`
to permanently clear one session's history.

For multi-server / high-traffic setups, swap SQLite for Redis —
`main.py` has a commented-out snippet showing exactly what to
change (`RedisChatMessageHistory` instead of `SQLChatMessageHistory`,
plus a running Redis server). SQLite is the default here because it
needs no extra setup, which is enough for a local project or single
server.

## Next steps (once this works)
- Add a message limit / auto-expiry so `chat_memory.db` doesn't grow
  forever for long-lived sessions.
- Move from simple shared API keys to per-user accounts (e.g. with
  OAuth or JWT) if this ever needs real multi-user login instead of
  a handful of hardcoded keys.