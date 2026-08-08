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
- `POST /auth/register` — create an account:
  ```json
  { "username": "peter", "password": "at-least-8-characters" }
  ```
- `POST /auth/login` — log in (form fields, not JSON — see below),
  returns a JWT:
  ```json
  { "access_token": "eyJhbGciOi...", "token_type": "bearer" }
  ```
- `POST /session/start` — requires `Authorization: Bearer <token>`.
  Returns a fresh, unguessable `session_id`:
  ```json
  { "session_id": "kR7f...(long random string)" }
  ```
- `POST /chat` — requires the token + a `session_id` from
  `/session/start`:
  ```json
  { "session_id": "kR7f...", "message": "My name is Peter" }
  ```
  returns
  ```json
  { "reply": "Nice to meet you, Peter!" }
  ```
- `POST /reset/{session_id}` — requires the token; clears and
  permanently deletes that session
- `POST /auth/logout` — requires the token; invalidates it
  immediately instead of waiting for it to expire on its own
- `GET /` — health check (no auth needed)

### Setting up your .env
1. Copy `.env.example` to a new file named `.env`.
2. Generate a real secret key and paste it in:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
   ```
   SECRET_KEY=<paste the generated value here>
   ```
3. `.env` is gitignored — it will never be pushed to GitHub.

### Using accounts in /docs
1. Open `/auth/register`, try it out, register a username/password.
2. Open `/auth/login` — note this one takes form fields (`username`,
   `password`), not raw JSON, because it follows the OAuth2 password
   flow FastAPI expects. Try it and copy the `access_token` from the
   response.
3. Click the green **Authorize** button (top right of `/docs`) and
   paste just the token (no need to type "Bearer", Swagger adds that
   for you). Now every request from that page includes it
   automatically.

### Why real accounts instead of shared API keys?
Shared keys meant everyone using the same key could see and reset
each other's sessions, and there was no way to know who did what.
With accounts:
- Each person registers their own username/password.
- Logging in proves who you are and issues a token that expires
  (`ACCESS_TOKEN_EXPIRE_MINUTES`, 60 by default) — so a leaked token
  stops working on its own after a while, unlike a permanent shared
  key.
- Sessions are tied to a username, so one person's login can never
  see or reset another person's conversation, even by accident.

### Why `session_id`?
A web API can serve many users at once, so there's no single shared
"conversation" like there was in the CLI version. Each `session_id`
gets its own memory window — send the same `session_id` on every
request from a given chat window so the bot remembers them.

**Note:** session memory is persisted to a local SQLite file
(`chat_memory.db`, created automatically on first run) via
LangChain's `SQLChatMessageHistory`. Restarting the server does
NOT clear conversations — send the same `session_id` again and
the bot picks up right where it left off. Use `POST /reset/{session_id}`
to permanently clear one session's history.

For multi-server / high-traffic setups, swap SQLite for Redis —
`main.py`'s earlier version had a commented-out snippet for
`RedisChatMessageHistory`; ask if you want that added back in on
top of this version.

### Message limit and auto-expiry
Two separate protections keep `chat_memory.db` from growing forever:
- **`MAX_MESSAGES_PER_SESSION`** (200 by default) — after every chat
  turn, only the most recent N messages are kept per session; older
  ones are deleted from SQLite. This is independent of `WINDOW_SIZE`,
  which controls how much of that history is sent to the LLM per
  request — trimming controls storage, `WINDOW_SIZE` controls what
  the model sees.
- **`SESSION_EXPIRY_DAYS`** (30 by default) — a session untouched for
  that long is deleted entirely (both its record and its messages).
  Cleanup runs once at server startup, then automatically every 24
  hours in the background while the server keeps running — no manual
  step needed.

Both numbers are constants near the top of `main.py` — change them
if 200 messages or 30 days doesn't fit your use case.

### Logout / early token revocation
JWTs are normally "stateless" — the server doesn't track them, so
there's no built-in way to invalidate one before it naturally
expires. To get around that, every token now carries a unique ID
(`jti`). Calling `POST /auth/logout` records that ID in a small
`revoked_tokens` table; from then on, that specific token is
rejected even though it hasn't technically expired yet. Useful if a
device is lost or a token leaks somewhere it shouldn't have.

Note this only revokes the one token you're currently using — if
you're logged in on two devices, logging out on one doesn't affect
the other's token. Old revocation records are pruned automatically
by the same background cleanup that handles session expiry, once
the token they refer to would have expired anyway.

### Rate limiting
`/chat` is capped at `CHAT_RATE_LIMIT` (10 requests/minute by
default) **per account**, not per IP address — so switching Wi-Fi
networks doesn't reset the limit, and multiple people on the same
network (e.g. a school connection) don't share one limit. Going over
it returns a `429 Too Many Requests` response. Adjust
`CHAT_RATE_LIMIT` in `main.py` (e.g. `"30/minute"`) if 10/minute is
too strict for testing.

## Next steps (once this works)
- Add a `/auth/logout-all` that revokes every token for a user at
  once (currently logout only invalidates the one token you sent).
- Add refresh tokens, so users don't have to log in again every
  `ACCESS_TOKEN_EXPIRE_MINUTES` — a short-lived access token plus a
  longer-lived refresh token is the usual pattern.