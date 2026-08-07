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
- `POST /chat` — send a message, get a reply
  ```json
  { "session_id": "peter-1", "message": "My name is Peter" }
  ```
  returns
  ```json
  { "reply": "Nice to meet you, Peter!" }
  ```
- `POST /reset/{session_id}` — clear one session's memory
- `GET /` — health check

### Why `session_id`?
A web API can serve many users at once, so there's no single shared
"conversation" like there was in the CLI version. Each `session_id`
gets its own memory window — send the same `session_id` on every
request from a given user/chat window so the bot remembers them, and
a different `session_id` per user so their histories don't mix.

**Note:** session memory currently lives in RAM (a plain Python dict
in `main.py`), so it clears if the server restarts. That's fine for
development; production would need a persistent store like Redis or
a database instead.

## Next steps (once this works)
- Add persistent memory (SQLite/Redis) so sessions survive server
  restarts, not just within one run.
- Add authentication so `session_id` can't be guessed/spoofed by
  other users.