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

## Next steps (once this works)
- Wrap this in FastAPI to serve it as a web API (matches your other
  chatbot project).
- Add persistent memory (SQLite) so it remembers across restarts,
  not just within one run.