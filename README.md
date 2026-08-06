# Simple Chatbot with Memory

A minimal command-line chatbot that remembers the conversation using
LangChain + a local Ollama model (no API key or cost).

## How the "memory" works
LLMs are stateless — they don't remember anything by themselves.
"Memory" just means: **we keep a growing list of every message
(yours + the bot's) and re-send the whole list on every turn.**
That's what `chat_history` does in `app.py`. Type `reset` in the
chat to clear it and start fresh.

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
- Swap the in-memory list for LangChain's `ConversationBufferWindowMemory`
  to auto-limit history length (keeps token usage down on long chats).
- Wrap this in FastAPI to serve it as a web API (matches your other
  chatbot project).
- Add persistent memory (SQLite) so it remembers across restarts,
  not just within one run.
