"""
Simple CLI Chatbot with Memory
--------------------------------
Uses LangChain + a local Ollama model so the bot remembers earlier
messages in the same conversation (short-term / session memory).

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Pull a model:   ollama pull llama3.2
3. Install Python deps: pip install -r requirements.txt
4. Run: python app.py
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ---- Config ----
MODEL_NAME = "llama3.2"   # change to any model you've pulled with `ollama pull`
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers concise."


def main():
    # Initialize the local LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

    # This list IS the memory — every message (yours and the bot's)
    # stays here and gets sent back on every turn, so the model has
    # full context of the conversation so far.
    chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

    print("Simple Chatbot (type 'exit' to quit, 'reset' to clear memory)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Bot: Goodbye!")
            break

        if user_input.lower() == "reset":
            chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
            print("Bot: Memory cleared.\n")
            continue

        if not user_input:
            continue

        # 1. Add the user's message to memory
        chat_history.append(HumanMessage(content=user_input))

        # 2. Send the FULL history (memory) to the model
        response = llm.invoke(chat_history)

        # 3. Add the bot's reply to memory too, so it remembers what it said
        chat_history.append(AIMessage(content=response.content))

        print(f"Bot: {response.content}\n")


if __name__ == "__main__":
    main()
