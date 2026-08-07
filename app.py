"""
Simple CLI Chatbot with Memory
--------------------------------
Uses LangChain + a local Ollama model. Memory is handled by
ConversationBufferWindowMemory, which only keeps the last `k`
exchanges instead of the entire conversation — so token usage
(and cost/latency, if you ever swap in a paid API) stays flat
even on very long chats.

Note: ConversationBufferWindowMemory is a deprecated "classic" memory
class — as of LangChain 1.0 it lives in the separate langchain-classic
package (still works fine, just no longer in the core langchain package).

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Pull a model:   ollama pull llama3.2
3. Install Python deps: pip install -r requirements.txt
4. Run: python app.py
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_classic.memory import ConversationBufferWindowMemory

# ---- Config ----
MODEL_NAME = "llama3.2"   # change to any model you've pulled with `ollama pull`
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers concise."
WINDOW_SIZE = 5            # how many user<->bot exchanges to remember (not total messages)


def main():
    # Initialize the local LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

    # ConversationBufferWindowMemory keeps only the last WINDOW_SIZE
    # (human, ai) exchange pairs. Once you go over that, the oldest
    # exchange is dropped automatically — you never have to trim
    # anything by hand.
    memory = ConversationBufferWindowMemory(k=WINDOW_SIZE, return_messages=True)

    print(f"Simple Chatbot (remembers last {WINDOW_SIZE} exchanges)")
    print("Type 'exit' to quit, 'reset' to clear memory\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Bot: Goodbye!")
            break

        if user_input.lower() == "reset":
            memory.clear()
            print("Bot: Memory cleared.\n")
            continue

        if not user_input:
            continue

        # 1. Add the user's message into the windowed memory
        memory.chat_memory.add_user_message(user_input)

        # 2. Pull out only the messages the window has kept, and
        #    prepend the system prompt (the system prompt isn't
        #    part of the window, so it never gets trimmed away).
        windowed_history = memory.load_memory_variables({})["history"]
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + windowed_history

        # 3. Send just that trimmed window to the model
        response = llm.invoke(messages)

        # 4. Save the bot's reply into memory too, so future turns
        #    see it (until it eventually ages out of the window)
        memory.chat_memory.add_ai_message(response.content)

        print(f"Bot: {response.content}\n")


if __name__ == "__main__":
    main()