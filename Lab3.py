import streamlit as st
from openai import OpenAI

# ---- Page setup ----
st.title("Lab 3: Streaming Chatbot")
st.write("Ask me anything! I'll explain it simply, then offer more info.")

# ---- OpenAI client from secrets (no text input) ----
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API Key not found in Streamlit secrets. Add OPENAI_API_KEY to secrets.toml.")
    st.stop()
client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-4o-mini"

# ---- System prompt: defines the bot's behavior (Part C) ----
# Explains simply (10-year-old level), and always asks "Do you want more info?"
SYSTEM_PROMPT = (
    "You are a friendly assistant for a 10-year-old child. "
    "Always explain answers in very simple words, short sentences, and easy examples. "
    "After you answer the user's question, always end your reply by asking: 'Do you want more info?'. "
    "If the user says yes, give more simple details about the SAME topic, then ask 'Do you want more info?' again. "
    "If the user says no, say okay and ask: 'What would you like to know next?'. "
    "Keep a warm, encouraging tone."
)

# ---- Initialize conversation memory in session_state (Part A) ----
# We store the FULL visible history for the chat display.
if "lab3_messages" not in st.session_state:
    st.session_state.lab3_messages = [
        {"role": "assistant", "content": "Hi! What would you like to learn about today?"}
    ]

# ---- Show the conversation so far ----
for msg in st.session_state.lab3_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
if user_input := st.chat_input("Type your message..."):

    # 1) Save + show the user's message
    st.session_state.lab3_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Build the buffer sent to the LLM (Part B):
    #    - always keep the system prompt
    #    - keep only the last 2 user messages AND their replies
    #    We take the last 4 items (2 user + 2 assistant) from history,
    #    then prepend the system prompt so it's never dropped.
    buffer = [{"role": "system", "content": SYSTEM_PROMPT}]
    buffer += st.session_state.lab3_messages[-4:]

    # 3) Stream the assistant's reply
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=buffer,
            stream=True,
        )
        response = st.write_stream(stream)

    # 4) Save the assistant's reply to memory
    st.session_state.lab3_messages.append({"role": "assistant", "content": response})