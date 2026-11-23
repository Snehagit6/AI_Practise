import streamlit as st
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage
import requests
import os

# --- Page config ---
st.set_page_config(page_title="WonderBot", layout="centered")
st.title("🤖 WonderBot")

# --- Sidebar: API Key and Model Selection ---
with st.sidebar:
    st.header("🔐 API Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    model = st.selectbox("Choose Model", ["llama3-8b-8192", "meta-llama/llama-4-scout-17b-16e-instruct"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

# --- Session State for Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Input box for user prompt ---
user_input = st.chat_input("Ask a question...")

if user_input:
    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar.")
    else:
        # Append user input to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Initialize Groq model using OpenAI-compatible ChatOpenAI
        payload = {
            "model": model, "messages": [{"role": "user", "content": user_input}]}
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"{api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=payload)
        response_content = response.json()["choices"][0]["message"]["content"]
        # return response.json()["choices"][0]["message"]["content"]
        # os.environ["OPENAI_API_KEY"] = api_key
        # llm = ChatOpenAI(
        #     model_name=model,
        #     temperature=temperature,
        #     openai_api_base="https://api.groq.com/openai/v1"
        # )
        #
        # # Run the model and get response
        # response = llm([HumanMessage(content=user_input)])

        # Append bot reply to history
        st.session_state.chat_history.append({"role": "assistant", "content": response_content})

# --- Display chat messages ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
