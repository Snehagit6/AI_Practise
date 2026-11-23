import streamlit as st
import faiss
import os
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from groq_api import generate_llm_api_response


# ================================
# VECTOR MEMORY SETUP
# ================================

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DB_PATH = "vector_store/index.faiss"
DOC_STORE_PATH = "vector_store/docs.pkl"

os.makedirs("vector_store", exist_ok=True)


def load_vector_store():

    if os.path.exists(VECTOR_DB_PATH):

        index = faiss.read_index(VECTOR_DB_PATH)

        with open(DOC_STORE_PATH, "rb") as f:
            docs = pickle.load(f)

    else:

        index = faiss.IndexFlatL2(384)
        docs = []

    return index, docs


def save_vector_store(index, docs):

    faiss.write_index(index, VECTOR_DB_PATH)

    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(docs, f)


def add_to_vector_memory(text):

    embedding = EMBED_MODEL.encode([text])
    index, docs = load_vector_store()

    index.add(np.array(embedding).astype("float32"))

    docs.append(text)

    save_vector_store(index, docs)


def retrieve_similar(query, k=3):

    index, docs = load_vector_store()

    if len(docs) == 0:
        return []

    embedding = EMBED_MODEL.encode([query])

    D, I = index.search(np.array(embedding).astype("float32"), k)

    results = []

    for idx in I[0]:
        if idx < len(docs):
            results.append(docs[idx])

    return results


# ================================
# STREAMLIT UI
# ================================

st.set_page_config(page_title="AI Code Generator", layout="wide")

st.title("🧠 AI Python Code Generator Assistant")


# ================================
# SESSION MEMORY
# ================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a Python expert helping generate functions."
        }
    ]


# ================================
# STRUCTURED MEMORY
# ================================

if "task_memory" not in st.session_state:

    st.session_state.task_memory = {
        "function_description": "",
        "generated_code": "",
        "documentation": "",
        "tests": ""
    }


# ================================
# DISPLAY CHAT HISTORY
# ================================

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])

    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])


# ================================
# USER INPUT
# ================================

user_input = st.chat_input("Describe the function you want")


if user_input:

    st.chat_message("user").write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # ================================
    # VECTOR RETRIEVAL
    # ================================

    similar_context = retrieve_similar(user_input)

    retrieved_text = "\n\n".join(similar_context)


    # ================================
    # BUILD PROMPT
    # ================================

    task_state = st.session_state.task_memory

    system_context = f"""
Previous knowledge:

{retrieved_text}

Current task state:

{task_state}

Generate Python code accordingly.
"""


    messages = st.session_state.messages + [
        {"role": "system", "content": system_context}
    ]


    # ================================
    # LLM CALL
    # ================================

    response = generate_llm_api_response(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages
    )


    # ================================
    # SAVE RESPONSE
    # ================================

    st.chat_message("assistant").write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


    # ================================
    # UPDATE STRUCTURED MEMORY
    # ================================

    if "def " in response:

        st.session_state.task_memory["generated_code"] = response


    # ================================
    # SAVE TO VECTOR MEMORY
    # ================================

    add_to_vector_memory(response)


# ================================
# SIDE PANEL MEMORY VIEW
# ================================

with st.sidebar:

    st.header("📦 Task Memory")

    st.json(st.session_state.task_memory)

    st.header("📚 Vector Memory Stats")

    index, docs = load_vector_store()

    st.write(f"Stored code examples: {len(docs)}")