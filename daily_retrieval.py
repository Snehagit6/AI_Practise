import os
import re
import json
import numpy as np
import logging
from datetime import datetime

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

from paths.global_dirs import LOGS
current_date = str(datetime.now()).replace(' ', '_').replace(':', '_')
logging.basicConfig(level=logging.INFO, filename=os.path.join(LOGS, os.path.basename(__file__).\
                                                              replace(".py", f"_{current_date}")), \
                    format=' %(asctime)s -  \%(levelname)s -  %(message)s')

from groq_api import generate_llm_api_response

# --------------------------
# CONFIG
# --------------------------
NOTES_PATH = "D:\\AI_Wonders\\data\\notes1.txt"           # your text file (one note per line or paragraph)
INDEX_PATH = "D:\\AI_Wonders\\data\\notes1.faiss.index"   # where we'll save the faiss index
METADATA_PATH = "D:\\AI_Wonders\\data\\notes_meta1.json"  # save the original text list & embeddings meta

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"    # small, fast embedder
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # example LLM (no HF auth required)
TOP_K = 1
TEMPERATURE = 0.2
MAX_AGENT_ITER = 3   # avoid infinite loops

load_dotenv()
API_KEY = os.getenv("GROK_API_KEY")
# --------------------------
# 1) Read notes.txt
# --------------------------


def load_notes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Create it with one note per line.")
    with open(path, "r", encoding="utf-8") as f:
        # split by lines, ignore empty lines; you can change to paragraph-chunking if needed
        notes = [line.strip() for line in f.readlines() if line.strip()]
    return notes

# --------------------------
# Build or load embeddings + FAISS index
# --------------------------
def build_and_save_index(notes, embed_model, index_path=INDEX_PATH, meta_path=METADATA_PATH):
    # Takes each sentence in notes
    #
    # Passes it through a pretrained Transformer model (MiniLM)
    #
    # Converts each sentence into a high-dimensional numeric vector that captures its meaning
    #
    # These vectors are called sentence embeddings.
    embeddings = embed_model.encode(notes, convert_to_numpy=True, show_progress_bar=True)
    print(embeddings.shape)

    # Dimensionality Extraction
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    # persist
    faiss.write_index(index, index_path)
    # save metadata: texts + (optionally) embeddings (embeddings can be large; we save texts + dim)
    meta = {"notes": notes, "dim": dim}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return index

def load_index(index_path=INDEX_PATH, meta_path=METADATA_PATH):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    notes = meta["notes"]
    return index, notes

# --------------------------
# 3) Retrieval tool (external tool for the agent)
# --------------------------
def retrieve_from_faiss(query, embed_model, index, notes, top_k=TOP_K, score_threshold=None):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(q_emb).astype('float32'), k=top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(notes):
            continue
        # Optionally apply a distance/similarity threshold
        if score_threshold is not None and dist > score_threshold:
            continue
        results.append({"text": notes[idx], "distance": float(dist)})
    print(f"Retrieved from faiss: {results}")
    logging.info(f"Retrieved from faiss: {results[0]['text']}")
    return results

# --------------------------
# 4) Helper: parse LLM's ReAct output for Action / Answer
# --------------------------
ACTION_RE = re.compile(r"Action:\s*retrieve\((?:\"|')(.+?)(?:\"|')\)", re.IGNORECASE)
ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)

def extract_action(text):
    m = ACTION_RE.search(text)
    return m.group(1).strip() if m else None

def extract_answer(text):
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None

# --------------------------
# 5) Initialize models (embedding + LLM)
# --------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Build or load index
if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    print("Building FAISS index from notes...")
    notes = load_notes(NOTES_PATH)
    index = build_and_save_index(notes, embed_model)
else:
    print("Loading existing FAISS index...")
    index, notes = load_index()

# Initialize LLM pipeline (text-generation). If your environment can't load the HF model,
# replace this with your local runner (Ollama, local llama.cpp wrapper, etc.)
print("Loading LLM pipeline (may take a while)...")



# Small wrapper to call the LLM with a ReAct-style system prompt
# SYSTEM_PROMPT = """
#     "You are a helpful personal memory assistant. You should mainly retrieve contents from the notes file at NOTES_PATH.Follow this exact format when you decide to use a tool:\n\n"
#     "Thought: <your internal reasoning>\n"
#     "Action: retrieve(\"<natural language query to the notes>\")   # only 'retrieve' actions allowed\n"
#     "Observation: <filled by the environment after an action>\n"
#     "Thought: <updated reasoning>\n"
#     "Answer: <final concise answer based only on the Observations>\n\n"
#     "If you don't need to call retrieve, you can directly produce Answer.\n"
#     "Be concise and only use facts from retrieved notes. If uncertain, say 'Not enough context.'\n"
# """

# SYSTEM_PROMPT: Goal of the AI Agent
SYSTEM_PROMPT = """You are an intelligent assistant that can think and act step by step.
Use this format strictly:

Thought: reason about the question
Action: choose one of [Retrieve, Answer]
Action Input: the exact query or context
Observation: result from the tool (if any)
Answer: your final answer to the user

If your Action == Retrieve, then use below tools strictly:
Available tools:

```json
{
    "retrieve_from_faiss": {
        "description": "Retrieve data from the faiss index from retrieve_from_faiss method",
        "parameters": {}
    }
```

If you cannot complete the task, explain why in the Final Answer.
Example: "Final Answer: I could not find a matching note for 'userquery'."
"""

# --------------------------
# 6) The Agentic loop orchestration
# --------------------------
def agentic_answer(user_query, max_iterations=MAX_AGENT_ITER):
    # Step 0: compose initial prompt
    context = ""
    context_prompt = f"{SYSTEM_PROMPT}\nUser: {user_query}\nThought:"
    prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{user_query}"}, \
              {"role": "assistant", "content": context}]

    for each in range(MAX_AGENT_ITER):
        # MAX_AGENT_ITER :->> Thinks → Acts → Observes → Updates → Repeats.
        llm_response = generate_llm_api_response(model=LLM_MODEL_NAME, messages=prompt)
        print("LLM (initial) output:\n", llm_response, "\n" + "-"*40)
        logging.info("LLM (initial) output:\n", llm_response, "\n" + "-"*40)
        print(f"Iteration:{each}")
        if "Action:" in llm_response:
            # Extract Action to determine if tool needs to be called
            action = llm_response.split("Action:")[1].split("\n")[0].strip()
            if "Retrieve" in action:
                query = llm_response.split("Action Input:")[1].split("\n")[0].strip()
                print(f"Action Input:{query}")
                retrieved = retrieve_from_faiss(query, embed_model, index, notes, top_k=TOP_K)
                observation = f"Observation: Retrieved notes: {retrieved}\n"
                context += f"{llm_response}\n{observation}\nThought:"
            elif "Answer" in action:
                print("\n Final Answer")
                print(llm_response.split('Answer:')[1].strip())
                break
        else:
            break



# --------------------------
# 7) Example usage
# --------------------------
if __name__ == "__main__":

    user_q = input("Please post your query:\n")
    retrieved = retrieve_from_faiss(query=user_q, embed_model=embed_model, index=index, notes=notes, top_k=TOP_K)
    print(f"FAISS Retrieved: {retrieved[0]['text']}")
    # print("Execute the Agentic Loop")
    # agentic_answer(user_q)
