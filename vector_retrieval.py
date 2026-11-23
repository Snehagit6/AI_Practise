"""
This Vector retrieval process is via in-house TF-IDF as SentenceTransformer model call can be restricted \
in on-premise environments.
Vectorization used : NLP + scikit-learn.(enterprise-safe)
Instead of using a transformer embedding (like MiniLM or E5), you’ll use:

TfidfVectorizer or CountVectorizer → to turn text into numeric vectors,

then index those vectors in FAISS,

and perform semantic-like retrieval (keyword-weighted but still meaningful).

This process is purely Retrieval base.
"""


import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from daily_retrieval import load_notes

# CONFIG
# --------------------------
NOTES_PATH = "D:\\AI_Wonders\\data\\notes1.txt"           # your text file (one note per line or paragraph)
INDEX_PATH = "D:\\AI_Wonders\\data\\notes2.faiss.index"   # where we'll save the faiss index
METADATA_PATH = "D:\\AI_Wonders\\data\\notes_meta2.json"  # save the original text list & embeddings meta

# Readlines from notes1.txt
notes = load_notes(path=NOTES_PATH)

# Create TF-IDF embeddings
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_embeddings = vectorizer.fit_transform(notes).toarray().astype("float32")

# Create FAISS index
dimension = tfidf_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(tfidf_embeddings)
print(f"✅ FAISS index created with {index.ntotal} notes.")

query = input("Post your query: \n")

# Vectorize query
query_vec = vectorizer.transform([query]).toarray().astype("float32")
distances, indices = index.search(query_vec, k=3)

results = []
for dist, idx in zip(distances[0], indices[0]):
    if idx < 0 or idx >= len(notes):
        continue
    results.append({"note": notes[idx], "distance": float(dist)})

min_distance = min([result['distance'] for result in results])
print([result['note'] for result in results if result['distance'] == min_distance])

