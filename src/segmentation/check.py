"""Test section-aware retrieval"""
import faiss
import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Load
index = faiss.read_index("data/processed/faiss/faiss_index.bin")
with open("data/processed/embeddings/paragraph_ids.json") as f:
    para_ids = json.load(f)

db = sqlite3.connect("data/processed/indexed/paragraphs.db")
cursor = db.cursor()

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Test query
query = "What were the facts of the case?"
query_vec = model.encode([query], normalize_embeddings=True)

# Search
scores, indices = index.search(query_vec, k=10)

print(f"Query: {query}\n")
print("Top results with sections:")

for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    para_id = para_ids[idx]
    
    cursor.execute(
        "SELECT judgment_id, section, section_conf, text FROM paragraphs WHERE id = ?",
        (para_id,)
    )
    row = cursor.fetchone()
    
    print(f"\n[{i}] Score: {score:.3f} | Section: {row[1]} (conf={row[2]:.2f})")
    print(f"    Case: {row[0]}")
    print(f"    Text: {row[3][:100]}...")

db.close()
