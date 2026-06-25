"""
Test RAG retrieval system with SQLite
"""

import faiss
import json
import sqlite3
from sentence_transformers import SentenceTransformer

def test_retrieval():
    print("="*70)
    print("Testing RAG Retrieval")
    print("="*70)
    
    # Load FAISS index
    index = faiss.read_index("data/processed/faiss/faiss_index.bin")
    
    # Load paragraph IDs
    with open("data/processed/embeddings/paragraph_ids.json", encoding="utf-8") as f:
        para_ids = json.load(f)
    
    # Connect to SQLite
    conn = sqlite3.connect("data/processed/indexed/paragraphs.db")
    cursor = conn.cursor()
    
    # Load embedding model (MUST match indexing)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    print(f"✓ Ready with {index.ntotal:,} vectors\n")
    
    queries = [
        "right to privacy under Article 21",
        "anticipatory bail conditions",
        "burden of proof in criminal cases",
        "doctrine of legitimate expectation"
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print('='*70)
        
        query_vec = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        scores, indices = index.search(query_vec, k=3)
        
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(para_ids):
                continue
            
            para_id = para_ids[idx]
            
            cursor.execute(
                "SELECT judgment_id, page_no, text FROM paragraphs WHERE id = ?",
                (para_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                continue
            
            judgment_id, page_no, text = row
            
            print(f"\n[{rank}] Score: {score:.4f}")
            print(f"Source: {judgment_id} | Page: {page_no}")
            print(f"Text: {text[:300]}...")
    
    conn.close()

if __name__ == "__main__":
    test_retrieval()
