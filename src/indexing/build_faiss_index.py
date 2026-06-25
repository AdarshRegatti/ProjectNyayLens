"""
NyayLens – FAISS Index Builder (Merged Embeddings)
"""

import faiss
import numpy as np
import json
from pathlib import Path


def build_faiss_index():
    print("=" * 70)
    print("NyayLens – Building FAISS Index")
    print("=" * 70)

    embeddings_dir = Path("data/processed/embeddings")
    output_dir = Path("data/processed/faiss")
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_file = embeddings_dir / "paragraph_embeddings.npy"
    ids_file = embeddings_dir / "paragraph_ids.json"
    meta_file = embeddings_dir / "embedding_metadata.json"

    # --- Load data ---
    print("\nLoading embeddings...")
    embeddings = np.load(embeddings_file)

    print("Loading paragraph IDs...")
    with open(ids_file, "r") as f:
        paragraph_ids = json.load(f)

    # --- Safety checks ---
    assert embeddings.shape[0] == len(paragraph_ids), (
        f"Mismatch: {embeddings.shape[0]} embeddings vs "
        f"{len(paragraph_ids)} paragraph IDs"
    )

    print(f"✓ Loaded {embeddings.shape[0]:,} vectors")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")

    # --- Normalize (cosine similarity) ---
    print("\nNormalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # --- Build FAISS index ---
    dim = embeddings.shape[1]
    
    # Switch to HNSW for rapid Approximate Nearest Neighbor search
    # M=32 is number of connections per layer, typical for dense models
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40  # Depth of search during index build
    index.hnsw.efSearch = 64        # Depth of search during inference

    print(f"Adding {embeddings.shape[0]:,} vectors to HNSW index...")
    index.add(embeddings)

    print(f"✓ FAISS index contains {index.ntotal:,} vectors")

    # --- Save index ---
    index_file = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_file))

    # --- Save FAISS metadata ---
    with open(meta_file, "r") as f:
        emb_meta = json.load(f)

    faiss_meta = {
        "index_type": "IndexHNSWFlat",
        "metric": "cosine_similarity",
        "dimension": dim,
        "total_vectors": int(index.ntotal),
        "embedding_model": emb_meta.get("model_name", "unknown"),
    }

    with open(output_dir / "faiss_metadata.json", "w") as f:
        json.dump(faiss_meta, f, indent=2)

    print(f"\n✓ FAISS index saved to: {index_file}")
    print("=" * 70)
    print("✓ FAISS Index Build Complete")
    print("=" * 70)


if __name__ == "__main__":
    build_faiss_index()
