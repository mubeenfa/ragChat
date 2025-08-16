import faiss, pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import torch

# ---------------- CONFIG ----------------
VECTORSTORE_PATH = '../data/vectorstore/faiss_index'
METADATA_PATH = '../data/vectorstore/metadata.pkl'
EMBED_MODEL = "all-MiniLM-L6-V2"
# ----------------------------------------

index = faiss.read_index(VECTORSTORE_PATH)

# Load metadata and texts
with open(METADATA_PATH, "rb") as f:
    store = pickle.load(f)
metadata = store["metadata"]
texts = store["texts"]

# Load embedder (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)

def retrieve(query: str, k: int = 4) -> List[Tuple[str, Dict, float]]:
    """
    Retrieve top-k most similar chunks for a query.

    Returns: List of (chunk_text, metadata, distance)
    Lower distance = more similar for L2 FAISS index
    """
    # Embed the query
    query_vec = embedder.encode([query])

    # Search FAISS index
    distances, ids = index.search(query_vec, k)

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        if idx == -1:
            continue
        chunk_text = texts[idx]
        chunk_meta = metadata[idx]
        results.append((chunk_text, chunk_meta, float(dist)))

    return results

# Quick test
if __name__ == "__main__":
    query = "How to grow saffron?"
    top_chunks = retrieve(query, k=4)
    for text, meta, dist in top_chunks:
        print(f"Distance: {dist:.4f}  (lower is better)")
        print(f"Source: {meta.get('source')} | Page: {meta.get('page')}")
        print(f"Text: {text[:150]}...\n")