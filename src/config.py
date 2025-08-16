"""
All configuration in one place to avoid hunting through code.
Adjust MODEL paths and parameters here.
"""

from pathlib import Path
import os

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
VSTORE_DIR = DATA_DIR / "vectorstore"

VECTORSTORE_PATH = VSTORE_DIR / "faiss_index"
METADATA_PATH = VSTORE_DIR / "metadata.pkl"

# Your local model (make sure file exists)
LLM_MODEL_PATH = PROJECT_ROOT / "models" / "llm" / "Phi-3-mini-4k-instruct-q4.gguf"
print(LLM_MODEL_PATH)

# Embeddings (must match what you used during ingestion)
EMBED_MODEL_NAME = "all-MiniLM-L6-V2"

# Retrieval
TOP_K_DEFAULT = 4

# LLM runtime parameters — tune to your machine (8 GB RAM)
LLM_CTX = 4096                # Phi-3-mini-4k supports 4k tokens
LLM_THREADS = os.cpu_count() or 8
LLM_BATCH = 256               # reduce if you see OOM on 8 GB
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 384          # cap generation length to keep latency reasonable

# API server
API_HOST = "0.0.0.0"
API_PORT = 8000

# CORS (Streamlit runs on 8501 by default)
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost",
    "http://127.0.0.1",
]
