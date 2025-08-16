# GPT Prompt:
"""
Your Role: You are a senior engineer with extensive experience in building rag & llm based chat systems.
Understand, Analyse and save this context in memory.

Do not proceed and wait for my Instructions.
"""


# Local RAG — Detailed Implementation Plan

## Goal
Build a fully local Retrieval-Augmented Generation (RAG) system to load documents (PDFs, text, etc.), process them into embeddings, store them locally, retrieve relevant chunks for a query, and generate an answer using a locally run LLM. No external APIs.

---

## Step-by-Step Implementation

### Step 1 — Project Structure
```
project_root/
│
├── docs/                 # Source documents (PDFs, text, etc.)
├── data/
│   ├── processed/        # Processed text chunks
│   ├── embeddings/       # Saved embeddings + metadata
│   └── vectorstore/      # Vector DB storage (FAISS, Chroma, etc.)
│
├── models/               # Local LLM models
│   ├── embeddings/       # Local embedding models (sentence-transformers)
│   └── llm/               # LLM binaries/weights
│
├── src/
│   ├── __init__.py
│   ├── config.py         # All constants + paths
│   ├── ingest.py         # Load docs → chunk → embed → store
│   ├── query.py          # Retrieve relevant chunks
│   ├── chat.py           # RAG pipeline: retrieve + answer
│   └── utils.py          # Helpers (logging, cleaning, etc.)
│
├── rag_cli.py            # CLI for ingest/query/chat
├── requirements.txt
├── README.md
└── Dockerfile
```

---

### Step 2 — Tools & Libraries

#### Core Processing
- **LangChain** — document loaders, text splitters
- **PyPDFLoader** — PDF parsing
- **sentence-transformers** — local embeddings (e.g., `all-MiniLM-L6-V2`)
- **FAISS** — local vector database

#### Local LLM
Options:
- **llama.cpp** — CPU/GPU quantized `.gguf` models
- **GPT4All** — simple local model runner
- **transformers + bitsandbytes** — GPU-friendly, low-memory inference

#### Utils
- **tqdm** — progress bars
- **rich** — styled CLI output
- **dotenv** — manage configs
- **uvicorn/FastAPI** (optional) — API mode

---

### Step 3 — Step-by-Step Flow

#### Phase 1: Document Ingestion
1. **Load documents** from `docs/` using:
   - PDF: `PyPDFLoader`
   - Text/Markdown: `TextLoader`
2. **Chunk documents** with `RecursiveCharacterTextSplitter`
   - chunk_size: 800–1000 characters
   - chunk_overlap: 100–150 characters
3. **Generate embeddings**
   - Model: `"all-MiniLM-L6-V2"` or `"multi-qa-mpnet-base-dot-v1"`
4. **Save vector store**
   - Backend: FAISS
   - Save to `data/vectorstore/`

#### Phase 2: Query & Retrieval
1. Load vector store
2. Embed query
3. Perform similarity search (`k=4`)
4. Return top chunks with metadata

#### Phase 3: RAG Answer Generation
1. Pass retrieved chunks + query into a **prompt template**
2. Send prompt to local LLM
3. Stream answer back to user
4. (Optional) Include citations from chunk metadata

---

### Step 4 — Future Enhancements
- **Reranking** retrieved results
- **Multi-query expansion** for better recall
- **Hybrid search** (vector + keyword)
- **GUI** (Streamlit or Gradio)
- **Metadata filters** (date, source, etc.)
- **Incremental updates** without full re-ingestion
- **Voice Input**
---

## Example Prompt Template
```
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
```

---

## Minimum Hardware Recommendations
- **CPU-only**: 8+ cores, 16GB RAM (llama.cpp, 4-bit quantized)
- **GPU**: 6–8GB VRAM (transformers + bitsandbytes)
- Storage: 5–10 GB for models

---

## Milestones
1. **M1** — Ingest PDFs → FAISS → Query embeddings
2. **M2** — Integrate local LLM for answering
3. **M3** — CLI + API interface + GUI
4. **M4** — Optimizations


# from project root
pip install -r requirements.txt

# launch API
uvicorn src.api:app --host 127.0.0.1 --port 8000
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is hydroponics?", "k": 4}'

# Expected JSON
{
  "answer": "…your concise answer…",
  "references": [
    {
      "source": "Hydroponically Growth of Saffron Concept.pdf",
      "page": 4,
      "distance": 0.5879,
      "snippet": "Hydroponics is a method of growing plants without soil..."
    }
  ]
}

# Reload (after re-ingesting docs)
curl -X POST http://localhost:8000/reload

# Health
curl http://localhost:8000/health






