import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle

from sqlalchemy.testing.suite.test_reflection import metadata
from tqdm import tqdm   # progress bars for loops and long operations
from rich.console import Console    # library for colorful status messages


# ---------------- CONFIG ----------------
DOCS_DIR = '../docs'
VECTORSTORE_PATH = '../data/vectorstore/faiss_index'
METADATA_PATH = '../data/vectorstore/metadata.pkl'
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_MODEL = "all-MiniLM-L6-V2"
# ----------------------------------------

# ----------------------------------------
# Metadata preserved — source filename + page number for each chunk.
# FAISS persistence — saves vectors locally for future searches.
# Metadata persistence — pickle file stores metadata list (parallel to FAISS index).
# Batch-friendly embedding — show_progress_bar=True to monitor long jobs.
# Safe directories — auto-create data/vectorstore/.
# ----------------------------------------


console = Console()  # instantiate
allDocs = []

# Load PDFs
console.print("[bold blue]📂 Loading PDFs...[/bold blue]")

for fileName in os.listdir(DOCS_DIR):
    if fileName.lower().endswith('.pdf'):
        pdf_path = os.path.join(DOCS_DIR,fileName)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()   # returns one Document per page

        # Metadata Preservation
        for i, page in enumerate(pages):
            page.metadata["source"] = fileName
            page.metadata["page"] = i+1

        allDocs.extend(pages)

console.print(f"[green]✅ Loaded {len(allDocs)} pages.[/green]")

# For RAG, a common setup is:
# Chunk size: ~800–1000 characters (or ~200–300 tokens)
# Overlap: 100–150 characters (to preserve context between chunks)

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

console.print("[bold blue]✂️ Splitting into chunks...[/bold blue]")
chunks = splitter.split_documents(allDocs)
console.print(f"[green]✅ Split into {len(chunks)} chunks.[/green]")


# Creating embeddings for our chunks so they can be stored and searched later.
console.print("[bold blue]🧠 Creating embeddings...[/bold blue]")
embedder = SentenceTransformer(EMBED_MODEL)
texts = [chunk.page_content for chunk in chunks]
metadata = [chunk.metadata for chunk in chunks]

embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
console.print(f"[green]✅ Created embeddings for {len(embeddings)} chunks.[/green]")


# Save to FAISS index + metadata (Facebook AI Similarity Search (a library for fast vector similarity search).)
console.print("[bold blue]💾 Saving FAISS index and metadata...[/bold blue]")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)    # L2 distance (Euclidean distance) to measure similarity.
index.add(embeddings)

os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
faiss.write_index(index, VECTORSTORE_PATH)

with open(METADATA_PATH, "wb") as f:
    # pickle.dump(metadata, f)
    pickle.dump({"metadata": metadata, "texts": texts}, f)


console.print(f"[green] Saved FAISS index to {VECTORSTORE_PATH}[/green]")
console.print(f"[green] Saved metadata to {METADATA_PATH}[/green]")