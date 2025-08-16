import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

# ---------------- CONFIG ----------------
VECTORSTORE_PATH = '../data/vectorstore/faiss_index'
METADATA_PATH = '../data/vectorstore/metadata.pkl'
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast embeddings
LLM_PATH = "../models/llm/Phi-3-mini-4k-instruct-q4.gguf"
TOP_K = 3
# ----------------------------------------

console = Console()

# Load vectorstore
index = faiss.read_index(VECTORSTORE_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Load Phi-3 with llama.cpp
llm = Llama(
    model_path=LLM_PATH,
    n_ctx=4096,
    n_threads=6,  # Adjust for CPU cores
    n_batch=512,  # Larger batch = faster
    # temperature=0.3,
    # repeat_penalty=1.1,
    verbose=False
)

def retrieve_chunks(query, k=3):
    query_embedding = embed_model.encode(query)  # No list wrapper
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    chunks = [metadata[int(i)]['text'] for i in indices[0] if int(i) in metadata]
    return chunks

# # what is hydroponics?
# # Assistant: Hydroponics is a method of growing plants without soil, using mineral nutrient solutions in an aqueous solvent.
#
# # what equipments or tools do i need to get started with hydroponics?

def rag_answer(question, k=TOP_K):
    chunks = retrieve_chunks(question, k)
    context = "\n\n".join(chunks)   

    prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that "I don't know." DON'T MAKE UP ANYTHING.

    {context}

    Answer the Question based on the above context: {question}
    Answer:"""

    output = llm(prompt, max_tokens=300, stop=["\nQuestion:", "Context:"])
    answer = output["choices"][0]["text"].strip()
    return answer, chunks


if __name__ == "__main__":
    while True:
        query = input("\nAsk: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        answer, refs = rag_answer(query)
        print("\nAnswer:", answer)
        print("\nReferences:")
        for i, chunk in enumerate(refs, 1):
            print(f"{i}. {chunk[:150]}...")
        # for r in refs:
        #     print("-", r[:80], "...")


