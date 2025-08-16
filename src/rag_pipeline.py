"""
RAG engine:
- Loads FAISS index + metadata
- Loads SentenceTransformer embeddings
- Loads local Phi-3 GGUF via llama.cpp
- Provides retrieve() and answer() with references (source, page, snippet, distance)

Assumes metadata.pkl was saved as: {"metadata": [...], "texts": [...]}
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from .config import (
    VECTORSTORE_PATH, METADATA_PATH, EMBED_MODEL_NAME,
    LLM_MODEL_PATH, LLM_CTX, LLM_THREADS, LLM_BATCH,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, TOP_K_DEFAULT,
)

class RAGEngine:
    def __init__(self) -> None:
        self.index = None
        self.metadata_rows: List[Dict[str, Any]] | None = None
        self.embedder: SentenceTransformer | None = None
        self.llm: Llama | None = None

    # ---------- Loaders ----------
    def load_vectorstore(self) -> None:
        # Load FAISS index
        self.index = faiss.read_index(str(VECTORSTORE_PATH))

        # Load metadata
        with open(METADATA_PATH, "rb") as f:
            store = pickle.load(f)

        # Normalize into a list of dicts: each item contains "text" + metadata fields
        if isinstance(store, dict) and "metadata" in store and "texts" in store:
            metas = store["metadata"]
            texts = store["texts"]
            self.metadata_rows = []
            for i in range(len(texts)):
                row_meta = metas[i] if isinstance(metas[i], dict) else {}
                self.metadata_rows.append({
                    **row_meta,
                    "text": texts[i],
                })
        elif isinstance(store, list):
            # Already a list of dicts
            self.metadata_rows = store
        else:
            raise ValueError(
                "metadata.pkl format unexpected. Expected {'metadata': [...], 'texts': [...]}."
            )

    def load_embedder(self) -> None:
        # If you have CUDA, you can pass device='cuda' for speed
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def load_llm(self) -> None:
        self.llm = Llama(
            model_path=str(LLM_MODEL_PATH),
            n_ctx=LLM_CTX,
            n_threads=LLM_THREADS,
            n_batch=LLM_BATCH,
            verbose=False,
        )

    def load_all(self) -> None:
        self.load_vectorstore()
        self.load_embedder()
        self.load_llm()

    # ---------- Core ops ----------
    def retrieve(self, question: str, k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        """
        Return top-k hits with: idx, text, source, page, distance
        """
        if self.index is None or self.metadata_rows is None or self.embedder is None:
            raise RuntimeError("RAGEngine not loaded. Call load_all() first.")

        # Encode query → shape (1, dim)
        q = self.embedder.encode(question, convert_to_numpy=True)
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)

        distances, ids = self.index.search(q, k)
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ids[0]):
            if idx == -1:
                continue
            if idx < 0 or idx >= len(self.metadata_rows):
                # Out-of-sync guard
                continue
            meta = self.metadata_rows[idx]
            hits.append({
                "idx": int(idx),
                "text": meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", None),
                "distance": float(distances[0][rank]),
            })
        return hits

    def _build_messages(self, question: str, hits: list[dict]) -> list[dict]:
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        system_msg = (
            "You are a concise, helpful assistant. "
            "Answer ONLY using the provided context. "
            "If the answer is not fully contained in the context, reply exactly: I don't know."
        )

        user_msg = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "- Answer in 1–3 sentences.\n"
            "- Do not include the words 'Output' or any special markers.\n"
            "- Do not mention the context explicitly.\n"
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]


    def answer(self, question: str, k: int = TOP_K_DEFAULT) -> tuple[str, list[dict]]:
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_all() first.")

        hits = self.retrieve(question, k=k)

        # If nothing retrieved, short-circuit with "I don't know."
        if not hits:
            return "I don't know.", []

        prompt = self._build_messages(question, hits)
        # print(prompt)
        out = self.llm.create_chat_completion(
            messages=prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stop=["\nQuestion:", "\nContext:"],
        )
        answer = out["choices"][0]["message"]["content"].strip()

        # Post-clean: remove accidental leading/trailing quote blocks or "Output=" artifacts if any
        if answer.lower().startswith("output="):
            answer = answer[len("output="):].strip()
        answer = answer.replace("Output:", "").replace("Output=", "").strip()

        return answer, hits


# Singleton for the API to import
engine = RAGEngine()
