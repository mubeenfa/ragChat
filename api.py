"""
FastAPI backend exposing your local RAG:
- POST /ask    → get answer + references
- POST /reload → reload FAISS/metadata/LLM without restarting
- GET  /health → readiness probe

Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from src.rag_pipeline import engine
from src.config import (
    API_HOST, API_PORT, TOP_K_DEFAULT, ALLOWED_ORIGINS
)

app = FastAPI(title="Local RAG API (M3)", version="1.0.0")

# Allow Streamlit (localhost:8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    k: int = Field(default=TOP_K_DEFAULT, ge=1, le=10)

class AskResponse(BaseModel):
    answer: str
    references: List[Dict[str, Any]]

@app.on_event("startup")
def on_startup() -> None:
    engine.load_all()

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/reload")
def reload_all() -> Dict[str, str]:
    engine.load_all()
    return {"status": "reloaded"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:

    answer, refs = engine.answer(req.question, k=req.k)

    # Pick the single best hit (lowest distance)
    best = None
    if refs:
        best = min(refs, key=lambda r: r.get("distance", float("inf")))

    cleaned = []
    if best:
        txt = str(best.get("text", ""))
        snippet = txt[:240] + ("..." if len(txt) > 240 else "")
        cleaned.append({
            "source": str(best.get("source", "unknown")),
            "page": best.get("page"),
            "distance": float(best.get("distance")) if best.get("distance") is not None else None,
            "snippet": snippet,
        })

    return AskResponse(answer=answer, references=cleaned)
