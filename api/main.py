from __future__ import annotations

import os
import threading
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import AskRequest, AskResponse, RebuildRequest
from api.rebuild import rebuild
from rag.engine import RAGEngine, RAGConfig

app = FastAPI(
    title= "RAG pour Puls-Events",
    version = "0.1.0",
    description="API RAG (FAISS + LangChain + Mistral) sur événements culturel à Paris",
)

_engine_lock = threading.Lock()
_engine: RAGEngine | None = None

def get_engine() -> RAGEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            cfg = RAGConfig(
                retrieve_k=int(os.getenv("RETRIEVE_K", "80")),
                max_events=int(os.getenv("MAX_EVENTS", "8")),
            )
            _engine = RAGEngine(cfg)
        return _engine


def reload_engine() -> None:
    global _engine
    with _engine_lock:
        cfg = RAGConfig(
            retrieve_k=int(os.getenv("RETRIEVE_K", "80")),
            max_events=int(os.getenv("MAX_EVENTS", "8")),
        )
        _engine = RAGEngine(cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    engine = get_engine()
    total_docs = engine.vs.index.ntotal if hasattr(engine.vs, "index") else None
    return {
        "embedding_model": engine.config.embedding_model,
        "llm_model": engine.config.mistral_model,
        "retrieve_k": engine.config.retrieve_k,
        "max_events": engine.config.max_events,
        "score_threshold": engine.config.score_threshold,
        "total_documents_indexed": total_docs,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    engine = get_engine()
    engine.config.retrieve_k = payload.top_k
    engine.config.max_events = payload.max_events
    return engine.ask(payload.question)


@app.post("/rebuild")
def rebuild_index(
    payload: RebuildRequest,
    x_rebuild_token: str | None = Header(default=None),
):
    token_required = os.getenv("REBUILD_TOKEN")
    if token_required and x_rebuild_token != token_required:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid rebuild token")

    try:
        rebuild(mode=payload.mode, force=payload.force)
        reload_engine()
        return {"status": "ok", "message": "Index rebuilt and engine reloaded", "mode": payload.mode}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})