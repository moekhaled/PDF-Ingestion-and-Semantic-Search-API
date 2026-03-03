from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.logging_utils import setup_logging, log_event
from app.embedding import FastEmbedder

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

setup_logging(LOG_LEVEL)
logger = logging.getLogger("embedder")

app = FastAPI(title="Embedder Service", version="1.0.0")

_embedder: Optional[FastEmbedder] = None
_embedder_error: Optional[str] = None


@app.on_event("startup")
def startup() -> None:
    global _embedder, _embedder_error
    try:
        _embedder = FastEmbedder(model_name=EMBEDDING_MODEL)
        _embedder_error = None
    except Exception as e:
        # No fallback: keep service up but NOT READY. Calls will 503.
        _embedder = None
        _embedder_error = f"{type(e).__name__}: {e}"
        logger.exception("embedder_startup_failed_no_fallback")


def _require_ready() -> FastEmbedder:
    if _embedder is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Embedder not ready",
                "model": EMBEDDING_MODEL,
                "startup_error": _embedder_error,
            },
        )
    return _embedder


class EmbedDocumentsRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class EmbedDocumentsResponse(BaseModel):
    vectors: List[List[float]]
    dim: int


class EmbedQueryRequest(BaseModel):
    text: str


class EmbedQueryResponse(BaseModel):
    vector: List[float]
    dim: int


@app.get("/healthz")
def healthz():
    ready = _embedder is not None
    return {
        "status": "ok",
        "ready": ready,
        "model": EMBEDDING_MODEL,
        "dim": (_embedder.dim if _embedder else None),
        "startup_error": (None if ready else _embedder_error),
    }


@app.post("/embed_documents", response_model=EmbedDocumentsResponse)
def embed_documents(req: EmbedDocumentsRequest, x_request_id: str | None = Header(default=None)):
    rid = x_request_id or os.urandom(8).hex()
    emb = _require_ready()

    t0 = time.perf_counter()
    log_event(
        logger,
        "embed_documents_start",
        request_id=rid,
        n_texts=len(req.texts),
        total_chars=sum(len(t) for t in req.texts),
        model=EMBEDDING_MODEL,
    )

    try:
        vectors = emb.embed_documents(req.texts)
    except Exception:
        logger.exception("embed_documents_failed", extra={"extra": {"request_id": rid}})
        raise HTTPException(status_code=500, detail={"error": "Embedding failed"})

    dt_ms = int((time.perf_counter() - t0) * 1000)
    log_event(logger, "embed_documents_done", request_id=rid, dim=emb.dim, ms=dt_ms)
    return {"vectors": vectors, "dim": emb.dim}


@app.post("/embed_query", response_model=EmbedQueryResponse)
def embed_query(req: EmbedQueryRequest, x_request_id: str | None = Header(default=None)):
    rid = x_request_id or os.urandom(8).hex()
    emb = _require_ready()

    t0 = time.perf_counter()
    log_event(logger, "embed_query_start", request_id=rid, chars=len(req.text), model=EMBEDDING_MODEL)

    try:
        vector = emb.embed_query(req.text)
    except Exception:
        logger.exception("embed_query_failed", extra={"extra": {"request_id": rid}})
        raise HTTPException(status_code=500, detail={"error": "Embedding failed"})

    dt_ms = int((time.perf_counter() - t0) * 1000)
    log_event(logger, "embed_query_done", request_id=rid, dim=emb.dim, ms=dt_ms)
    return {"vector": vector, "dim": emb.dim}