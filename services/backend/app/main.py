from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.settings import settings
from app.logging_utils import setup_logging, log_event
from app.core.embedder_client import EmbedderClient
from app.core.tokenizer_provider import TokenizerProvider
from app.core.vector_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService
from app.services.search_service import SearchService
from app.routers.ingest import router as ingest_router
from app.routers.search import router as search_router

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Ingestor & Semantic Search API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Swagger expects {"error": "..."} bodies.
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

@app.on_event("startup")
async def startup():
    log_event(logger, "startup_begin")

    # Remote embedder service (Docker DNS name: "embedder")
    embedder = EmbedderClient(base_url=settings.embedder_url)
    tokenizer_provider = TokenizerProvider(model_name=settings.embedding_model)

    # Qdrant can take a moment to become ready; retry a bit.
    last_err: Exception | None = None
    store = None
    for attempt in range(1, 31):
        try:
            store = QdrantVectorStore(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection_name=settings.collection_name,
                vector_size=settings.embedding_dim_fallback,
            )
            break
        except Exception as e:
            last_err = e
            log_event(logger, "qdrant_not_ready", attempt=attempt, url=settings.qdrant_url)
            time.sleep(1)

    if store is None:
        raise RuntimeError(f"Qdrant not reachable: {last_err}")

    app.state.embedder = embedder
    app.state.store = store
    app.state.ingestion_service = IngestionService(
        store=store,
        embedder=embedder,
        target_tokens=settings.target_tokens,
        overlap_tokens=settings.overlap_tokens,
        max_tokens=settings.max_tokens,
        tokenizer_provider=tokenizer_provider,
    )
    app.state.search_service = SearchService(store=store, embedder=embedder, top_k=settings.top_k)

    log_event(
        logger,
        "startup_ready",
        model=settings.embedding_model,
        dim=settings.embedding_dim_fallback,
        qdrant=settings.qdrant_url,
        collection=settings.collection_name,
    )

app.include_router(ingest_router)
app.include_router(search_router)

@app.get("/health")
async def health():
    return {"status": "ok"}
