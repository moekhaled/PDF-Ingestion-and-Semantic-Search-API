from __future__ import annotations

from pydantic import BaseModel
import os

class Settings(BaseModel):
    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "pdf_chunks")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim_fallback: int = int(os.getenv("EMBEDDING_DIM_FALLBACK", "384"))
    embedder_url: str = os.getenv("EMBEDDER_URL", "http://embedder:8001")

    # Chunking
    target_tokens: int = int(os.getenv("CHUNK_TARGET_TOKENS", "256"))
    overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
    max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", "512"))

    # Search
    top_k: int = int(os.getenv("TOP_K", "5"))

    # App
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
