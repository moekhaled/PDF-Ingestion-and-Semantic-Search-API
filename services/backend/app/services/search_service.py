from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.core.embedder_client import EmbedderClient
from app.core.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, store: QdrantVectorStore, embedder: EmbedderClient, top_k: int):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def search(self, query: str, request_id) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            raise ValueError("Query cannot be empty.")
        qvec = self.embedder.embed_query(q)
        hits = self.store.search(qvec, top_k=self.top_k)
        return [{"document": h.document, "score": h.score, "content": h.content} for h in hits]
