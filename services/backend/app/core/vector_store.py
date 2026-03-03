from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class SearchHit:
    document: str
    score: float
    content: str

class QdrantVectorStore:
    def __init__(self, url: str, api_key: str | None, collection_name: str, vector_size: int):
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

        # Ensure collection exists with correct vector size
        existing = self.client.get_collections().collections
        names = {c.name for c in existing}
        if collection_name not in names:
            logger.info("Creating Qdrant collection '%s' (size=%d)", collection_name, vector_size)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            # Optionally verify size
            try:
                info = self.client.get_collection(collection_name)
                size = info.config.params.vectors.size  # type: ignore
                if int(size) != int(vector_size):
                    raise ValueError(f"Existing collection '{collection_name}' has vector size {size}, expected {vector_size}.")
            except Exception as e:
                logger.warning("Could not verify collection vector size: %s", e)

    def upsert(self, points: List[Dict[str, Any]]) -> None:
        """
        points: list of {id, vector, payload}
        """
        from qdrant_client.http.models import PointStruct

        ps = [PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points]
        try:
            self.client.upsert(collection_name=self.collection_name, points=ps)
        except Exception:
            logger.exception("Qdrant upsert failed", extra={"extra": {"collection": self.collection_name, "points": len(ps)}})
            raise
    def search(self, vector: List[float], top_k: int) -> List[SearchHit]:
        res = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        hits: List[SearchHit] = []
        for r in res:
            payload = r.payload or {}
            hits.append(
                SearchHit(
                    document=str(payload.get("document", "")),
                    content=str(payload.get("content", "")),
                    score=float(r.score),
                )
            )
        return hits
    def delete_by_document(self, document: str) -> None:
        """
        Delete all points whose payload has {"document": <document>}.
        Safe to call even if nothing matches.
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        flt = Filter(
            must=[
                FieldCondition(
                    key="document",
                    match=MatchValue(value=document),
                )
            ]
        )

        try:
            # Count existing chunks under the Filename if any and return it for logging
            before = self.client.count(collection_name=self.collection_name, count_filter=flt, exact=True).count
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=flt,
                wait=True,
            )
            return before
        except Exception:
            logger.exception(
                "Qdrant delete_by_document failed",
                extra={"extra": {"collection": self.collection_name, "document": document}},
            )
            raise
