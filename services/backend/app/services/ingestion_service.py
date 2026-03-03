from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from app.core.chunking import chunk_text
from app.core.embedder_client import EmbedderClient
from app.core.text_extract import extract_text_from_pdf_bytes
from app.core.tokenizer_provider import TokenizerProvider
from app.core.vector_store import QdrantVectorStore
from app.logging_utils import log_event

logger = logging.getLogger(__name__)


@dataclass
class IngestedFile:
    filename: str
    bytes_: bytes


class IngestionService:
    def __init__(
        self,
        store: QdrantVectorStore,
        embedder: EmbedderClient,
        target_tokens: int,
        overlap_tokens: int,
        max_tokens: int,
        tokenizer_provider: TokenizerProvider,
    ):
        self.store = store
        self.embedder = embedder
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens
        self.tokenizer_provider = tokenizer_provider

    def ingest_files(self, files: List[IngestedFile], request_id) -> List[str]:
        ingested_names: List[str] = []

        for f in files:
            filename = f.filename
            data = f.bytes_

            self._validate_pdf(filename)

            full_text, mode = extract_text_from_pdf_bytes(data, filename=filename, request_id=request_id)
            # If extraction yields no text, we DO NOT delete existing vectors for that filename.
            if not full_text:
                # Nothing to ingest; still count as ingested to satisfy contract.
                ingested_names.append(filename)
                continue

            self._delete_existing_chunks_if_any(filename, request_id)

            chunks = self._chunk(full_text)
            if not chunks:
                ingested_names.append(filename)
                continue

            vectors = self._embed_in_batches(chunks, request_id, batch_size=64)

            if len(vectors) != len(chunks):
                raise RuntimeError(
                    f"Embedding vector count mismatch: got {len(vectors)} for {len(chunks)} chunks"
                )

            points = self._build_points(filename, chunks, vectors)
            self.store.upsert(points)

            ingested_names.append(filename)

        return ingested_names

    # -------------------------
    # Helpers 
    # -------------------------

    def _validate_pdf(self, filename: str) -> None:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are accepted.")

    
    def _delete_existing_chunks_if_any(self, filename: str, request_id) -> int:
        # Delete happens AFTER successful extraction.
        deleted_chunks = self.store.delete_by_document(filename)

        if deleted_chunks > 0:
            log_event(
                logger,
                "ingest_replace_by_filename",
                request_id=request_id,
                document=filename,
                deleted_chunks=deleted_chunks,
                action="deleted_old_chunks_before_ingest",
            )

        return deleted_chunks

    def _chunk(self, full_text: str) -> List[str]:
        return chunk_text(
            full_text,
            target_tokens=self.target_tokens,
            overlap_tokens=self.overlap_tokens,
            max_tokens=self.max_tokens,
            tokenizer=self.tokenizer_provider.get(),
        )

    def _embed_in_batches(
        self, chunks: List[str], request_id, batch_size: int = 64
    ) -> List[List[float]]:
        # Fixed batching loop + extend results
        vectors: List[List[float]] = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_vectors = self.embedder.embed_documents(batch, request_id=request_id)
            vectors.extend(batch_vectors)

        return vectors

    def _build_points(
        self, filename: str, chunks: List[str], vectors: List[List[float]]
    ) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []

        for i, (content, vec) in enumerate(zip(chunks, vectors)):
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{i}"))
            payload = {
                "document": filename,
                "chunk_id": i,
                "content": content,
            }
            points.append({"id": pid, "vector": vec, "payload": payload})

        return points

        