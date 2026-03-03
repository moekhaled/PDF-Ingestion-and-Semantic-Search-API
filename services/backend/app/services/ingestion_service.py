from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any,List,Dict


from app.core.chunking import chunk_text
from app.core.text_extract import extract_text_from_pdf_bytes
from app.core.vector_store import QdrantVectorStore

from app.core.embedder_client import EmbedderClient
from app.core.tokenizer_provider import TokenizerProvider
from app.logging_utils import log_event

import uuid

logger = logging.getLogger(__name__)

@dataclass
class IngestedFile:
    filename: str
    bytes_: bytes

class IngestionService:
    def __init__(self, store: QdrantVectorStore, embedder: EmbedderClient, target_tokens: int, overlap_tokens: int, max_tokens: int,  tokenizer_provider: TokenizerProvider):
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
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are accepted.")

            full_text, mode = extract_text_from_pdf_bytes(data, filename=filename, request_id=request_id)
            if not full_text:
                # Nothing to ingest; still count as ingested to satisfy contract.
                ingested_names.append(filename)
                continue
            # Delete this File's chunks from the Vector DB incase exists to avoid duplication
            deleted_chunks = self.store.delete_by_document(filename)
            if deleted_chunks>0:
                log_event(
                        logger,
                        "ingest_replace_by_filename",
                        request_id=request_id,
                        document=filename,
                        deleted_chunks=deleted_chunks,
                        action="deleted_old_chunks_before_ingest",
                    )
            chunks = chunk_text(
                full_text,
                target_tokens=self.target_tokens,
                overlap_tokens=self.overlap_tokens,
                max_tokens=self.max_tokens,
                tokenizer=self.tokenizer_provider.get(),
            )
            if not chunks:
                ingested_names.append(filename)
                continue
            
            vectors: list[list[float]] = []
            batch_size = 64  

            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                batch_vectors = self.embedder.embed_documents(batch, request_id=request_id)
                vectors.extend(batch_vectors)

            if len(vectors) != len(chunks):
                raise RuntimeError(f"Embedding vector count mismatch: got {len(vectors)} for {len(chunks)} chunks")

            points: List[Dict[str, Any]] = []
            for i, (content, vec) in enumerate(zip(chunks, vectors)):
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{i}"))
                payload = {
                    "document": filename,
                    "chunk_id": i,
                    "content": content,
                }
                points.append({"id": pid, "vector": vec, "payload": payload})

            self.store.upsert(points)
            ingested_names.append(filename)

        return ingested_names
