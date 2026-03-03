from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FastEmbedder:
    """
    Uses fastembed TextEmbedding and must be able to load the model.
    """
    model_name: str

    def __post_init__(self) -> None:
        from fastembed import TextEmbedding  # type: ignore

        logger.info("embedder_init_start", extra={"extra": {"model": self.model_name}})
        self._model = TextEmbedding(model_name=self.model_name)

        # Warm-up once to force model availability + determine dimension
        sample = list(self._model.embed(["warmup"]))[0]
        self._dim = int(len(sample))
        logger.info("embedder_init_done", extra={"extra": {"model": self.model_name, "dim": self._dim}})

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = list(self._model.embed(texts))
        # fastembed returns iterables of floats; convert to plain lists for JSON
        return [list(map(float, e)) for e in embs]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]