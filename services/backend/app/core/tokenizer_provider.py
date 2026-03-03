from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TokenizerProvider:
    model_name: str
    use_fast: bool = True

    _tokenizer: Optional[Any] = None
    _attempted: bool = False

    def get(self):
        """
        Lazy-load tokenizer once. Never raises (so ingestion won't fail just because
        HF download/cache fails). Returns None if unavailable.
        """
        if self._attempted:
            return self._tokenizer

        self._attempted = True
        try:
            from transformers import AutoTokenizer  # type: ignore
            logger.info("tokenizer_load_start", extra={"extra": {"model": self.model_name}})
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=self.use_fast)
            logger.info("tokenizer_load_done", extra={"extra": {"model": self.model_name}})
        except Exception:
            logger.exception("tokenizer_load_failed", extra={"extra": {"model": self.model_name}})
            self._tokenizer = None

        return self._tokenizer