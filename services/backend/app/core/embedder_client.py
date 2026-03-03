from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class EmbedderClient:
    base_url: str
    timeout_s: float = 30.0
    retries: int = 10
    retry_sleep_s: float = 0.3

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(timeout=self.timeout_s)

    def _post_json(self, path: str, payload: dict, request_id: Optional[str]) -> dict:
        headers = {}
        if request_id:
            headers["x-request-id"] = request_id

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._client.post(f"{self.base_url}{path}", json=payload, headers=headers)
                # If embedder is up but not ready, it may return 503; retry a few times.
                if r.status_code in (502, 503, 504):
                    raise RuntimeError(f"Embedder temporary error {r.status_code}: {r.text}")
                r.raise_for_status()
                data = r.json()
                logger.info(
                    "embedder_call_ok",
                    extra={"extra": {"request_id": request_id, "path": path, "attempt": attempt, "ms": int((time.perf_counter()-t0)*1000)}},
                )
                return data
            except Exception as e:
                last_exc = e
                logger.warning(
                    "embedder_call_retry",
                    extra={"extra": {"request_id": request_id, "path": path, "attempt": attempt, "error": str(e)}},
                )
                time.sleep(self.retry_sleep_s)

        logger.exception(
            "embedder_call_failed",
            extra={"extra": {"request_id": request_id, "path": path}},
        )
        raise RuntimeError(f"Embedder request failed after retries: {last_exc}")

    def embed_documents(self, texts: List[str], request_id: Optional[str] = None) -> List[List[float]]:
        data = self._post_json("/embed_documents", {"texts": texts}, request_id=request_id)
        # Required key per our embedder service contract:
        vectors = data["vectors"]
        return vectors

    def embed_query(self, text: str, request_id: Optional[str] = None) -> List[float]:
        data = self._post_json("/embed_query", {"text": text}, request_id=request_id)
        vector = data["vector"]
        return vector