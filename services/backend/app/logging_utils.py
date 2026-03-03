from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any
import traceback

def setup_logging(level: str = "INFO") -> None:
    # Simple JSON logging to stdout (Docker-friendly).
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc_type"] = record.exc_info[0].__name__
                payload["exc"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()
            elif record.stack_info:
                payload["stack"] = record.stack_info
            # If caller added structured fields:
            extra = getattr(record, "extra", None)
            if isinstance(extra, dict):
                payload.update(extra)
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]

def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    # Keep this as a single line in logs (easy to grep).
    logger.info(event, extra={"extra": fields})
