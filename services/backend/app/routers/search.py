from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.logging_utils import log_event
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter()

class SearchRequest(BaseModel):
    query: str

@router.post("/search/")
async def search(req: SearchRequest, request: Request):
    svc: SearchService = request.app.state.search_service
    request_id = request.headers.get("x-request-id") or os.urandom(8).hex()
    log_event(logger, "search_start", request_id=request_id, query_len=len(req.query or ""))

    try:
        results = svc.search(req.query, request_id)
        log_event(logger, "search_done", request_id=request_id, results=len(results))
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception:
        logger.exception("Search failed.")
        raise HTTPException(status_code=500, detail={"error": "Search processing failed."})
