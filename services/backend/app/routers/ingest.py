from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Union

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile

from app.logging_utils import log_event
from app.services.ingestion_service import IngestionService, IngestedFile

logger = logging.getLogger(__name__)
router = APIRouter()

def _read_dir_pdfs(dir_path: str) -> List[IngestedFile]:
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise ValueError("Provided directory path does not exist or is not a directory.")
    out: List[IngestedFile] = []
    for fp in sorted(p.glob("*.pdf")):
        out.append(IngestedFile(filename=fp.name, bytes_=fp.read_bytes()))
    return out

@router.post("/ingest/")
async def ingest(request: Request):
    """
    Accepts multipart/form-data with field name 'input', which can be:
      - UploadFile (single)
      - multiple UploadFiles
      - a text field (directory path)
    """
    svc: IngestionService = request.app.state.ingestion_service

    form = await request.form()
    items = form.getlist("input")

    request_id = request.headers.get("x-request-id") or os.urandom(8).hex()
    log_event(logger, "ingest_start", request_id=request_id, item_count=len(items))

    if not items:
        raise HTTPException(status_code=400, detail={"error": "No input provided."})

    # Directory path case:
    if len(items) == 1 and isinstance(items[0], str):
        try:
            files = _read_dir_pdfs(items[0])
            ingested = svc.ingest_files(files, request_id)
            log_event(logger, "ingest_done", request_id=request_id, files=len(ingested))
            return {"message": f"Successfully ingested {len(ingested)} PDF documents.", "files": ingested}
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})
        except Exception as e:
            logger.exception("Ingest directory failed.", extra={"extra": {"request_id": request_id}})
            raise HTTPException(status_code=500, detail={"error": "Failed to process uploaded file."})

    # File upload case:
    upload_files: List[UploadFile] = []
    for it in items:
        if isinstance(it, UploadFile):
            upload_files.append(it)
        else:
            # Mixed types not supported
            raise HTTPException(status_code=400, detail={"error": "Invalid input type."})

    ingested_files: List[IngestedFile] = []
    for uf in upload_files:
        filename = uf.filename or "uploaded.pdf"
        data = await uf.read()
        ingested_files.append(IngestedFile(filename=filename, bytes_=data))

    try:
        ingested = svc.ingest_files(ingested_files, request_id)
        log_event(logger, "ingest_done", request_id=request_id, files=len(ingested))
        return {"message": f"Successfully ingested {len(ingested)} PDF documents.", "files": ingested}
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception:
        logger.exception("Ingest failed.", extra={"extra": {"request_id": request_id}})
        raise HTTPException(status_code=500, detail={"error": "Failed to process uploaded file."})
