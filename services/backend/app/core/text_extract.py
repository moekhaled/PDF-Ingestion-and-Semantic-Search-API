from __future__ import annotations

import io
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def extract_text_from_pdf_bytes(data: bytes, *, filename: str, request_id: str) -> Tuple[str,str]:
    """
    Returns (text, mode) where mode is 'pdf' or 'decode_fallback'. Open for extensibility, such as adding an OCR Model later.

    If PDF parsing fails, returns best-effort decoded text.
    """
    # Try pypdf first (lightweight).
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
        full = "\n\n".join(t for t in pages).strip()
        if full:
            logger.info(
                "pdf_parse_ok",
                extra={"extra": {"request_id": request_id, "filename": filename, "chars": len(full)}},
            )
            return full, "pdf"
        # Parsed but extracted nothing → still fallback (common with scanned PDFs)
        logger.warning(
            "pdf_parse_empty_text_fallback_decode",
            extra={"extra": {"request_id": request_id, "filename": filename}},
        )
    except Exception as e:
        logger.warning(
            "pdf_parse_failed_fallback_decode",
            extra={"extra": {"request_id": request_id, "filename": filename, "error": str(e)}})
    # Fallback: treat bytes as text (decodes potential data in a non-PDF bytes with .pdf extension).
    try:
        txt = data.decode("utf-8", errors="ignore").strip()
        if txt:
            logger.info(
                "decode_fallback_ok",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "filename": filename,
                        "chars": len(txt),
                    }
                },
            )
            return txt, "decode_fallback"
    except Exception:
        logger.error(
        "decode_fallback_failed",
        extra={"extra": {"request_id": request_id, "filename": filename}},
        exc_info=True,
        )
    return "", "decode_fallback"
