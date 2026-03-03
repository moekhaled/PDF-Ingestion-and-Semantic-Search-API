from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

def _normalize_text(text: str) -> str:
    # Normalize line endings and whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive spaces but keep newlines.
    text = re.sub(r"[ \t]+", " ", text)
    # Reduce 3+ newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _split_into_blocks(text: str) -> List[str]:
    # Basic paragraph-ish segmentation.
    blocks = [b.strip() for b in re.split(r"\n\n+", text) if b.strip()]
    return blocks

def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        # Approximation: word count.
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())
from typing import List

def chunk_text(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    tokenizer=None,
) -> List[str]:
    """
    Simple, robust chunking:
    - normalize
    - split into blocks by blank lines
    - pack blocks into ~target_tokens chunks
    - overlap by token budget
    """
    text = _normalize_text(text)
    if not text:
        return []

    blocks = _split_into_blocks(text)

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if not current:
            return
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
        current = []
        current_tokens = 0

    for block in blocks:
        bt = _count_tokens(block, tokenizer)
        # If block itself is too large, hard-split by characters as fallback.
        if bt > max_tokens:
            # Flush any existing chunk first.
            flush()
            ids = tokenizer(block, add_special_tokens=False).input_ids
            n = len(ids)
            for start in range(0, n, target_tokens):
                end = min(start + target_tokens, n)
                piece = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()
                if piece:
                    chunks.append(piece)
            continue

        # If adding the block exceeds target, flush.
        if current and (current_tokens + bt) > target_tokens:
            flush()

        current.append(block)
        current_tokens += bt

        # If we somehow exceed max_tokens, flush immediately.
        if current_tokens >= max_tokens:
            flush()

    flush()

    # Apply overlap (token-based if tokenizer exists; else block-based).
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: List[str] = []
    for idx, ch in enumerate(chunks):
        if idx == 0:
            overlapped.append(ch)
            continue
        if tokenizer is None:
            # Block-ish overlap: take last ~1 paragraph from previous chunk.
            paras = re.split(r"\n\n+", overlapped[-1])
            tail = paras[-1] if paras else ""
            merged = (tail + "\n\n" + ch).strip() if tail else ch
            overlapped.append(merged)
            continue

        # Token overlap: take last overlap_tokens tokens from previous chunk and prepend.
        try:
            prev = overlapped[-1]
            prev_tokens = tokenizer.encode(prev, add_special_tokens=False)
            if len(prev_tokens) > overlap_tokens:
                tail_tokens = prev_tokens[-overlap_tokens:]
            else:
                tail_tokens = prev_tokens
            tail_text = tokenizer.decode(tail_tokens)
            merged = (tail_text + "\n\n" + ch).strip()
            overlapped.append(merged)
        except Exception:
            overlapped.append(ch)

    return overlapped
