"""
retrievaloptimizer.py — Post-retrieval filtering for Lexora.

Replaces the old stub (length-only filter) with:
  - Cosine score threshold (drops low-relevance chunks)
  - Minimum text length guard
  - Deduplication by near-identical text (Jaccard similarity)
"""

import logging
from config import RETRIEVAL_MIN_SCORE

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 40   # drop chunks shorter than this (OCR noise, headers)
JACCARD_DEDUP_THRESHOLD = 0.85   # drop if >85% token overlap with an already-kept chunk


def _jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def filter_chunks(chunks: list[dict], min_score: float = None) -> list[dict]:
    """
    Filter and deduplicate retrieved chunks.

    Args:
        chunks:    List of chunk dicts from retriever (must have "score" key).
        min_score: Minimum cosine similarity score to keep. Defaults to config value.

    Returns:
        Filtered, deduplicated list.
    """
    min_score = min_score if min_score is not None else RETRIEVAL_MIN_SCORE
    kept = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        # 1. Length guard
        if len(text) < MIN_CHUNK_CHARS:
            logger.debug(f"[Filter] Dropped (too short): '{text[:40]}'")
            continue

        # 2. Score threshold (use vector score if reranker score not yet assigned)
        score = chunk.get("score", chunk.get("rrf_score", 1.0))
        if score < min_score:
            logger.debug(f"[Filter] Dropped (low score {score:.3f}): '{text[:40]}'")
            continue

        # 3. Near-duplicate dedup
        is_dup = False
        for kept_chunk in kept:
            if _jaccard(text, kept_chunk["text"]) >= JACCARD_DEDUP_THRESHOLD:
                is_dup = True
                logger.debug(f"[Filter] Dropped (near-duplicate): '{text[:40]}'")
                break

        if not is_dup:
            kept.append(chunk)

    logger.info(f"[Filter] {len(chunks)} → {len(kept)} chunks after filtering")
    return kept
