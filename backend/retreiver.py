"""
retriever.py — Hybrid retrieval + cross-encoder reranking for Lexora.

Pipeline:
  1. Vector search (ChromaDB cosine similarity) → top-K candidates
  2. BM25 keyword search → top-K candidates
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) → final top-N
  5. Score threshold filter (retrieval optimizer)

This gives you:
  - Semantic recall from embeddings
  - Exact keyword precision from BM25
  - Quality ranking from the reranker
"""

import logging
from config import (
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
)
from vectorstore import search as vector_search
from bm25index import search_bm25
from retrievaloptimizer import filter_chunks

logger = logging.getLogger(__name__)

# ── Cross-encoder reranker (lazy singleton) ──────────────────────────────────
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        from config import RERANKER_MODEL
        logger.info(f"[Reranker] Loading: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("[Reranker] Loaded.")
    return _reranker


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    k_rrf: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.
    RRF score = Σ 1/(k + rank_i) across all lists.
    Deduplicates by (doc_name, page, chunk_index).
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    def _key(c: dict) -> str:
        return f"{c.get('doc_name','?')}__p{c.get('page',0)}__c{c.get('chunk_index',0)}"

    for rank, chunk in enumerate(vector_results, start=1):
        key = _key(chunk)
        scores[key]    = scores.get(key, 0.0) + VECTOR_WEIGHT * (1.0 / (k_rrf + rank))
        chunk_map[key] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        key = _key(chunk)
        scores[key]    = scores.get(key, 0.0) + BM25_WEIGHT * (1.0 / (k_rrf + rank))
        if key not in chunk_map:
            chunk_map[key] = chunk

    merged = sorted(chunk_map.values(), key=lambda c: scores[_key(c)], reverse=True)
    for c in merged:
        c["rrf_score"] = round(scores[_key(c)], 6)

    return merged


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    k: int = None,
    doc_filter: str = None,
    use_reranker: bool = True,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query:        User query string.
        k:            Number of final chunks to return (after reranking).
        doc_filter:   Restrict to a specific doc_name.
        use_reranker: Set False to skip cross-encoder (faster, lower quality).

    Returns:
        List of chunk dicts ordered by relevance, each with:
          text, page, doc_name, source_type, score, reranker_score (if used)
    """
    k = k or RERANKER_TOP_N

    # 1. Vector search
    vec_results = vector_search(query, k=RETRIEVAL_TOP_K, doc_filter=doc_filter)
    logger.info(f"[Retriever] Vector search → {len(vec_results)} candidates")

    # 2. BM25 search
    bm25_results = search_bm25(query, k=RETRIEVAL_TOP_K, doc_filter=doc_filter)
    logger.info(f"[Retriever] BM25 search → {len(bm25_results)} candidates")

    # 3. RRF merge
    merged = _rrf_merge(vec_results, bm25_results)
    logger.info(f"[Retriever] After RRF merge → {len(merged)} candidates")

    # 4. Score threshold filter
    filtered = filter_chunks(merged)
    logger.info(f"[Retriever] After score filter → {len(filtered)} candidates")

    if not filtered:
        logger.warning("[Retriever] No chunks passed the score filter.")
        return []

    # 5. Cross-encoder reranking
    if use_reranker and len(filtered) > 1:
        reranker = _get_reranker()
        pairs = [(query, c["text"]) for c in filtered]
        re_scores = reranker.predict(pairs)

        for chunk, score in zip(filtered, re_scores):
            chunk["reranker_score"] = round(float(score), 4)

        filtered.sort(key=lambda c: c["reranker_score"], reverse=True)
        logger.info(f"[Retriever] After reranking → returning top {k}")

    return filtered[:k]
