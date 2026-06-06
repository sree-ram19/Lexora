"""
bm25index.py — In-memory BM25 index for Lexora hybrid retrieval.

Used alongside vector search to catch exact keyword matches that
dense embeddings often miss (acronyms, proper nouns, numbers).

The index is rebuilt per-query session from ChromaDB contents.
For large corpora, persist to disk — see TODO at bottom.
"""

import logging
import re
from config import CHROMA_COLLECTION, CHROMA_PATH

logger = logging.getLogger(__name__)

_bm25_index = None
_bm25_corpus: list[dict] = []   # parallel list of chunk dicts


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def build_index(chunks: list[dict] = None) -> None:
    """
    Build (or rebuild) the BM25 index.

    Args:
        chunks: List of chunk dicts (from vectorstore or chunker).
                If None, loads all chunks from ChromaDB.
    """
    global _bm25_index, _bm25_corpus

    from rank_bm25 import BM25Okapi

    if chunks is None:
        chunks = _load_all_from_chroma()

    if not chunks:
        logger.warning("[BM25] No chunks to index.")
        return

    _bm25_corpus = chunks
    tokenized = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized)
    logger.info(f"[BM25] Index built with {len(chunks)} chunks.")


def _load_all_from_chroma() -> list[dict]:
    """Load all documents from ChromaDB into a flat list."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(CHROMA_COLLECTION)
    result = col.get(include=["documents", "metadatas"])

    chunks = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        chunks.append({
            "text":        doc,
            "page":        meta.get("page", 0),
            "doc_name":    meta.get("doc_name", "unknown"),
            "source_type": meta.get("source_type", "text"),
            "chunk_index": meta.get("chunk_index", 0),
        })
    return chunks


def search_bm25(query: str, k: int = 20, doc_filter: str = None) -> list[dict]:
    """
    BM25 keyword search.

    Args:
        query:      Query string.
        k:          Number of top results to return.
        doc_filter: If set, restrict to this doc_name.

    Returns:
        List of result dicts with "bm25_score" added.
    """
    global _bm25_index, _bm25_corpus

    if _bm25_index is None:
        logger.info("[BM25] Index not built — building now from ChromaDB...")
        build_index()

    if _bm25_index is None:
        return []

    tokens = _tokenize(query)
    scores = _bm25_index.get_scores(tokens)

    # Pair scores with corpus
    scored = list(zip(scores, _bm25_corpus))

    # Filter by doc_name if requested
    if doc_filter:
        scored = [(s, c) for s, c in scored if c.get("doc_name") == doc_filter]

    # Sort descending and take top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    results = []
    for score, chunk in top:
        if score <= 0:
            continue
        results.append({**chunk, "bm25_score": round(float(score), 4)})

    return results


def invalidate() -> None:
    """Force rebuild on next search (call after ingesting new docs)."""
    global _bm25_index, _bm25_corpus
    _bm25_index = None
    _bm25_corpus = []
    logger.info("[BM25] Index invalidated — will rebuild on next search.")
