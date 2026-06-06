"""
vectorstore.py — ChromaDB vector store for Lexora.

Upgrades:
  - Batch embedding (no per-chunk model call loop).
  - Deduplication: chunks already in DB (by chunk_id) are skipped.
  - Multi-document support: search can filter by doc_name.
  - Distance scores are returned so retrieval optimizer can threshold them.
  - doc_exists() to check before ingesting.
  - delete_doc() to remove a document's chunks.
"""

import logging
import chromadb
from embeddings import get_embedding, get_embeddings_batch
from config import CHROMA_PATH, CHROMA_COLLECTION, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)

# ── ChromaDB client ──────────────────────────────────────────────────────────
_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},   # cosine similarity
)


# ── Ingestion ────────────────────────────────────────────────────────────────

def doc_exists(doc_name: str) -> bool:
    """Return True if any chunk from this doc_name is already in the collection."""
    results = collection.get(where={"doc_name": doc_name}, limit=1)
    return len(results["ids"]) > 0


def get_existing_chunk_ids(doc_name: str) -> set[str]:
    """Return the set of chunk_ids already stored for a given doc_name."""
    results = collection.get(where={"doc_name": doc_name})
    return set(results["ids"])


def add_chunks(chunks: list[dict], doc_name: str = None, force: bool = False) -> int:
    """
    Embed and store chunks in ChromaDB.

    Args:
        chunks:   Output of chunker.chunk_text(). Each chunk must have chunk_id.
        doc_name: Override doc_name (uses chunk["doc_name"] by default).
        force:    If True, re-ingest even if doc already exists (deletes first).

    Returns:
        Number of chunks actually added (skips duplicates).
    """
    if not chunks:
        logger.warning("[VectorStore] add_chunks called with empty list.")
        return 0

    effective_doc = doc_name or chunks[0].get("doc_name", "unknown")

    if force:
        delete_doc(effective_doc)
        logger.info(f"[VectorStore] Force mode: deleted existing chunks for '{effective_doc}'")

    # Find which chunk_ids are new
    existing_ids = get_existing_chunk_ids(effective_doc)
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info(f"[VectorStore] '{effective_doc}' already fully indexed — skipping.")
        return 0

    logger.info(f"[VectorStore] Adding {len(new_chunks)} new chunks (skipping {len(chunks) - len(new_chunks)} duplicates)")

    # Batch embed
    texts = [c["text"] for c in new_chunks]
    embeddings = get_embeddings_batch(texts)

    ids        = [c["chunk_id"] for c in new_chunks]
    metadatas  = [
        {
            "page":        c.get("page", 0),
            "doc_name":    c.get("doc_name", effective_doc),
            "source_type": c.get("source_type", "text"),
            "chunk_index": c.get("chunk_index", 0),
        }
        for c in new_chunks
    ]

    # ChromaDB has a max batch size of 5461 — chunk if needed
    BATCH = 500
    for start in range(0, len(new_chunks), BATCH):
        collection.add(
            ids=ids[start:start + BATCH],
            embeddings=embeddings[start:start + BATCH],
            documents=texts[start:start + BATCH],
            metadatas=metadatas[start:start + BATCH],
        )

    logger.info(f"[VectorStore] Stored {len(new_chunks)} chunks for '{effective_doc}'")
    return len(new_chunks)


def delete_doc(doc_name: str) -> None:
    """Remove all chunks belonging to doc_name."""
    existing = collection.get(where={"doc_name": doc_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info(f"[VectorStore] Deleted {len(existing['ids'])} chunks for '{doc_name}'")


def list_docs() -> list[str]:
    """Return all unique doc_names currently indexed."""
    all_meta = collection.get(include=["metadatas"])
    names = {m.get("doc_name", "unknown") for m in all_meta["metadatas"]}
    return sorted(names)


# ── Search ───────────────────────────────────────────────────────────────────

def search(
    query: str,
    k: int = None,
    doc_filter: str = None,
) -> list[dict]:
    """
    Vector similarity search.

    Args:
        query:       Query string.
        k:           Number of results to return.
        doc_filter:  If set, restrict search to this doc_name.

    Returns:
        List of result dicts:
          {
            "text":        str,
            "page":        int,
            "doc_name":    str,
            "source_type": str,
            "chunk_index": int,
            "distance":    float,   # cosine distance (lower = more similar)
            "score":       float,   # 1 - distance (higher = more similar)
          }
    """
    k = k or RETRIEVAL_TOP_K
    q_emb = get_embedding(query)

    where = {"doc_name": doc_filter} if doc_filter else None

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "text":        doc,
            "page":        meta.get("page", 0),
            "doc_name":    meta.get("doc_name", "unknown"),
            "source_type": meta.get("source_type", "text"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance":    dist,
            "score":       round(1.0 - dist, 4),
        })

    return output
