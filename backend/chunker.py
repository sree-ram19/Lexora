"""
chunker.py — Sentence-aware chunker for Lexora.

Upgrade over naive word-split:
  - Uses NLTK sentence tokenizer → no mid-sentence cuts.
  - Builds chunks by accumulating sentences until word budget is hit.
  - Overlaps by carrying over trailing sentences (not arbitrary word slices).
  - Preserves all metadata from pdfloader (doc_name, page, source_type).
  - Assigns a deterministic chunk_id for dedup.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)


def _ensure_nltk():
    """Download NLTK punkt tokenizer data if not already present."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("[Chunker] Downloading NLTK punkt_tab...")
            nltk.download("punkt_tab", quiet=True)
        return nltk
    except ImportError:
        raise ImportError("nltk is required: pip install nltk")


def _sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    nltk = _ensure_nltk()
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)


def _word_count(text: str) -> int:
    return len(text.split())


def _make_chunk_id(doc_name: str, page: int, chunk_index: int, text: str) -> str:
    """
    Deterministic chunk ID based on content.
    Same chunk from same doc will always produce the same ID → safe to re-ingest.
    """
    raw = f"{doc_name}__p{page}__c{chunk_index}__{text[:80]}"
    return hashlib.md5(raw.encode()).hexdigest()


def chunk_text(
    pages: list[dict],
    chunk_size: int = None,
    overlap: int = None,
) -> list[dict]:
    """
    Sentence-aware chunking.

    Args:
        pages:       Output of pdfloader.extract_text().
        chunk_size:  Max words per chunk. Defaults to config.CHUNK_SIZE.
        overlap:     Number of overlap words (carried as trailing sentences).
                     Defaults to config.CHUNK_OVERLAP.

    Returns:
        List of chunk dicts:
          {
            "chunk_id":    str,   # deterministic content hash
            "text":        str,
            "page":        int,
            "doc_name":    str,
            "source_type": str,
            "chunk_index": int,   # sequential index within the doc
          }
    """
    # Import here to avoid circular at module load
    from config import CHUNK_SIZE, CHUNK_OVERLAP
    chunk_size = chunk_size or CHUNK_SIZE
    overlap    = overlap    or CHUNK_OVERLAP

    all_chunks = []

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        doc_name    = page.get("doc_name", "unknown")
        page_num    = page.get("page", 0)
        source_type = page.get("source_type", "text")

        sentences = _sentence_tokenize(text)
        if not sentences:
            continue

        # ── Build chunks by accumulating sentences ──────────────────────────
        current_sentences: list[str] = []
        current_words: int = 0
        chunk_index: int = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_words = _word_count(sent)

            if current_words + sent_words <= chunk_size:
                current_sentences.append(sent)
                current_words += sent_words
                i += 1
            else:
                if current_sentences:
                    chunk_text_str = " ".join(current_sentences)
                    chunk_id = _make_chunk_id(doc_name, page_num, chunk_index, chunk_text_str)
                    all_chunks.append({
                        "chunk_id":    chunk_id,
                        "text":        chunk_text_str,
                        "page":        page_num,
                        "doc_name":    doc_name,
                        "source_type": source_type,
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1

                    # ── Overlap: keep trailing sentences that fit in overlap budget ──
                    overlap_sentences = []
                    overlap_words = 0
                    for s in reversed(current_sentences):
                        w = _word_count(s)
                        if overlap_words + w <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_words += w
                        else:
                            break

                    current_sentences = overlap_sentences
                    current_words = overlap_words
                else:
                    # Single sentence exceeds chunk_size — emit it as-is
                    chunk_text_str = sent
                    chunk_id = _make_chunk_id(doc_name, page_num, chunk_index, chunk_text_str)
                    all_chunks.append({
                        "chunk_id":    chunk_id,
                        "text":        chunk_text_str,
                        "page":        page_num,
                        "doc_name":    doc_name,
                        "source_type": source_type,
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1
                    i += 1

        # ── Flush remaining sentences ───────────────────────────────────────
        if current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunk_id = _make_chunk_id(doc_name, page_num, chunk_index, chunk_text_str)
            all_chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text_str,
                "page":        page_num,
                "doc_name":    doc_name,
                "source_type": source_type,
                "chunk_index": chunk_index,
            })

    logger.info(f"[Chunker] Produced {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks
