"""
embeddings.py — Embedding model singleton for Lexora.

- Loads model once at import time (module-level singleton).
- Supports batch encoding for performance.
- Model configurable via config.py / .env.
"""

import logging
from config import EMBED_MODEL

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy singleton — load model on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[Embeddings] Loading model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info("[Embeddings] Model loaded.")
    return _model


def get_embedding(text: str) -> list[float]:
    """Embed a single string. Returns a float list."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True).tolist()


def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Batch embed multiple strings — much faster than calling get_embedding() in a loop.
    Uses sentence-transformers built-in batching.
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]
