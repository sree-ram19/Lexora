"""
config.py — Central configuration for Lexora backend.
All tunables live here. Override via .env or environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_MAX_TOKENS   = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_TEMPERATURE  = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
GROQ_TIMEOUT_SEC  = int(os.getenv("GROQ_TIMEOUT_SEC", "30"))
GROQ_MAX_RETRIES  = int(os.getenv("GROQ_MAX_RETRIES", "3"))

# ── Embeddings ──────────────────────────────────────────────────────────────
EMBED_MODEL       = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Reranker ────────────────────────────────────────────────────────────────
RERANKER_MODEL    = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N    = int(os.getenv("RERANKER_TOP_N", "5"))

# ── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K         = int(os.getenv("RETRIEVAL_TOP_K", "20"))   # fetch wide
RETRIEVAL_MIN_SCORE     = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.30"))  # distance threshold
BM25_WEIGHT             = float(os.getenv("BM25_WEIGHT", "0.4"))    # hybrid blend weight
VECTOR_WEIGHT           = float(os.getenv("VECTOR_WEIGHT", "0.6"))

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "400"))   # words per chunk
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "60"))

# ── OCR ─────────────────────────────────────────────────────────────────────
OCR_ENGINE        = os.getenv("OCR_ENGINE", "tesseract")  # "tesseract" | "easyocr" | "auto" (tesseract only for speed)
OCR_LANG          = os.getenv("OCR_LANG", "en")
OCR_DPI           = int(os.getenv("OCR_DPI", "150"))  # reduced from 300 for 4x speedup
SCANNED_TEXT_THRESHOLD = int(os.getenv("SCANNED_TEXT_THRESHOLD", "100"))  # increased to skip weak scans

# ── Vector DB ───────────────────────────────────────────────────────────────
CHROMA_PATH       = os.getenv("CHROMA_PATH", "./db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "lexora")

# ── Diagrams ────────────────────────────────────────────────────────────────
DIAGRAM_OUTPUT_DIR = os.getenv("DIAGRAM_OUTPUT_DIR", "./outputs/diagrams")
os.makedirs(DIAGRAM_OUTPUT_DIR, exist_ok=True)
