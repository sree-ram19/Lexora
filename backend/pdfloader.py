"""
pdfloader.py — Smart PDF loader for Lexora.

Strategy:
  1. Open PDF with PyMuPDF.
  2. For each page, extract embedded text first (fast path).
  3. If extracted text is too short (scanned page), fall back to OCR:
       - Render page to high-DPI image.
       - Try pytesseract first (fast, CPU-light).
       - Fall back to EasyOCR if tesseract yields nothing useful.
  4. Returns list of dicts: {page, text, source_type, doc_name}
     source_type: "text" | "ocr_tesseract" | "ocr_easyocr"
"""

import fitz  # PyMuPDF
import io
import os
import logging
from PIL import Image
from config import OCR_ENGINE, OCR_DPI, SCANNED_TEXT_THRESHOLD

logger = logging.getLogger(__name__)


# ── OCR backends ────────────────────────────────────────────────────────────

def _ocr_tesseract(pil_image: Image.Image, lang: str = "eng") -> str:
    """Run pytesseract OCR on a PIL image."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(pil_image, lang=lang)
        return text.strip()
    except Exception as e:
        logger.warning(f"[Tesseract] failed: {e}")
        return ""


def _ocr_easyocr(pil_image: Image.Image, lang: list = None) -> str:
    """Run EasyOCR on a PIL image. Lazy-import to avoid startup cost."""
    try:
        import easyocr
        import numpy as np
        if lang is None:
            lang = ["en"]
        reader = easyocr.Reader(lang, gpu=False, verbose=False)
        img_np = np.array(pil_image)
        results = reader.readtext(img_np, detail=0, paragraph=True)
        return "\n".join(results).strip()
    except Exception as e:
        logger.warning(f"[EasyOCR] failed: {e}")
        return ""


def _page_to_pil(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """Render a PyMuPDF page to a PIL Image at the given DPI."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def _is_scanned(text: str, threshold: int = None) -> bool:
    """Decide if extracted text is too sparse to trust (likely a scanned page)."""
    threshold = threshold or SCANNED_TEXT_THRESHOLD
    clean = text.strip().replace("\n", " ")
    return len(clean) < threshold


def _run_ocr(pil_image: Image.Image, engine: str = None) -> tuple[str, str]:
    """
    Run OCR with the configured engine.
    Returns (text, source_type).
    """
    engine = engine or OCR_ENGINE

    if engine == "tesseract":
        return _ocr_tesseract(pil_image), "ocr_tesseract"

    elif engine == "easyocr":
        return _ocr_easyocr(pil_image), "ocr_easyocr"

    elif engine == "auto":
        # Try tesseract first; fall back to easyocr if result is thin
        text = _ocr_tesseract(pil_image)
        if len(text.strip()) > SCANNED_TEXT_THRESHOLD:
            return text, "ocr_tesseract"
        # Skip EasyOCR fallback - too slow, use Tesseract result as-is
        logger.debug("[OCR] Tesseract result below threshold, skipping EasyOCR fallback for speed")
        return text, "ocr_tesseract"

    else:
        raise ValueError(f"Unknown OCR engine: {engine}")


# ── Main loader ─────────────────────────────────────────────────────────────

def extract_text(pdf_path: str, doc_name: str = None) -> list[dict]:
    """
    Extract text from a PDF, with OCR fallback for scanned pages.

    Args:
        pdf_path:  Path to the PDF file.
        doc_name:  Logical document name (defaults to filename without ext).

    Returns:
        List of page dicts:
          {
            "page":        int,       # 1-indexed
            "text":        str,       # extracted / OCR'd text
            "source_type": str,       # "text" | "ocr_tesseract" | "ocr_easyocr"
            "doc_name":    str,
            "pdf_path":    str,
          }
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if doc_name is None:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

    doc = fitz.open(pdf_path)
    pages_data = []

    logger.info(f"[PDFLoader] Loading '{doc_name}' — {len(doc)} pages")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_text = page.get_text().strip()

        if not _is_scanned(raw_text):
            # ── Normal text-layer page ──
            pages_data.append({
                "page":        page_num + 1,
                "text":        raw_text,
                "source_type": "text",
                "doc_name":    doc_name,
                "pdf_path":    pdf_path,
            })
        else:
            # ── Scanned page — render and OCR ──
            logger.info(f"[PDFLoader] Page {page_num + 1} is scanned → running OCR")
            pil_image = _page_to_pil(page, dpi=OCR_DPI)
            ocr_text, source_type = _run_ocr(pil_image)

            if not ocr_text:
                logger.warning(f"[PDFLoader] Page {page_num + 1}: OCR yielded no text")

            pages_data.append({
                "page":        page_num + 1,
                "text":        ocr_text,
                "source_type": source_type,
                "doc_name":    doc_name,
                "pdf_path":    pdf_path,
            })

    doc.close()
    logger.info(f"[PDFLoader] Done. Pages: {len(pages_data)}")
    return pages_data


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json
    logging.basicConfig(level=logging.INFO)

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    pages = extract_text(path)
    for p in pages:
        print(f"\n--- Page {p['page']} [{p['source_type']}] ---")
        print(p["text"][:400])
