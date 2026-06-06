"""
rag.py — Orchestrator for Lexora RAG pipeline.

Upgrades over v1:
  - Groq call wrapped with tenacity retry (rate limit, timeout).
  - Model, max_tokens, temperature all from config.
  - Rich citations: (Doc: X | Page: N | Type: ocr/text).
  - Optional diagram generation: Groq first extracts structured data,
    diagrammer renders it as PNG.
  - Graceful error messages instead of crashes.
  - answer_question() returns a result dict, not a bare string.
"""

import os
import json
import logging
from groq import Groq, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    GROQ_TIMEOUT_SEC,
    GROQ_MAX_RETRIES,
)
from retreiver import retrieve_chunks
from diagrammer import detect_and_render

logger = logging.getLogger(__name__)

# ── Groq client ──────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file."
    )

_groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SEC)


# ── Retry-wrapped Groq call ──────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    stop=stop_after_attempt(GROQ_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_groq(messages: list[dict], max_tokens: int = None) -> str:
    """Make a Groq API call with retry on rate limit / timeout."""
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens or GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
    )
    return response.choices[0].message.content


# ── Prompt builders ──────────────────────────────────────────────────────────

def _build_answer_prompt(question: str, context_blocks: list[dict]) -> list[dict]:
    """Build messages for the main Q&A call."""
    context_str = "\n\n".join(
        f"[Doc: {c['doc_name']} | Page: {c['page']} | Source: {c.get('source_type','text')}]\n{c['text']}"
        for c in context_blocks
    )

    system = (
        "You are Lexora AI — a precise, citation-focused document intelligence assistant.\n"
        "Rules:\n"
        "1. Answer ONLY using the provided context. Do not hallucinate.\n"
        "2. If the answer is not in the context, reply exactly: 'Not found in the provided documents.'\n"
        "3. Cite every factual claim with (Doc: <name>, Page: <N>).\n"
        "4. Be concise. Prefer structured answers (bullet points, numbered lists) for complex responses.\n"
        "5. If the question asks for numbers, dates, or comparisons — extract them precisely."
    )

    user = f"Context:\n{context_str}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def _build_diagram_prompt(question: str, context_blocks: list[dict]) -> list[dict]:
    """
    Ask Groq to extract structured data from the context for diagram rendering.
    Returns JSON with diagram type + data, or {"type": "none"}.
    """
    context_str = "\n\n".join(c["text"] for c in context_blocks)

    system = (
        "You are a data extraction assistant. Given document context and a question, "
        "determine if the answer contains data that can be visualized as a chart or diagram.\n\n"
        "If yes, extract the data and return ONLY a valid JSON object (no markdown, no preamble) "
        "in ONE of these formats:\n\n"
        '{"type": "bar_chart", "title": "...", "labels": [...], "values": [...], "unit": "..."}\n'
        '{"type": "line_chart", "title": "...", "x_label": "...", "y_label": "...", '
        '"series": [{"name": "...", "x": [...], "y": [...]}]}\n'
        '{"type": "pie_chart", "title": "...", "labels": [...], "values": [...]}\n'
        '{"type": "table", "title": "...", "headers": [...], "rows": [[...], ...]}\n'
        '{"type": "timeline", "title": "...", "events": [{"date": "...", "label": "..."}]}\n\n'
        'If the data is not visualizable, return exactly: {"type": "none"}'
    )

    user = f"Context:\n{context_str}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ── Main entry point ─────────────────────────────────────────────────────────

def answer_question(
    question: str,
    doc_filter: str = None,
    generate_diagram: bool = True,
    use_reranker: bool = True,
) -> dict:
    """
    Full RAG pipeline: retrieve → answer → (optionally) diagram.

    Args:
        question:         User query.
        doc_filter:       Restrict retrieval to one doc_name.
        generate_diagram: If True, attempt to render a chart alongside the answer.
        use_reranker:     If True, cross-encoder reranks retrieved chunks.

    Returns:
        {
          "answer":       str,         # LLM text answer
          "sources":      list[dict],  # chunks used (with page, doc, score)
          "diagram_path": str | None,  # path to PNG if generated
          "error":        str | None,  # error message if something failed
        }
    """
    result = {
        "answer":       "",
        "sources":      [],
        "diagram_path": None,
        "error":        None,
    }

    # 1. Retrieve
    try:
        chunks = retrieve_chunks(
            question,
            doc_filter=doc_filter,
            use_reranker=use_reranker,
        )
    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        result["error"] = f"Retrieval error: {e}"
        return result

    if not chunks:
        result["answer"] = "Not found in the provided documents."
        return result

    result["sources"] = [
        {
            "doc_name":    c.get("doc_name"),
            "page":        c.get("page"),
            "source_type": c.get("source_type"),
            "score":       c.get("reranker_score", c.get("score", 0)),
            "preview":     c["text"][:120] + "...",
        }
        for c in chunks
    ]

    # 2. Answer
    try:
        messages = _build_answer_prompt(question, chunks)
        result["answer"] = _call_groq(messages)
    except RateLimitError:
        result["error"]  = "Groq rate limit reached. Please wait a moment and try again."
        result["answer"] = "Service temporarily unavailable — rate limit hit."
        return result
    except APITimeoutError:
        result["error"]  = f"Groq request timed out after {GROQ_TIMEOUT_SEC}s."
        result["answer"] = "Request timed out. Please try again."
        return result
    except APIStatusError as e:
        result["error"]  = f"Groq API error {e.status_code}: {e.message}"
        result["answer"] = "An API error occurred."
        return result
    except Exception as e:
        logger.error(f"[RAG] Unexpected LLM error: {e}")
        result["error"]  = str(e)
        result["answer"] = "An unexpected error occurred."
        return result

    # 3. Diagram (optional, non-blocking)
    if generate_diagram:
        try:
            diagram_messages = _build_diagram_prompt(question, chunks)
            diagram_json_str = _call_groq(diagram_messages, max_tokens=600)

            # Strip markdown fences if Groq wraps in ```json
            diagram_json_str = diagram_json_str.strip()
            if diagram_json_str.startswith("```"):
                lines = diagram_json_str.split("\n")
                diagram_json_str = "\n".join(
                    l for l in lines if not l.startswith("```")
                ).strip()

            structured = json.loads(diagram_json_str)
            result["diagram_path"] = detect_and_render(structured, question)

        except json.JSONDecodeError:
            logger.warning("[RAG] Diagram extraction returned invalid JSON — skipping diagram.")
        except Exception as e:
            logger.warning(f"[RAG] Diagram generation failed (non-fatal): {e}")

    return result
