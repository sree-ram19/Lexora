"""
test_pipeline.py — CLI entrypoint for Lexora backend.

Usage:
  # Ingest a PDF and start interactive Q&A
  python test_pipeline.py path/to/document.pdf

  # Ingest multiple PDFs
  python test_pipeline.py doc1.pdf doc2.pdf doc3.pdf

  # Query a specific document
  python test_pipeline.py --doc report2024 doc.pdf

  # Skip re-ingestion (document already indexed)
  python test_pipeline.py --no-ingest

  # Force re-ingestion even if already indexed
  python test_pipeline.py --force path/to/document.pdf

  # Disable diagram generation
  python test_pipeline.py --no-diagram path/to/document.pdf

  # List all indexed documents
  python test_pipeline.py --list-docs
"""

import argparse
import logging
import os
import sys

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lexora.pipeline")


def resolve_path(raw: str) -> str:
    """
    Robustly resolve a file path regardless of:
      - Spaces in filename or directory names
      - Windows backslashes vs forward slashes
      - Relative vs absolute paths
      - Accidental shell splitting (rejoins fragmented args)
    Returns the resolved absolute path string.
    """
    # Normalize slashes for the current OS
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    # Expand ~ and env vars, then make absolute
    expanded = os.path.expandvars(os.path.expanduser(normalized))
    return os.path.abspath(expanded)


def resolve_pdf_args(raw_args: list[str]) -> list[str]:
    """
    argparse splits 'My Report.pdf' into ['My', 'Report.pdf'] when unquoted.
    This function detects fragmented paths and rejoins them by checking
    which combinations actually exist on disk.

    Strategy:
      - Try each arg as-is first (quoted paths work fine).
      - If it doesn't exist, try joining it with the next arg(s) with a space
        until we find a valid file or exhaust options.
    """
    resolved = []
    i = 0
    while i < len(raw_args):
        candidate = raw_args[i]
        # Try progressively joining more tokens with spaces
        found = False
        for j in range(i, len(raw_args)):
            joined = " ".join(raw_args[i:j + 1])
            path   = resolve_path(joined)
            if os.path.isfile(path):
                resolved.append(path)
                i = j + 1
                found = True
                break
        if not found:
            # No valid file found — pass through as-is (error handled later)
            resolved.append(resolve_path(candidate))
            i += 1
    return resolved


def ingest(pdf_path: str, force: bool = False) -> str:
    """Ingest a PDF: load → chunk → store. Returns doc_name."""
    from pdfloader import extract_text
    from chunker import chunk_text
    from vectorstore import add_chunks, doc_exists
    from bm25index import invalidate

    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

    if not force and doc_exists(doc_name):
        logger.info(f"[Pipeline] '{doc_name}' already indexed — skipping ingestion. Use --force to re-ingest.")
        return doc_name

    logger.info(f"[Pipeline] Ingesting: {pdf_path}")
    pages  = extract_text(pdf_path, doc_name=doc_name)
    chunks = chunk_text(pages)
    added  = add_chunks(chunks, doc_name=doc_name, force=force)
    invalidate()  # BM25 index needs rebuild after new doc

    logger.info(f"[Pipeline] Ingestion complete: {added} chunks added for '{doc_name}'")
    return doc_name


def run_qa_loop(doc_filter: str = None, generate_diagram: bool = True):
    """Interactive Q&A loop."""
    from rag import answer_question

    print("\n" + "═" * 60)
    print("  LEXORA AI — Document Intelligence")
    if doc_filter:
        print(f"  Searching in: {doc_filter}")
    print("  Type 'exit' or Ctrl+C to quit.")
    print("  Type 'list' to see indexed documents.")
    print("  Type 'nodiagram' to toggle diagrams off/on.")
    print("═" * 60 + "\n")

    diagram_on = generate_diagram

    while True:
        try:
            q = input("Ask Lexora ▶ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Lexora] Goodbye.")
            break

        if not q:
            continue

        if q.lower() == "exit":
            print("[Lexora] Goodbye.")
            break

        if q.lower() == "list":
            from vectorstore import list_docs
            docs = list_docs()
            print(f"[Lexora] Indexed documents ({len(docs)}):")
            for d in docs:
                print(f"  • {d}")
            continue

        if q.lower() == "nodiagram":
            diagram_on = not diagram_on
            print(f"[Lexora] Diagrams {'enabled' if diagram_on else 'disabled'}.")
            continue

        print()
        result = answer_question(
            q,
            doc_filter=doc_filter,
            generate_diagram=diagram_on,
        )

        # ── Print answer ──────────────────────────────────────────────────
        print("─" * 60)
        print("ANSWER:")
        print(result["answer"])

        # ── Print sources ─────────────────────────────────────────────────
        if result["sources"]:
            print("\nSOURCES:")
            for src in result["sources"]:
                score_key = "score"
                print(
                    f"  • [{src['doc_name']} | p.{src['page']} | {src['source_type']}]"
                    f"  score={src.get(score_key, 0):.3f}"
                )
                print(f"    {src['preview']}")

        # ── Print diagram path ────────────────────────────────────────────
        if result["diagram_path"]:
            print(f"\nDIAGRAM: {result['diagram_path']}")

        # ── Print error if any ────────────────────────────────────────────
        if result["error"]:
            print(f"\n⚠ Error: {result['error']}")

        print("─" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Lexora — RAG pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdfs", nargs="*", help="PDF file(s) to ingest"
    )
    parser.add_argument(
        "--doc", type=str, default=None,
        help="Filter Q&A to a specific doc_name"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-ingestion even if document is already indexed"
    )
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="Skip ingestion — go straight to Q&A"
    )
    parser.add_argument(
        "--no-diagram", action="store_true",
        help="Disable diagram generation"
    )
    parser.add_argument(
        "--list-docs", action="store_true",
        help="List all indexed documents and exit"
    )

    args = parser.parse_args()

    # ── List docs mode ────────────────────────────────────────────────────
    if args.list_docs:
        from vectorstore import list_docs
        docs = list_docs()
        if not docs:
            print("No documents indexed yet.")
        else:
            print(f"Indexed documents ({len(docs)}):")
            for d in docs:
                print(f"  • {d}")
        sys.exit(0)

    # ── Ingestion ──────────────────────────────────────────────────────────
    if not args.no_ingest:
        if not args.pdfs:
            parser.error("Provide at least one PDF path, or use --no-ingest.")

        # Rejoin space-fragmented paths (e.g. unquoted: My Report.pdf)
        pdf_paths = resolve_pdf_args(args.pdfs)

        for pdf_path in pdf_paths:
            if not os.path.isfile(pdf_path):
                logger.error(
                    f"File not found: {pdf_path}\n"
                    f"  Tip: Filenames with spaces should be quoted:\n"
                    f"       python test_pipeline.py \"path/to/My Report.pdf\""
                )
                sys.exit(1)
            if not pdf_path.lower().endswith(".pdf"):
                logger.error(f"Not a PDF: {pdf_path}")
                sys.exit(1)
            ingest(pdf_path, force=args.force)

    # ── Q&A loop ───────────────────────────────────────────────────────────
    run_qa_loop(
        doc_filter=args.doc,
        generate_diagram=not args.no_diagram,
    )


if __name__ == "__main__":
    main()