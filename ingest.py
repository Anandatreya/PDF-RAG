"""
ingest.py — Command-line PDF batch ingestion script (Ollama Edition)
─────────────────────────────────────────────────────────────────────
Run this to pre-index a folder of PDFs before launching the Streamlit app.
The resulting FAISS index is saved to ./vector_store/ and auto-loaded by app.py.

Embeddings are generated locally using Ollama (llama3.1 by default).
Make sure Ollama is running:  ollama serve
And the model is pulled:      ollama pull llama3.1

Usage:
    python ingest.py --dir ./docs
    python ingest.py --dir ./docs --chunk-size 1000 --overlap 150
    python ingest.py --file report.pdf --file manual.pdf
    python ingest.py --dir ./docs --ollama-model nomic-embed-text
    python ingest.py --dir ./docs --ollama-url http://192.168.1.10:11434
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def build_ollama_embeddings(model: str, base_url: str):
    """
    Instantiate OllamaEmbeddings from langchain_ollama.
    Falls back to langchain_community if langchain_ollama is not installed.

    Args:
        model:    Ollama model name (e.g. 'llama3.1', 'nomic-embed-text').
        base_url: Ollama server URL (e.g. 'http://localhost:11434').

    Returns:
        An embeddings instance compatible with LangChain's FAISS integration.
    """
    try:
        from langchain_ollama import OllamaEmbeddings
        logger.info("Using langchain_ollama.OllamaEmbeddings")
        return OllamaEmbeddings(model=model, base_url=base_url)
    except ImportError:
        pass

    try:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
        logger.warning(
            "langchain_ollama not found — falling back to langchain_community. "
            "Install the newer package:  pip install langchain-ollama"
        )
        return OllamaEmbeddings(model=model, base_url=base_url)
    except ImportError:
        raise ImportError(
            "Neither langchain_ollama nor langchain_community is installed.\n"
            "Run:  pip install langchain-ollama"
        )


def check_ollama_reachable(base_url: str, model: str) -> None:
    """
    Performs a quick connectivity check against the Ollama server.
    Exits early with a clear error if the server is not reachable.

    Args:
        base_url: Ollama server URL.
        model:    Model name to verify is available.
    """
    import urllib.request
    import urllib.error
    import json

    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            # Normalize: Ollama tags may include ":latest" suffix
            available_base = [n.split(":")[0] for n in available]
            if model not in available and model not in available_base:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s\n"
                    "Pull it with:  ollama pull %s",
                    model, available or "(none)", model,
                )
            else:
                logger.info("✓ Ollama reachable. Model '%s' is available.", model)
    except urllib.error.URLError as exc:
        logger.error(
            "Cannot reach Ollama at %s — is it running?\n"
            "Start it with:  ollama serve\n"
            "Error: %s",
            base_url, exc,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into the FAISS vector store using local Ollama embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input sources ────────────────────────────────────────────────────────
    parser.add_argument(
        "--dir", type=str,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--file", type=str, action="append", dest="files", default=[],
        help="Individual PDF file path (can repeat: --file a.pdf --file b.pdf)",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--chunk-size", type=int, default=800,
        help="Characters per chunk (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=120,
        help="Chunk overlap in characters (default: 120)",
    )

    # ── Ollama settings ───────────────────────────────────────────────────────
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama server base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model", type=str, default="llama3.1",
        help=(
            "Ollama model for embeddings (default: llama3.1). "
            "For dedicated embedding quality use: nomic-embed-text"
        ),
    )

    # ── Output / retrieval ────────────────────────────────────────────────────
    parser.add_argument(
        "--persist-dir", type=str, default="./vector_store",
        help="Output directory for FAISS index (default: ./vector_store)",
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="Number of results to return in smoke test (default: 4)",
    )
    parser.add_argument(
        "--test-query", type=str, default=None,
        help="Optional smoke-test query to run after indexing",
    )

    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not args.dir and not args.files:
        parser.error("Provide at least one --dir or --file argument.")

    # ── Check Ollama is running and model exists ───────────────────────────────
    logger.info("Checking Ollama server at %s ...", args.ollama_url)
    check_ollama_reachable(args.ollama_url, args.ollama_model)

    # ── Lazy imports (keeps --help fast even if deps are missing) ─────────────
    from rag.document_processor import DocumentProcessor
    from rag.vector_store import VectorStoreManager

    # ── Build embeddings ───────────────────────────────────────────────────────
    logger.info(
        "Initialising Ollama embeddings  model='%s'  url='%s'",
        args.ollama_model, args.ollama_url,
    )
    embeddings = build_ollama_embeddings(
        model=args.ollama_model,
        base_url=args.ollama_url,
    )

    # ── Process documents ─────────────────────────────────────────────────────
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    all_chunks = []

    # Process directory
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error("Directory not found: %s", args.dir)
            sys.exit(1)
        logger.info("Processing directory: %s", dir_path)
        chunks = processor.process_directory(str(dir_path))
        all_chunks.extend(chunks)
        logger.info("  → %d chunks from directory", len(chunks))

    # Process individual files
    for file_path in args.files:
        p = Path(file_path)
        if not p.exists():
            logger.warning("File not found, skipping: %s", file_path)
            continue
        logger.info("Processing file: %s", p)
        chunks = processor.process_pdf(str(p))
        all_chunks.extend(chunks)
        logger.info("  → %d chunks from %s", len(chunks), p.name)

    if not all_chunks:
        logger.error(
            "No chunks produced. Check that your PDFs are readable "
            "(scanned/image-only PDFs require OCR pre-processing)."
        )
        sys.exit(1)

    logger.info("Total chunks to embed and index: %d", len(all_chunks))

    # ── Build / update FAISS vector store ─────────────────────────────────────
    logger.info("Building FAISS index (this may take a while for large corpora) ...")
    vsm = VectorStoreManager(
        embeddings=embeddings,        # Pass the Ollama embeddings object directly
        persist_dir=args.persist_dir,
    )
    vsm.add_documents(all_chunks)

    logger.info("✓ Indexed %d chunks  →  %s", len(all_chunks), args.persist_dir)
    logger.info("Vector store size: %d vectors", vsm.document_count)

    # ── Optional smoke test ────────────────────────────────────────────────────
    if args.test_query:
        logger.info("Running smoke-test query: '%s'", args.test_query)
        results = vsm.similarity_search_with_score(args.test_query, k=args.k)
        if results:
            logger.info("Top %d results:", len(results))
            for i, (doc, score) in enumerate(results, 1):
                src = doc.metadata.get("source", "?")
                pg  = doc.metadata.get("page",   "?")
                logger.info(
                    "  [%d] %s p.%s  score=%.4f  preview='%s'",
                    i, src, pg, score, doc.page_content[:80],
                )
        else:
            logger.warning("Smoke test returned no results — check your model and index.")

    logger.info("\nDone. Launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
