"""
rag/document_processor.py
──────────────────────────
Loads PDF files and splits them into overlapping text chunks
ready for embedding and indexing into FAISS.

Dependencies:
    pip install pypdf langchain langchain-community
"""

import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles PDF loading and text chunking.

    Args:
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Load a single PDF and return a list of text chunks as Documents.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of LangChain Document objects with metadata
            (source filename and page number).
        """
        path = Path(file_path)
        logger.info("Loading PDF: %s", path.name)

        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            raise ImportError(
                "langchain_community is not installed.\n"
                "Run:  pip install langchain-community pypdf"
            )

        loader = PyPDFLoader(str(path))
        pages = loader.load()  # Each page becomes one Document

        if not pages:
            logger.warning("No text extracted from %s — it may be a scanned/image PDF.", path.name)
            return []

        # Split pages into smaller chunks
        chunks = self.splitter.split_documents(pages)

        # Ensure metadata carries the original filename (not the full path)
        for chunk in chunks:
            chunk.metadata["source"] = path.name

        logger.info("  %s  →  %d pages  →  %d chunks", path.name, len(pages), len(chunks))
        return chunks

    def process_directory(self, dir_path: str) -> List[Document]:
        """
        Process all PDF files found in a directory (non-recursive).

        Args:
            dir_path: Path to the directory containing PDFs.

        Returns:
            Combined list of Document chunks from all PDFs.
        """
        directory = Path(dir_path)
        pdf_files = sorted(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in: %s", dir_path)
            return []

        logger.info("Found %d PDF file(s) in %s", len(pdf_files), dir_path)

        all_chunks: List[Document] = []
        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_path))
                all_chunks.extend(chunks)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process %s: %s", pdf_path.name, exc)

        return all_chunks
