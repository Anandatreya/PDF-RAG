"""
rag/vector_store.py
────────────────────
Manages a FAISS vector store:
  - Accepts any LangChain-compatible embeddings object (Ollama, OpenAI, etc.)
  - Persists the index to disk and reloads it automatically on startup
  - Exposes a retriever for use in the RAG pipeline

Dependencies:
    pip install faiss-cpu langchain langchain-community
"""

import logging
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Name of the FAISS index files saved inside persist_dir
_INDEX_NAME = "faiss_index"


class VectorStoreManager:
    """
    Wraps a FAISS vector store with persistence and a simple management API.

    Args:
        embeddings:   Any LangChain Embeddings instance (e.g. OllamaEmbeddings).
        persist_dir:  Directory where the FAISS index is saved and loaded from.
    """

    def __init__(self, embeddings: Embeddings, persist_dir: str = "./vector_store"):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self._store = None  # FAISS instance, None until documents are added

        os.makedirs(persist_dir, exist_ok=True)
        self._try_load()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the vector store contains at least one document."""
        return self._store is not None

    @property
    def document_count(self) -> int:
        """Number of vectors currently stored."""
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def add_documents(self, documents: List[Document]) -> None:
        """
        Embed and add documents to the store, then persist to disk.

        If a store already exists on disk the new documents are merged in
        (no duplicates check — re-run ingest to rebuild from scratch).

        Args:
            documents: List of LangChain Document objects to index.
        """
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError(
                "langchain_community is not installed.\n"
                "Run:  pip install langchain-community faiss-cpu"
            )

        logger.info("Embedding %d document chunks (this may take a while) ...", len(documents))

        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._store.add_documents(documents)

        self._save()
        logger.info("Vector store saved to '%s'  (%d vectors total)", self.persist_dir, self.document_count)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """
        Return the top-k most similar documents along with their similarity scores.

        Args:
            query: Natural language search query.
            k:     Number of results to return.

        Returns:
            List of (Document, score) tuples ordered by relevance.

        Raises:
            RuntimeError: If the vector store is empty.
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is empty. Index documents first.")
        return self._store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = 4, search_type: str = "mmr"):
        """
        Return a LangChain retriever backed by this FAISS store.

        Args:
            k:           Number of documents to retrieve.
            search_type: 'similarity' or 'mmr' (Maximal Marginal Relevance —
                         reduces redundancy in retrieved chunks).

        Returns:
            A LangChain VectorStoreRetriever.

        Raises:
            RuntimeError: If the vector store is empty.
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is empty. Index documents first.")

        search_kwargs = {"k": k}
        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def clear(self) -> None:
        """Delete all vectors from the in-memory store and remove saved files."""
        import shutil
        self._store = None
        index_path = os.path.join(self.persist_dir, _INDEX_NAME)
        if os.path.exists(index_path):
            shutil.rmtree(index_path, ignore_errors=True)
        logger.info("Vector store cleared.")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist the FAISS index to disk."""
        if self._store is None:
            return
        save_path = os.path.join(self.persist_dir, _INDEX_NAME)
        self._store.save_local(save_path)
        logger.debug("FAISS index saved to %s", save_path)

    def _try_load(self) -> None:
        """
        Attempt to load an existing FAISS index from disk.
        Silently skips if no saved index is found.
        """
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            return  # Will raise properly when add_documents is called

        load_path = os.path.join(self.persist_dir, _INDEX_NAME)
        if os.path.exists(load_path):
            try:
                self._store = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # Safe: our own saved index
                )
                logger.info(
                    "Loaded existing FAISS index from '%s'  (%d vectors)",
                    load_path, self.document_count,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load existing index (%s). Starting fresh.", exc)
                self._store = None
