"""
rag/pipeline.py
────────────────
Modern LangChain RAG pipeline using:
  - create_history_aware_retriever
  - create_retrieval_chain
  - RunnableWithMessageHistory   (avoids the chat_history type bug)
"""

import json, logging
import urllib.error, urllib.request
from collections import deque
from typing import Any, Dict, List, Optional

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ── prompts ───────────────────────────────────────────────────────────────────

CONTEXTUALISE_PROMPT = """\
Given the chat history and the latest user question (which may reference \
prior turns), rewrite the question into a self-contained standalone question. \
Do NOT answer it — just reformulate it. If it is already standalone, return as-is.\
"""

ANSWER_PROMPT = """\
You are a helpful document analyst. Answer using ONLY the retrieved context below.

Rules:
- Quote and paraphrase facts that are clearly stated in the context.
- If the user asks about a topic and ANY relevant info appears in the context, summarise it. \
  Do not say "not mentioned" if related information is present — extract what's there.
- If a topic is genuinely absent from the context, say: \
  "I couldn't find information about this in the uploaded documents."
- Cite sources as [Source: filename, p.N] when referring to specific facts.
- Be thorough but concise. Use bullet points for lists.

Retrieved context:
─────────────────
{context}
─────────────────
"""


# ── windowed history ──────────────────────────────────────────────────────────

class WindowedChatMessageHistory(BaseChatMessageHistory):
    """Keeps only the last `k` exchanges (k human + k AI = 2k messages)."""

    def __init__(self, k: int = 6):
        self._messages: deque = deque(maxlen=2 * k)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        return list(self._messages)

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages.clear()


# ── connectivity check ────────────────────────────────────────────────────────

def _check_ollama(base_url: str, model: str) -> None:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data  = json.loads(r.read())
            names = [m["name"].split(":")[0] for m in data.get("models", [])]
            if model not in names:
                logger.warning("Model '%s' not found. Pull with: ollama pull %s", model, model)
            else:
                logger.info("Ollama ready. Model '%s' available.", model)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {base_url}. Run: ollama serve\n{e}") from e


# ── pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    DEFAULT_SESSION = "default"

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name:    str   = "llama3.1",
        ollama_url:    str   = "http://localhost:11434",
        temperature:   float = 0.2,
        k_documents:   int   = 6,
        memory_window: int   = 6,
        search_type:   str   = "similarity",   # 'similarity' or 'mmr'
    ):
        self.vsm           = vector_store_manager
        self.model_name    = model_name
        self.ollama_url    = ollama_url
        self.temperature   = temperature
        self.k_documents   = k_documents
        self.memory_window = memory_window
        self.search_type   = search_type

        _check_ollama(self.ollama_url, self.model_name)

        self._histories: Dict[str, WindowedChatMessageHistory] = {}
        self.llm   = self._build_llm()
        self.chain = self._build_chain()

        logger.info(
            "RAGPipeline ready  model='%s'  k=%d  search='%s'  window=%d",
            self.model_name, self.k_documents, self.search_type, self.memory_window,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def query(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.vsm.is_ready:
            raise RuntimeError("Vector store is empty. Index documents first.")

        sid = session_id or self.DEFAULT_SESSION
        logger.info("Query [%s]: %s", sid, question[:80])

        result = self.chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": sid}},
        )

        return {
            "answer":           result.get("answer", ""),
            "source_documents": result.get("context", []),
            "question":         question,
        }

    def reset_memory(self, session_id: Optional[str] = None) -> None:
        sid = session_id or self.DEFAULT_SESSION
        if sid in self._histories:
            self._histories[sid].clear()
        logger.info("Memory cleared for session '%s'", sid)

    def update_settings(
        self,
        model_name:  Optional[str]   = None,
        ollama_url:  Optional[str]   = None,
        temperature: Optional[float] = None,
        k_documents: Optional[int]   = None,
        search_type: Optional[str]   = None,
    ) -> None:
        changed = False
        if model_name  and model_name  != self.model_name:  self.model_name  = model_name;  changed = True
        if ollama_url  and ollama_url  != self.ollama_url:  self.ollama_url  = ollama_url;  changed = True
        if temperature is not None and temperature != self.temperature: self.temperature = temperature; changed = True
        if k_documents and k_documents != self.k_documents: self.k_documents = k_documents; changed = True
        if search_type and search_type != self.search_type: self.search_type = search_type; changed = True
        if changed:
            _check_ollama(self.ollama_url, self.model_name)
            self.llm   = self._build_llm()
            self.chain = self._build_chain()
            logger.info("Pipeline rebuilt.")

    # ── private builders ──────────────────────────────────────────────────────

    def _build_llm(self) -> BaseChatModel:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=self.model_name, base_url=self.ollama_url, temperature=self.temperature)
        except ImportError:
            pass
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
            return ChatOllama(model=self.model_name, base_url=self.ollama_url, temperature=self.temperature)
        except ImportError:
            raise ImportError("Run: pip install langchain-ollama")

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = WindowedChatMessageHistory(k=self.memory_window)
        return self._histories[session_id]

    def _build_chain(self):
        retriever = self.vsm.get_retriever(k=self.k_documents, search_type=self.search_type)

        contextualise_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALISE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualise_prompt)

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(self.llm, answer_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
