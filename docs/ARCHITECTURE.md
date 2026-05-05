# Architecture

This document explains how the four core files work together.

## High-level overview

```
                    ┌──────────────────────────┐
                    │       Browser UI         │
                    │  (served from app.py)    │
                    └────────────┬─────────────┘
                                 │ HTTP/JSON
                    ┌────────────▼─────────────┐
                    │      FastAPI server      │
                    │         (app.py)         │
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼──────────┐    ┌────────▼──────────┐   ┌─────────▼──────────┐
│ DocumentProcessor│    │ VectorStoreManager│   │    RAGPipeline     │
│                  │    │                   │   │                    │
│  PDF → chunks    │    │  chunks → vectors │   │ question → answer  │
│  (PyPDFLoader)   │    │   (FAISS, local)  │   │  (LangChain chain) │
└──────────────────┘    └────────┬──────────┘   └─────────┬──────────┘
                                 │                        │
                                 ▼                        ▼
                          ┌──────────────┐         ┌──────────────┐
                          │ FAISS index  │         │    Ollama    │
                          │  (on disk)   │         │  (local LLM) │
                          └──────────────┘         └──────────────┘
```

## File responsibilities

### `app.py`
The FastAPI server. Defines:
- **REST endpoints** (`/api/upload`, `/api/chat`, etc.)
- **The HTML UI** served from `/`
- **Module-level singletons** for the vector store and pipeline (shared across requests)

This is the only file with FastAPI/HTTP concerns. Everything else is pure Python.

### `rag/document_processor.py`
Wraps `PyPDFLoader` and `RecursiveCharacterTextSplitter`. Takes a file path, returns a list of `Document` objects with `.page_content` and `.metadata` (filename + page number).

**No state.** Pure transformation: PDF → chunks.

### `rag/vector_store.py`
Wraps FAISS. Responsibilities:
- Embed chunks via Ollama and store in FAISS
- Persist the index to disk
- Auto-load it on startup
- Expose a retriever for the pipeline

**Holds state** (the FAISS instance), but the state lives on disk between runs.

### `rag/pipeline.py`
The RAG brain. Builds a LangChain chain that:
1. Rewrites follow-up questions into standalone queries
2. Retrieves the top-k chunks from the vector store
3. Augments the system prompt with those chunks
4. Sends to Ollama for generation
5. Returns answer + source documents

**Holds session state** (conversation memory) but no document state.

## Data flow — indexing phase

```
User uploads file via POST /api/upload
        │
        ▼
DocumentProcessor.process_pdf(path)
        │  returns List[Document]
        ▼
VectorStoreManager.add_documents(chunks)
        │  embeds via Ollama → adds to FAISS → saves to disk
        ▼
Pipeline is invalidated (force rebuild on next chat)
```

## Data flow — query phase

```
User asks question via POST /api/chat
        │
        ▼
RAGPipeline.query(question)
        │
        ├─→ History-aware retriever
        │     ├─→ if follow-up: condense to standalone query
        │     └─→ retrieve top-k chunks from FAISS
        │
        ├─→ Stuff-documents chain
        │     └─→ inject chunks into prompt → Ollama → answer
        │
        ▼
Returns {answer, source_documents, question}
```

## State management

| State | Where it lives | Persists across restarts? |
|---|---|---|
| FAISS index | `./vector_store/` on disk | ✅ Yes |
| Indexed document metadata | `_indexed_docs` in `app.py` | ❌ No (rebuilt on startup from FAISS) |
| Conversation memory | `_histories` dict in `RAGPipeline` | ❌ No (in-memory only) |
| Loaded models | Ollama process | ❌ No (Ollama re-loads on demand) |

For multi-user isolation in production, move the singletons into per-session storage (e.g. keyed by a session cookie).

## Why these choices?

### Why FAISS instead of a hosted vector DB?
Local-first. No network calls, no API keys, no monthly bills. FAISS handles tens of thousands of vectors comfortably on a laptop.

### Why Ollama instead of OpenAI?
Privacy. Your documents and questions never leave your machine. Ollama runs llama3.1, mistral, qwen, and many others locally with no API key.

### Why a separate embedding model?
Chat models are trained for generation, not similarity. Using `nomic-embed-text` (a model trained specifically for embeddings) dramatically improves retrieval quality.

### Why LangChain?
Two reasons:
1. **`create_retrieval_chain`** + **`RunnableWithMessageHistory`** handle the awkward parts of conversational RAG (question rewriting, message history typing, etc.)
2. Easy to swap components — change the LLM, retriever, or memory class without touching the rest

### Why FastAPI over Flask?
- Auto-generated OpenAPI docs at `/docs`
- Native async support for streaming
- Pydantic models give you free request validation
- Modern, fast, and a pleasure to write
