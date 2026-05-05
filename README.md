# PDF RAG Chat

> A **fully local** Retrieval-Augmented Generation (RAG) chat application for your PDF documents. Upload PDFs, ask questions in natural language, get grounded answers with source citations — all running on your own machine via Ollama. No API keys, no data leaving your computer.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/Ollama-local-000000?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/FAISS-vector%20store-0467DF" alt="FAISS">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

---

## ✨ Features

- 🔒 **100% local & private** — your documents never leave your machine
- 💬 **Conversational interface** — multi-turn chat with rolling memory
- 📚 **Source citations** — every answer shows the PDF passages used
- 🎯 **Grounded answers** — model only uses your documents, never hallucinates
- ⚡ **FastAPI backend** — clean REST API with automatic OpenAPI docs
- 🎨 **Built-in elegant UI** — drag-and-drop uploads, dark theme, no separate frontend needed
- 🔄 **Persistent index** — FAISS vectors survive restarts; index once, query forever
- 🧠 **Smart retrieval** — uses dedicated embedding models (nomic-embed-text) for high-quality search

---

## 🎬 Demo

| Welcome screen | Chat with sources |
|---|---|
| ![welcome](docs/screenshot-welcome.png) | ![chat](docs/screenshot-chat.png) |

> Add screenshots to `docs/` after your first run to bring this section to life.

---

## 🏗️ Architecture

```
┌──────────────┐      ┌────────────────┐      ┌──────────────┐
│   Browser    │──────│   FastAPI      │──────│   Ollama     │
│  (built-in   │ HTTP │   (app.py)     │ HTTP │   (local)    │
│     UI)      │      │                │      │              │
└──────────────┘      └────────┬───────┘      └──────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
        │  Document    │ │  Vector   │ │     RAG     │
        │  Processor   │ │   Store   │ │   Pipeline  │
        │  (PyPDF)     │ │  (FAISS)  │ │ (LangChain) │
        └──────────────┘ └───────────┘ └─────────────┘
```

**Two-phase workflow:**
1. **Indexing** — PDFs → text chunks → embeddings → FAISS index (saved to disk)
2. **Querying** — Question → retrieve chunks → augment prompt → generate answer

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a deeper dive.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 or higher**
- **[Ollama](https://ollama.ai)** installed and running
- ~8 GB free RAM (for `llama3.1`)

### 1. Install Ollama and pull the models

```bash
# Install from https://ollama.ai
ollama serve                       # start the server (keep running)
ollama pull llama3.1               # the chat model (~4.7 GB)
ollama pull nomic-embed-text       # the embedding model (~270 MB)
```

### 2. Clone and set up the project

```bash
git clone https://github.com/<your-username>/pdf-rag-chat.git
cd pdf-rag-chat

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn app:app --reload --port 8000
```

### 4. Open in your browser

Navigate to **[http://localhost:8000](http://localhost:8000)** — that's it. Drag and drop PDFs into the sidebar, click **Index Documents**, then start chatting.

---

## 📋 Project Structure

```
pdf-rag-chat/
├── app.py                       # FastAPI server + built-in UI
├── ingest.py                    # CLI for batch PDF ingestion
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── .gitignore
├── LICENSE
├── README.md
│
├── rag/                         # Core RAG package
│   ├── __init__.py
│   ├── document_processor.py    # PDF loader & text splitter
│   ├── vector_store.py          # FAISS wrapper with persistence
│   └── pipeline.py              # ConversationalRetrievalChain
│
├── docs/                        # Extended documentation
│   ├── ARCHITECTURE.md
│   ├── CONFIGURATION.md
│   ├── API.md
│   └── TROUBLESHOOTING.md
│
└── vector_store/                # FAISS index (auto-created, gitignored)
```

---

## ⚙️ Configuration

All settings can be overridden via environment variables or a `.env` file. Copy `.env.example` to `.env` to get started:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1` | Chat model for generation |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for retrieval |
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `300` | Overlap between chunks |
| `K_DOCUMENTS` | `6` | Chunks retrieved per query |
| `SEARCH_TYPE` | `similarity` | `similarity` or `mmr` |
| `TEMPERATURE` | `0.2` | LLM sampling temperature |
| `MEMORY_WINDOW` | `6` | Past conversation turns to retain |
| `PERSIST_DIR` | `./vector_store` | Where to save the FAISS index |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for tuning advice.

---

## 🔌 API Reference

The FastAPI backend exposes a clean REST API. Interactive docs at **`/docs`** when running.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Built-in chat UI |
| `GET` | `/api/status` | Server, Ollama, and index status |
| `POST` | `/api/upload` | Upload and index PDF files |
| `POST` | `/api/chat` | Ask a question, get a grounded answer |
| `DELETE` | `/api/session` | Clear conversation memory |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents` | Wipe the entire vector store |

**Example — chat via cURL:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Full details in [docs/API.md](docs/API.md).

---

## 🛠 CLI: Batch Ingestion

To pre-index a folder of PDFs without using the UI:

```bash
python ingest.py --dir ./my_documents
python ingest.py --file report.pdf --file manual.pdf
python ingest.py --dir ./docs --chunk-size 1200 --chunk-overlap 200
python ingest.py --dir ./docs --test-query "what are the key policies?"
```

Run `python ingest.py --help` for all options.

---

## 🧠 How It Works

When you ask a question:

1. **Question rewriting** — if it's a follow-up, LangChain rewrites it into a standalone query using chat history
2. **Embedding** — your query is converted into a vector by `nomic-embed-text`
3. **Retrieval** — FAISS finds the top-k most similar chunks from your indexed PDFs
4. **Augmentation** — the chunks are injected into the system prompt as context
5. **Generation** — `llama3.1` generates an answer grounded in the retrieved context
6. **Response** — answer + cited source passages returned to the UI

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | Run `ollama serve` in a separate terminal |
| `Model 'llama3.1' not found` | Run `ollama pull llama3.1` |
| `ModuleNotFoundError: No module named 'rag'` | Make sure `rag/__init__.py` exists and you're running from the project root |
| Answers say "not in the documents" but they should be | Use the dedicated embedding model: `ollama pull nomic-embed-text` then click Clear everything and re-index |
| Slow first response | First query loads the model into memory; subsequent queries are fast |
| Out of memory | Use a smaller model like `llama3.2:1b` via `OLLAMA_MODEL=llama3.2:1b` |

More fixes in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 🗺️ Roadmap

- [ ] Streaming responses (token-by-token)
- [ ] Multi-user sessions with proper isolation
- [ ] Support for `.docx`, `.txt`, and `.md` files
- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Re-ranking with a cross-encoder
- [ ] Docker compose setup
- [ ] Authentication / API keys

---

## 🤝 Contributing

Contributions are warmly welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature
# make changes
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a pull request
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built on top of these excellent open-source projects:

- [LangChain](https://github.com/langchain-ai/langchain) — RAG orchestration
- [FastAPI](https://github.com/tiangolo/fastapi) — modern Python web framework
- [Ollama](https://github.com/ollama/ollama) — local LLM runtime
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI's similarity search
- [PyPDF](https://github.com/py-pdf/pypdf) — PDF parsing

---

<p align="center">
  <sub>Made with care · star ⭐ if it helped you</sub>
</p>
