"""
app.py — PDF RAG Chat (FastAPI + built-in UI)
──────────────────────────────────────────────
Run:   uvicorn app:app --reload --port 8000
Open:  http://localhost:8000

KEY CHANGE: separate models for embedding and chat.
  - Chat   : llama3.1            (good at conversation)
  - Embed  : nomic-embed-text    (built specifically for embeddings)

Pull both models once:
    ollama pull llama3.1
    ollama pull nomic-embed-text
"""

import logging, os, tempfile, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PERSIST_DIR    = os.getenv("PERSIST_DIR",    "./vector_store")
OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "llama3.1")           # chat model
EMBED_MODEL    = os.getenv("EMBED_MODEL",    "nomic-embed-text")   # embedding model
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE",    "1500"))   # bigger = more context per chunk
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "300"))    # bigger = less context loss at boundaries
K_DOCUMENTS    = int(os.getenv("K_DOCUMENTS",   "6"))      # more retrieved = more chance to hit the answer
SEARCH_TYPE    = os.getenv("SEARCH_TYPE", "similarity")    # 'similarity' beats 'mmr' for fact lookup
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.2"))
MEMORY_WINDOW  = int(os.getenv("MEMORY_WINDOW", "6"))

app = FastAPI(title="PDF RAG Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_vsm:          Optional[Any] = None
_pipeline:     Optional[Any] = None
_indexed_docs: List[Dict]    = []


# ── helpers ───────────────────────────────────────────────────────────────────
def _build_embeddings():
    """Use a dedicated embedding model — NOT the chat model."""
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

def _get_vsm():
    global _vsm
    if _vsm is None:
        from rag.vector_store import VectorStoreManager
        _vsm = VectorStoreManager(embeddings=_build_embeddings(), persist_dir=PERSIST_DIR)
    return _vsm

def _get_pipeline():
    global _pipeline
    vsm = _get_vsm()
    if not vsm.is_ready:
        raise HTTPException(400, "No documents indexed yet. Upload PDFs first.")
    if _pipeline is None:
        from rag.pipeline import RAGPipeline
        _pipeline = RAGPipeline(
            vector_store_manager=vsm,
            model_name=OLLAMA_MODEL,
            ollama_url=OLLAMA_URL,
            temperature=TEMPERATURE,
            k_documents=K_DOCUMENTS,
            memory_window=MEMORY_WINDOW,
            search_type=SEARCH_TYPE,
        )
    return _pipeline

def _fmt_sources(docs):
    seen, out = set(), []
    for d in docs:
        k = (d.metadata.get("source",""), str(d.metadata.get("page","")))
        if k in seen: continue
        seen.add(k)
        out.append({"source": d.metadata.get("source","unknown"),
                    "page":   d.metadata.get("page","?"),
                    "content": d.page_content[:400]})
    return out


# ── schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    answer: str; sources: List[Dict]; question: str

class UploadResponse(BaseModel):
    message: str; files_indexed: List[str]; chunks_added: int; total_vectors: int


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def api_status():
    vsm = _get_vsm()
    return {"chat_model": OLLAMA_MODEL, "embed_model": EMBED_MODEL,
            "documents": len(_indexed_docs), "vectors": vsm.document_count, "ready": vsm.is_ready}

@app.post("/api/upload", response_model=UploadResponse, status_code=201)
async def api_upload(files: List[UploadFile] = File(...)):
    global _pipeline
    from rag.document_processor import DocumentProcessor
    proc = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vsm  = _get_vsm()
    chunks, names = [], []
    existing = {d["name"] for d in _indexed_docs}

    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                raise HTTPException(400, f"'{f.filename}' is not a PDF.")
            fname   = Path(f.filename).name
            content = await f.read()
            if fname in existing: continue
            p = Path(tmp) / fname
            p.write_bytes(content)
            try:
                c = proc.process_pdf(str(p))
                for ch in c: ch.metadata["source"] = fname
                if not c: continue
                chunks.extend(c); names.append(fname)
                _indexed_docs.append({"name": fname, "chunks": len(c), "id": str(uuid.uuid4())})
            except Exception as e:
                raise HTTPException(500, f"Failed to process '{fname}': {e}")

    if not names:
        return UploadResponse(message="Already indexed.", files_indexed=[], chunks_added=0, total_vectors=vsm.document_count)

    try: vsm.add_documents(chunks)
    except Exception as e: raise HTTPException(500, f"Embedding failed: {e}")

    _pipeline = None
    return UploadResponse(message=f"Indexed {len(names)} file(s).",
                          files_indexed=names, chunks_added=len(chunks), total_vectors=vsm.document_count)

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    try:
        r = _get_pipeline().query(req.question)
        return ChatResponse(answer=r.get("answer",""), sources=_fmt_sources(r.get("source_documents",[])),
                            question=r.get("question", req.question))
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.delete("/api/session")
def api_clear_session():
    if _pipeline: _pipeline.reset_memory()
    return {"message": "Session cleared."}

@app.get("/api/documents")
def api_documents():
    vsm = _get_vsm()
    return {"documents": _indexed_docs, "total_vectors": vsm.document_count}

@app.delete("/api/documents")
def api_clear_documents():
    global _vsm, _pipeline, _indexed_docs
    if _vsm: _vsm.clear()
    _vsm = None; _pipeline = None; _indexed_docs = []
    return {"message": "All documents cleared."}


# ── UI (same as before, with one tweak: shows both models) ────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PDF RAG Chat</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0b0d11;--s1:#111318;--s2:#161a22;--s3:#1c2130;--b1:rgba(255,255,255,0.06);--b2:rgba(255,255,255,0.1);--b3:rgba(255,255,255,0.16);--orange:#FF6B2B;--orange-dim:rgba(255,107,43,0.12);--orange-glow:rgba(255,107,43,0.04);--text:#dde1eb;--muted:#6b7585;--subtle:#2a3040;--green:#3ecf8e;--green-dim:rgba(62,207,142,0.1);--red:#f87171;--red-dim:rgba(248,113,113,0.1);--blue:#60a5fa;--mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif;--display:'Syne',sans-serif;--serif:'Instrument Serif',serif;--r:10px}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--subtle);border-radius:4px}
.shell{display:flex;height:100vh;overflow:hidden}
.sidebar{width:260px;flex-shrink:0;background:var(--s1);border-right:1px solid var(--b1);display:flex;flex-direction:column}
.brand{display:flex;align-items:center;gap:10px;padding:22px 20px 18px;border-bottom:1px solid var(--b1)}
.brand-mark{width:34px;height:34px;border-radius:9px;background:var(--orange);display:flex;align-items:center;justify-content:center;font-family:var(--display);font-weight:700;font-size:14px;color:#fff;flex-shrink:0;box-shadow:0 4px 12px rgba(255,107,43,0.35)}
.brand-name{font-family:var(--display);font-weight:600;font-size:15px;color:var(--text);line-height:1.1}
.brand-sub{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:1px}
.status-row{display:flex;align-items:center;gap:8px;padding:12px 20px;border-bottom:1px solid var(--b1)}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:background .3s}
.status-dot.ready{background:var(--green);box-shadow:0 0 6px rgba(62,207,142,0.5)}
.status-dot.busy{background:var(--orange);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.status-text{font-family:var(--mono);font-size:11px;color:var(--muted)}
.status-text span{color:var(--text);font-weight:500}
.upload-section{padding:16px 16px 10px}
.section-label{font-family:var(--mono);font-size:10px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;padding:0 4px;margin-bottom:8px}
.drop-zone{border:1.5px dashed var(--b2);border-radius:var(--r);padding:20px 16px;text-align:center;cursor:pointer;transition:all .2s;background:var(--orange-glow);position:relative}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--orange);background:var(--orange-dim)}
.drop-icon{font-size:22px;margin-bottom:6px;display:block}
.drop-title{font-size:13px;font-weight:500;color:var(--text);margin-bottom:3px}
.drop-sub{font-size:11px;color:var(--muted)}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.upload-queue{display:flex;flex-direction:column;gap:5px;padding:0 4px;margin-top:4px;max-height:180px;overflow-y:auto}
.queue-item{display:flex;align-items:center;gap:8px;background:var(--s2);border:1px solid var(--b1);border-radius:8px;padding:7px 10px}
.queue-icon{width:26px;height:26px;border-radius:6px;background:var(--orange-dim);border:1px solid rgba(255,107,43,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.queue-icon svg{width:12px;height:12px;stroke:var(--orange);fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.queue-name{font-size:12px;color:var(--text);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.queue-size{font-family:var(--mono);font-size:10px;color:var(--muted)}
.queue-remove{background:none;border:none;cursor:pointer;color:var(--muted);padding:2px;display:flex;align-items:center;border-radius:4px;transition:all .15s}
.queue-remove:hover{color:var(--red);background:var(--red-dim)}
.queue-remove svg{width:12px;height:12px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round}
.upload-btn{width:100%;margin-top:10px;padding:9px;border-radius:8px;background:var(--orange);border:none;cursor:pointer;font-family:var(--display);font-weight:600;font-size:13px;color:#fff;letter-spacing:.02em;transition:all .15s;display:flex;align-items:center;justify-content:center;gap:7px}
.upload-btn:hover{background:#ff7d42;transform:translateY(-1px);box-shadow:0 4px 16px rgba(255,107,43,0.35)}
.upload-btn:active{transform:translateY(0)}
.upload-btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.upload-btn svg{width:14px;height:14px;stroke:#fff;fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}
.docs-section{flex:1;overflow-y:auto;padding:4px 16px 16px}
.doc-item{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:7px;transition:background .15s;cursor:default}
.doc-item:hover{background:var(--s2)}
.doc-dot{width:6px;height:6px;border-radius:50%;background:var(--green);flex-shrink:0}
.doc-name{font-size:12px;color:var(--text);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc-chunks{font-family:var(--mono);font-size:10px;color:var(--muted)}
.sidebar-footer{padding:12px 16px;border-top:1px solid var(--b1);display:flex;flex-direction:column;gap:6px}
.footer-stat{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px}
.footer-stat .lbl{color:var(--muted)}.footer-stat .val{color:var(--text)}
.btn-ghost{width:100%;padding:7px;border-radius:7px;background:none;border:1px solid var(--b2);cursor:pointer;color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.03em;transition:all .15s}
.btn-ghost:hover{border-color:var(--red);color:var(--red);background:var(--red-dim)}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.topbar{display:flex;align-items:center;gap:12px;padding:14px 24px;background:var(--s1);border-bottom:1px solid var(--b1);flex-shrink:0}
.topbar-title{font-family:var(--display);font-weight:600;font-size:15px}
.model-badge{display:flex;align-items:center;gap:6px;padding:4px 12px;background:var(--s2);border:1px solid var(--b2);border-radius:20px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.model-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 5px rgba(62,207,142,0.5)}
.topbar-actions{margin-left:auto;display:flex;gap:8px}
.btn-sm{padding:5px 12px;border-radius:6px;border:1px solid var(--b2);background:none;color:var(--muted);font-family:var(--mono);font-size:11px;cursor:pointer;transition:all .15s}
.btn-sm:hover{border-color:var(--b3);color:var(--text)}
.welcome{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px;gap:20px;text-align:center}
.welcome-icon{width:60px;height:60px;border-radius:16px;background:var(--orange-dim);border:1px solid rgba(255,107,43,0.25);display:flex;align-items:center;justify-content:center;font-size:26px;box-shadow:0 8px 32px rgba(255,107,43,0.12)}
.welcome-title{font-family:var(--serif);font-size:26px;color:var(--text);font-style:italic}
.welcome-sub{font-size:14px;color:var(--muted);max-width:360px;line-height:1.65}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:4px}
.chip{padding:7px 16px;border-radius:20px;border:1px solid var(--b2);background:var(--s2);font-size:12px;color:var(--muted);cursor:pointer;transition:all .15s;font-family:var(--mono)}
.chip:hover{border-color:var(--orange);color:var(--orange);background:var(--orange-dim)}
.welcome-steps{display:flex;gap:0;margin-top:8px;background:var(--s2);border:1px solid var(--b1);border-radius:12px;overflow:hidden}
.w-step{padding:14px 20px;text-align:left;flex:1;position:relative}
.w-step+.w-step::before{content:'';position:absolute;left:0;top:20%;bottom:20%;width:1px;background:var(--b1)}
.w-num{font-family:var(--mono);font-size:10px;color:var(--orange);margin-bottom:4px}
.w-text{font-size:12px;color:var(--muted);line-height:1.4}
.messages{flex:1;overflow-y:auto;padding:24px 28px;display:flex;flex-direction:column;gap:20px}
.msg{display:flex;gap:12px;animation:slideUp .25s ease-out}
@keyframes slideUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg-avatar{width:30px;height:30px;border-radius:8px;flex-shrink:0;margin-top:2px;display:flex;align-items:center;justify-content:center;font-family:var(--display);font-size:11px;font-weight:700}
.msg-avatar.ai{background:var(--s3);border:1px solid var(--b2);color:var(--orange)}
.msg-avatar.user{background:var(--orange-dim);border:1px solid rgba(255,107,43,0.25);color:var(--orange)}
.msg-body{flex:1;min-width:0}
.msg-meta{display:flex;align-items:baseline;gap:8px;margin-bottom:5px}
.msg-name{font-family:var(--display);font-size:13px;font-weight:600;color:var(--text)}
.msg-time{font-family:var(--mono);font-size:10px;color:var(--muted)}
.msg-text{font-size:14px;line-height:1.75;color:#c4c9d6}
.msg-text p+p{margin-top:10px}
.msg-text strong{color:var(--text);font-weight:500}
.msg-text code{font-family:var(--mono);font-size:12px;background:var(--s2);padding:1px 6px;border-radius:4px;color:var(--orange)}
.user-bubble{background:var(--s2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 14px;font-size:14px;line-height:1.7;color:var(--text)}
.sources-toggle{display:inline-flex;align-items:center;gap:5px;margin-top:10px;padding:4px 10px;border-radius:20px;border:1px solid var(--b2);background:var(--s2);font-family:var(--mono);font-size:10px;color:var(--muted);cursor:pointer;transition:all .15s}
.sources-toggle:hover{border-color:var(--b3);color:var(--text)}
.sources-toggle .ct{background:var(--orange-dim);color:var(--orange);border-radius:10px;padding:0 5px;font-size:10px}
.sources-list{margin-top:8px;display:none;flex-direction:column;gap:6px}
.sources-list.open{display:flex}
.source-card{background:var(--s2);border:1px solid var(--b1);border-left:2px solid var(--orange);border-radius:var(--r);padding:9px 12px}
.source-meta{font-family:var(--mono);font-size:10px;color:var(--orange);margin-bottom:4px}
.source-text{font-size:12px;color:var(--muted);line-height:1.55}
.thinking{display:flex;gap:4px;align-items:center;padding:4px 0}
.thinking span{width:6px;height:6px;border-radius:50%;background:var(--muted);animation:think .8s ease-in-out infinite}
.thinking span:nth-child(2){animation-delay:.15s}.thinking span:nth-child(3){animation-delay:.3s}
@keyframes think{0%,80%,100%{transform:scale(.8);opacity:.4}40%{transform:scale(1.1);opacity:1}}
.input-area{padding:16px 24px 20px;border-top:1px solid var(--b1);background:var(--s1);flex-shrink:0}
.input-wrap{display:flex;align-items:flex-end;gap:10px;background:var(--s2);border:1px solid var(--b2);border-radius:12px;padding:10px 12px;transition:border-color .2s}
.input-wrap:focus-within{border-color:rgba(255,107,43,0.4);box-shadow:0 0 0 3px rgba(255,107,43,0.06)}
textarea{flex:1;background:none;border:none;outline:none;resize:none;color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5;max-height:140px;min-height:22px;height:22px;padding:0}
textarea::placeholder{color:var(--muted)}
.send-btn{width:34px;height:34px;border-radius:8px;flex-shrink:0;background:var(--orange);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .15s}
.send-btn:hover{background:#ff7d42;transform:scale(1.05);box-shadow:0 4px 12px rgba(255,107,43,0.35)}
.send-btn:active{transform:scale(.97)}.send-btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.send-btn svg{width:14px;height:14px;stroke:#fff;fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}
.input-hint{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:7px;padding:0 2px}
.input-hint kbd{background:var(--s2);border:1px solid var(--b2);border-radius:3px;padding:1px 5px;font-family:var(--mono);font-size:10px}
.toast-wrap{position:fixed;bottom:24px;right:24px;display:flex;flex-direction:column;gap:8px;z-index:999}
.toast{padding:10px 16px;border-radius:9px;font-size:13px;animation:toastIn .2s ease-out;max-width:320px;display:flex;align-items:center;gap:8px}
.toast.ok{background:var(--green-dim);border:1px solid rgba(62,207,142,0.25);color:var(--green)}
.toast.err{background:var(--red-dim);border:1px solid rgba(248,113,113,0.2);color:var(--red)}
.toast.info{background:var(--s2);border:1px solid var(--b2);color:var(--text)}
@keyframes toastIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.progress-bar{height:2px;background:var(--b1);border-radius:1px;margin-top:8px;overflow:hidden;display:none}
.progress-bar.active{display:block}
.progress-fill{height:100%;background:var(--orange);border-radius:1px;width:0;transition:width .3s ease}
</style></head><body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand"><div class="brand-mark">R</div><div><div class="brand-name">RAG Chat</div><div class="brand-sub">PDF Intelligence</div></div></div>
    <div class="status-row"><div class="status-dot" id="statusDot"></div><div class="status-text" id="statusText">Connecting…</div></div>
    <div class="upload-section">
      <div class="section-label">Upload Documents</div>
      <div class="drop-zone" id="dropZone">
        <input type="file" id="fileInput" accept=".pdf" multiple>
        <span class="drop-icon">📄</span>
        <div class="drop-title">Drop PDFs here</div>
        <div class="drop-sub">or click to browse</div>
      </div>
      <div class="upload-queue" id="uploadQueue"></div>
      <div class="progress-bar" id="progressBar"><div class="progress-fill" id="progressFill"></div></div>
      <button class="upload-btn" id="uploadBtn" onclick="uploadFiles()" disabled>
        <svg viewBox="0 0 24 24"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>
        Index Documents
      </button>
    </div>
    <div class="docs-section"><div class="section-label" style="padding:8px 4px 6px" id="docsLabel">Indexed (0)</div><div id="docList"></div></div>
    <div class="sidebar-footer">
      <div class="footer-stat"><span class="lbl">Vectors</span><span class="val" id="vectorCount">0</span></div>
      <div class="footer-stat"><span class="lbl">Chat</span><span class="val" id="chatModel">—</span></div>
      <div class="footer-stat"><span class="lbl">Embed</span><span class="val" id="embedModel">—</span></div>
      <button class="btn-ghost" onclick="clearAll()" style="margin-top:4px">Clear everything</button>
    </div>
  </aside>
  <main class="main">
    <header class="topbar">
      <div class="topbar-title">Chat</div>
      <div class="model-badge"><div class="dot"></div><span id="topModelName">llama3.1</span></div>
      <div class="topbar-actions"><button class="btn-sm" onclick="clearSession()">New session</button></div>
    </header>
    <div class="welcome" id="welcomeScreen">
      <div class="welcome-icon">📚</div>
      <div class="welcome-title">Ask your documents anything</div>
      <div class="welcome-sub">Upload PDFs on the left, then ask questions in natural language. Answers are grounded in your documents with cited sources.</div>
      <div class="welcome-steps">
        <div class="w-step"><div class="w-num">01</div><div class="w-text">Drop PDFs in the sidebar</div></div>
        <div class="w-step"><div class="w-num">02</div><div class="w-text">Click Index Documents</div></div>
        <div class="w-step"><div class="w-num">03</div><div class="w-text">Ask anything below</div></div>
      </div>
      <div class="chips">
        <div class="chip" onclick="useChip(this)">Summarise this document</div>
        <div class="chip" onclick="useChip(this)">What are the key findings?</div>
        <div class="chip" onclick="useChip(this)">List all mentioned dates</div>
        <div class="chip" onclick="useChip(this)">What is the main argument?</div>
      </div>
    </div>
    <div class="messages" id="messages" style="display:none"></div>
    <div class="input-area">
      <div class="input-wrap">
        <textarea id="msgInput" placeholder="Ask a question about your documents…" rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <button class="send-btn" id="sendBtn" onclick="sendMsg()">
          <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
      <div class="input-hint"><kbd>Enter</kbd> send &nbsp;·&nbsp; <kbd>Shift+Enter</kbd> new line</div>
    </div>
  </main>
</div>
<div class="toast-wrap" id="toastWrap"></div>
<script>
let pendingFiles = []; let ready = false;
function fmt(b){if(b<1024)return b+'B';if(b<1048576)return(b/1024).toFixed(0)+'KB';return(b/1048576).toFixed(1)+'MB'}
function now(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}
function toast(msg,type='info',dur=3000){const w=document.getElementById('toastWrap');const t=document.createElement('div');t.className=`toast ${type}`;t.textContent=msg;w.appendChild(t);setTimeout(()=>{t.style.opacity='0';t.style.transform='translateY(4px)';t.style.transition='all .2s';setTimeout(()=>t.remove(),200)},dur)}
function autoResize(el){el.style.height='22px';el.style.height=Math.min(el.scrollHeight,140)+'px'}
function handleKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg()}}
function useChip(el){const i=document.getElementById('msgInput');i.value=el.textContent;i.focus();autoResize(i)}
async function refreshStatus(){try{const r=await fetch('/api/status');const d=await r.json();
  document.getElementById('statusDot').className='status-dot '+(d.ready?'ready':'');
  document.getElementById('statusText').innerHTML=d.ready?`Ready &nbsp;<span>${d.documents} doc${d.documents!==1?'s':''}</span>`:`<span>${d.documents}</span> doc${d.documents!==1?'s':''} indexed`;
  document.getElementById('chatModel').textContent=d.chat_model;
  document.getElementById('embedModel').textContent=d.embed_model;
  document.getElementById('topModelName').textContent=d.chat_model;
  document.getElementById('vectorCount').textContent=d.vectors.toLocaleString();
  ready=d.ready;renderDocList();
}catch(e){document.getElementById('statusDot').className='status-dot';document.getElementById('statusText').textContent='Offline'}}
async function renderDocList(){try{const r=await fetch('/api/documents');const d=await r.json();
  document.getElementById('docsLabel').textContent=`Indexed (${d.documents.length})`;
  document.getElementById('docList').innerHTML=d.documents.map(doc=>`<div class="doc-item"><div class="doc-dot"></div><div class="doc-name" title="${doc.name}">${doc.name}</div><div class="doc-chunks">${doc.chunks}c</div></div>`).join('')
}catch(e){}}
const dropZone=document.getElementById('dropZone'),fileInput=document.getElementById('fileInput');
['dragenter','dragover'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();dropZone.classList.add('drag-over')}));
['dragleave','drop'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();dropZone.classList.remove('drag-over')}));
dropZone.addEventListener('drop',e=>addFiles(e.dataTransfer.files));
fileInput.addEventListener('change',e=>addFiles(e.target.files));
function addFiles(files){const pdfs=Array.from(files).filter(f=>f.name.toLowerCase().endsWith('.pdf'));if(!pdfs.length){toast('Please select PDF files only.','err');return}pdfs.forEach(f=>{if(!pendingFiles.find(p=>p.name===f.name))pendingFiles.push(f)});renderQueue()}
function removeFile(n){pendingFiles=pendingFiles.filter(f=>f.name!==n);renderQueue()}
function renderQueue(){const q=document.getElementById('uploadQueue'),btn=document.getElementById('uploadBtn');
  q.innerHTML=pendingFiles.map(f=>`<div class="queue-item"><div class="queue-icon"><svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div><div class="queue-name">${f.name}</div><div class="queue-size">${fmt(f.size)}</div><button class="queue-remove" onclick="removeFile('${f.name}')"><svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button></div>`).join('');
  btn.disabled=pendingFiles.length===0;
}
async function uploadFiles(){if(!pendingFiles.length)return;const btn=document.getElementById('uploadBtn'),pb=document.getElementById('progressBar'),fill=document.getElementById('progressFill'),dot=document.getElementById('statusDot');
  btn.disabled=true;btn.innerHTML='<span style="display:flex;gap:5px;align-items:center"><span style="display:flex;gap:3px"><span style="width:5px;height:5px;border-radius:50%;background:#fff;animation:think .8s ease-in-out infinite"></span><span style="width:5px;height:5px;border-radius:50%;background:#fff;animation:think .8s ease-in-out infinite;animation-delay:.15s"></span><span style="width:5px;height:5px;border-radius:50%;background:#fff;animation:think .8s ease-in-out infinite;animation-delay:.3s"></span></span> Indexing…</span>';
  dot.className='status-dot busy';pb.classList.add('active');fill.style.width='30%';
  const fd=new FormData();pendingFiles.forEach(f=>fd.append('files',f));
  try{fill.style.width='60%';const res=await fetch('/api/upload',{method:'POST',body:fd});fill.style.width='90%';const data=await res.json();fill.style.width='100%';
    if(!res.ok)throw new Error(data.detail||'Upload failed');
    toast(`✓ Indexed ${data.chunks_added} chunks from ${data.files_indexed.length} file(s)`,'ok');
    pendingFiles=[];renderQueue();await refreshStatus();
    if(data.files_indexed.length>0){document.getElementById('welcomeScreen').style.display='none';document.getElementById('messages').style.display='flex'}
  }catch(e){toast(e.message,'err',5000)}finally{
    setTimeout(()=>{pb.classList.remove('active');fill.style.width='0'},600);
    btn.disabled=false;btn.innerHTML='<svg viewBox="0 0 24 24" style="width:14px;height:14px;stroke:#fff;fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg> Index Documents';
    dot.className='status-dot'+(ready?' ready':'')
  }
}
function showMessages(){document.getElementById('welcomeScreen').style.display='none';document.getElementById('messages').style.display='flex'}
function addMsg(role,content,sources=[]){showMessages();const area=document.getElementById('messages'),div=document.createElement('div');div.className='msg';
  const srcHtml=sources.length?`<div class="sources-toggle" onclick="toggleSrc(this)"><span class="ct">${sources.length}</span> source${sources.length!==1?'s':''} cited<span style="margin-left:2px;font-size:9px">▾</span></div><div class="sources-list">${sources.map((s,i)=>`<div class="source-card"><div class="source-meta">[${i+1}] ${s.source} · page ${s.page}</div><div class="source-text">${s.content}</div></div>`).join('')}</div>`:'';
  if(role==='user'){div.innerHTML=`<div class="msg-avatar user">U</div><div class="msg-body"><div class="msg-meta"><span class="msg-name">You</span><span class="msg-time">${now()}</span></div><div class="user-bubble">${content}</div></div>`}
  else{div.innerHTML=`<div class="msg-avatar ai">AI</div><div class="msg-body"><div class="msg-meta"><span class="msg-name">Assistant</span><span class="msg-time">${now()}</span></div><div class="msg-text"><p>${content.replace(/\n\n/g,'</p><p>').replace(/\n/g,'<br>')}</p></div>${srcHtml}</div>`}
  area.appendChild(div);area.scrollTop=area.scrollHeight;return div
}
function addThinking(){showMessages();const area=document.getElementById('messages'),div=document.createElement('div');div.className='msg';div.id='thinking';
  div.innerHTML=`<div class="msg-avatar ai">AI</div><div class="msg-body" style="padding-top:6px"><div class="thinking"><span></span><span></span><span></span></div></div>`;
  area.appendChild(div);area.scrollTop=area.scrollHeight
}
function removeThinking(){document.getElementById('thinking')?.remove()}
function toggleSrc(el){const list=el.nextElementSibling,open=list.classList.toggle('open');el.querySelector('span:last-child').textContent=open?'▴':'▾'}
async function sendMsg(){const input=document.getElementById('msgInput'),q=input.value.trim();if(!q)return;
  if(!ready){toast('Upload and index documents first.','err');return}
  const btn=document.getElementById('sendBtn');btn.disabled=true;input.value='';input.style.height='22px';
  addMsg('user',q);addThinking();
  try{const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const data=await res.json();removeThinking();
    if(!res.ok)throw new Error(data.detail||'Request failed');
    addMsg('ai',data.answer,data.sources||[])
  }catch(e){removeThinking();addMsg('ai','⚠ '+e.message);toast(e.message,'err',5000)}
  finally{btn.disabled=false;input.focus()}
}
async function clearSession(){await fetch('/api/session',{method:'DELETE'});document.getElementById('messages').innerHTML='';if(!ready){document.getElementById('welcomeScreen').style.display='flex';document.getElementById('messages').style.display='none'}toast('New session started','info')}
async function clearAll(){if(!confirm('Delete all indexed documents and clear the chat?'))return;await fetch('/api/documents',{method:'DELETE'});await fetch('/api/session',{method:'DELETE'});document.getElementById('messages').innerHTML='';document.getElementById('welcomeScreen').style.display='flex';document.getElementById('messages').style.display='none';pendingFiles=[];renderQueue();await refreshStatus();toast('Everything cleared','info')}
refreshStatus();setInterval(refreshStatus,15000);
fetch('/api/status').then(r=>r.json()).then(d=>{if(d.ready){document.getElementById('welcomeScreen').style.display='none';document.getElementById('messages').style.display='flex'}});
</script>
</body></html>"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML

@app.on_event("startup")
def on_startup():
    logger.info("PDF RAG Chat  →  http://localhost:8000")
    logger.info("Chat model:  %s", OLLAMA_MODEL)
    logger.info("Embed model: %s", EMBED_MODEL)
    vsm = _get_vsm()
    if vsm.is_ready:
        logger.info("Loaded existing FAISS index (%d vectors)", vsm.document_count)
