# Troubleshooting

A catalogue of common issues and how to fix them.

---

## Setup issues

### `ModuleNotFoundError: No module named 'rag'`

You're running the script from the wrong directory, or the `rag/` package is missing its `__init__.py`.

**Fix:**
```bash
# Make sure your folder structure is:
# project/
#   ├── app.py
#   └── rag/
#       ├── __init__.py        ← MUST EXIST
#       ├── document_processor.py
#       ├── vector_store.py
#       └── pipeline.py

# Then run from the project folder, NOT from inside rag/:
cd project/
uvicorn app:app --reload
```

### `Cannot reach Ollama at http://localhost:11434`

Ollama isn't running.

**Fix:**
```bash
# In a separate terminal, start the server:
ollama serve

# Verify it's reachable:
curl http://localhost:11434/api/tags
```

### `Model 'llama3.1' not found`

The model isn't downloaded yet.

**Fix:**
```bash
ollama pull llama3.1
ollama pull nomic-embed-text     # also pull the embedding model
```

Verify with `ollama list`.

### `pip install langchain-ollama` fails

You may need a newer version of pip:
```bash
python -m pip install --upgrade pip
pip install langchain-ollama
```

---

## Runtime issues

### "Variable chat_history should be a list of base messages, got of type str"

This is a known bug in the deprecated `ConversationalRetrievalChain`. Make sure your `pipeline.py` uses the modern API (`create_retrieval_chain` + `RunnableWithMessageHistory`) — the version in this repo is already updated.

### Answers say "I couldn't find this" but the info IS in the document

This is a **retrieval failure** — the embeddings aren't matching well. Most common causes:

1. **You're using a chat model for embeddings.** Run:
   ```bash
   ollama pull nomic-embed-text
   ```
   Then set `EMBED_MODEL=nomic-embed-text` in your `.env` and **re-index your documents** (click "Clear everything" then re-upload).

2. **Chunks are too small** — relevant info is split across chunk boundaries. Increase `CHUNK_SIZE` to 1500 and `CHUNK_OVERLAP` to 300, then re-index.

3. **Not enough chunks retrieved.** Set `K_DOCUMENTS=8` or `10`.

4. **MMR is too aggressive.** Set `SEARCH_TYPE=similarity` instead of `mmr`.

### First response is very slow (30+ seconds)

The first query loads the model into Ollama's memory. Subsequent queries should be much faster (1–5 seconds depending on model size).

If it's *always* slow:
- Try a smaller model (`llama3.2:3b` instead of `llama3.1`)
- Make sure Ollama isn't being CPU-throttled or running on integrated graphics

### Out of memory crash

You're using a model bigger than your RAM. Switch to a smaller one:
```bash
# In .env:
OLLAMA_MODEL=llama3.2:3b
# Or even smaller:
OLLAMA_MODEL=llama3.2:1b
```

---

## UI issues

### "Try it out" button does nothing in `/docs`

You need to **scroll down inside the green panel** after clicking it — the file picker and Execute button appear below.

This only applies if you're using the FastAPI Swagger UI. The built-in chat UI at `/` works without any of this clicking.

### Drag and drop doesn't work

Make sure:
- The file is a `.pdf` (other formats are rejected)
- You're not dragging into the wrong area — the orange dashed box

If it still fails, click the drop zone instead of dragging.

### The page shows "Offline"

The status indicator polls `/api/status` every 15 seconds. If it shows offline:
- Check the terminal for server errors
- Verify the server is running: `curl http://localhost:8000/api/status`

---

## Index issues

### The index doesn't load on restart

Check that `./vector_store/` exists and contains files:
```bash
ls -la ./vector_store/
```

You should see something like `index.faiss` and `index.pkl`. If it's empty, the index was never saved. Try uploading and indexing again.

### "Permission denied" when saving the index

The process can't write to `./vector_store/`. Either:
- Run from a folder you have write access to, or
- Set `PERSIST_DIR` to a writable path: `PERSIST_DIR=/tmp/vector_store`

### The vectors look corrupted / similarity search returns garbage

You probably changed the embedding model after indexing. Embedding models produce **incompatible vectors** — you cannot mix them.

**Fix:** click "Clear everything" in the UI, then re-upload all documents.

---

## Performance issues

### Indexing is slow

- Each chunk requires one Ollama API call to embed → larger PDFs take time
- Consider running `python ingest.py --dir ./your_docs` from the CLI for batch processing
- A 100-page PDF with default chunking takes ~30–60 seconds on a typical laptop

### Querying feels sluggish

Generation is usually the bottleneck. To speed it up:
- Use a smaller model (`llama3.2:3b`)
- Reduce `K_DOCUMENTS` to 4 (smaller prompt)
- Reduce `CHUNK_SIZE` so each retrieved chunk is shorter

### High RAM usage

`llama3.1` needs ~8 GB. Plus FAISS holds vectors in RAM. For low-RAM machines:
- Use `llama3.2:1b` (~2 GB)
- Don't index more than ~5000 chunks

---

## Still stuck?

1. Check the terminal logs — uvicorn prints helpful errors with stack traces
2. Try the simplest possible test: `curl http://localhost:8000/api/status`
3. Open an [issue](../../issues) with:
   - Your OS, Python, and Ollama versions
   - The exact command and full error message
   - What you've already tried from this guide
