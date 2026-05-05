# Configuration Guide

All settings can be controlled via environment variables (or a `.env` file). This page explains what each setting does and when to tune it.

## Models

### `OLLAMA_MODEL` — the chat model
Default: `llama3.1`

This is the model that **generates the answers**. Trade-offs by model:

| Model | Size | Quality | Speed |
|---|---|---|---|
| `llama3.2:1b` | ~1.3 GB | Lower | Very fast |
| `llama3.2:3b` | ~2 GB | Good | Fast |
| `llama3.1` | ~4.7 GB | Great | Medium |
| `mistral` | ~4 GB | Great | Medium |
| `qwen2.5:7b` | ~4.7 GB | Excellent | Medium |
| `llama3.1:70b` | ~40 GB | Best | Slow (needs 64+ GB RAM) |

Use a smaller model on low-RAM machines. Use a larger one when answer quality matters more than speed.

### `EMBED_MODEL` — the embedding model
Default: `nomic-embed-text`

This converts your text into vectors for retrieval. **Don't use a chat model here** — embedding-specific models are much better at similarity search.

| Model | Size | Notes |
|---|---|---|
| `nomic-embed-text` | ~270 MB | Fast, accurate, recommended default |
| `mxbai-embed-large` | ~670 MB | Higher quality, slower |
| `snowflake-arctic-embed` | ~330 MB | Excellent for multilingual |
| `all-minilm` | ~46 MB | Tiny, lower quality |

⚠️ **If you change this**, you must re-index all your documents because the new model produces incompatible vectors. Click "Clear everything" in the UI then re-upload.

## Chunking

### `CHUNK_SIZE` — characters per chunk
Default: `1500`

Larger chunks contain more context, which helps when answers span multiple paragraphs. Smaller chunks are more precisely retrieved but lose surrounding context.

| Use case | Recommended |
|---|---|
| Q&A on dense reports | 1500–2000 |
| Code documentation | 800–1200 |
| Reference manuals (lots of headings) | 1000–1500 |
| Conversational text (chat logs, transcripts) | 500–800 |

### `CHUNK_OVERLAP` — overlap between consecutive chunks
Default: `300`

Without overlap, a fact mentioned at the boundary of two chunks gets split. Overlap prevents this. Aim for **15–25%** of `CHUNK_SIZE`.

## Retrieval

### `K_DOCUMENTS` — chunks retrieved per query
Default: `6`

More chunks = higher chance of finding the answer, but the prompt gets longer (slower) and may include irrelevant noise (worse answers).

| Use case | Recommended |
|---|---|
| Specific factual questions | 4–6 |
| Comparative or analytical questions | 6–10 |
| Summarisation across many sections | 8–12 |

### `SEARCH_TYPE` — `similarity` vs `mmr`
Default: `similarity`

- **`similarity`** — pure cosine similarity. Best when you want the *most relevant* chunks. Good for fact lookup and direct questions.
- **`mmr`** — Maximal Marginal Relevance. Penalises chunks that are too similar to ones already retrieved, increasing diversity. Good for "give me an overview" or "compare these things" questions.

If you switch to `mmr` and answers become worse for direct questions, switch back.

## Generation

### `TEMPERATURE` — LLM creativity
Default: `0.2`

| Value | Behaviour |
|---|---|
| `0.0` | Deterministic — same answer every time |
| `0.2` | Mostly factual, slight variation (recommended for RAG) |
| `0.5` | Balanced |
| `0.8+` | Creative, possibly hallucinatory |

For RAG, lower is better — you want the model sticking closely to the retrieved context.

### `MEMORY_WINDOW` — past turns retained
Default: `6`

Each "turn" is one human + one AI message, so `6` keeps the last 12 messages. Higher values:
- ✅ Better at follow-up questions ("and what about X?")
- ❌ Longer prompts → slower responses, higher chance of confusion

## Storage

### `PERSIST_DIR` — where the FAISS index lives
Default: `./vector_store`

Change this if you want to:
- Keep multiple separate indexes (one per project)
- Store the index on a different drive
- Backup the index by copying the directory

## Tuning workflow

Don't change everything at once. Tune in this order:

1. **First, ensure the embedding model is `nomic-embed-text`** (or another dedicated embedding model). This makes the biggest single difference.
2. **Adjust `CHUNK_SIZE`** if you're getting "I couldn't find this" responses for things you know are in the docs. Try larger chunks first.
3. **Bump `K_DOCUMENTS`** to 8 or 10 if the right info is in your docs but not being retrieved.
4. **Switch to a bigger chat model** if retrieval is finding the right chunks but the answers feel weak.
5. **Adjust `TEMPERATURE`** last. Most issues are retrieval problems, not generation problems.
