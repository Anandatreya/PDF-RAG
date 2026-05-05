# API Reference

Base URL: `http://localhost:8000` (or wherever you're running the server).

The interactive Swagger UI is also available at `/docs` when the server is running.

---

## `GET /` — Web UI

Returns the built-in HTML chat interface. Just open this in your browser.

---

## `GET /api/status`

Check the health of the server, the Ollama connection, and the vector store.

**Response 200:**
```json
{
  "chat_model": "llama3.1",
  "embed_model": "nomic-embed-text",
  "documents": 3,
  "vectors": 247,
  "ready": true
}
```

| Field | Meaning |
|---|---|
| `chat_model` | The model used for generating answers |
| `embed_model` | The model used for retrieval |
| `documents` | Number of unique PDFs indexed |
| `vectors` | Total chunks stored in FAISS |
| `ready` | `true` if you can ask questions; `false` if you need to upload PDFs first |

---

## `POST /api/upload`

Upload one or more PDF files for indexing.

**Request:** `multipart/form-data` with field `files` containing one or more PDF files.

**Example with cURL:**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "files=@report.pdf" \
  -F "files=@policy.pdf"
```

**Response 201:**
```json
{
  "message": "Indexed 2 file(s).",
  "files_indexed": ["report.pdf", "policy.pdf"],
  "chunks_added": 87,
  "total_vectors": 247
}
```

**Errors:**
- `400` — file is not a PDF
- `500` — failed to process or embed the file

**Notes:**
- Files already indexed (by filename) are skipped
- Each file is split into chunks and embedded — this can take a few seconds per file

---

## `POST /api/chat`

Ask a question about the indexed documents.

**Request body:**
```json
{
  "question": "What is the refund policy?"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | yes | 1–2000 characters |

**Response 200:**
```json
{
  "answer": "According to the policy, refunds are available within 30 days...",
  "sources": [
    {
      "source": "policy.pdf",
      "page": 12,
      "content": "Customers may request a refund within 30 days of purchase..."
    },
    {
      "source": "policy.pdf",
      "page": 13,
      "content": "Refunds are processed within 5-7 business days..."
    }
  ],
  "question": "What is the refund policy?"
}
```

**Errors:**
- `400` — no documents indexed yet (upload first)
- `500` — LLM or retrieval error

**Conversation memory:** consecutive calls share the same conversation history (last 6 turns by default). Follow-ups like "and what about international orders?" will reference earlier context automatically.

---

## `DELETE /api/session`

Clear the conversation memory. Documents remain indexed; only the chat history is wiped.

**Response 200:**
```json
{ "message": "Session cleared." }
```

---

## `GET /api/documents`

List all indexed documents.

**Response 200:**
```json
{
  "documents": [
    { "name": "report.pdf", "chunks": 42, "id": "abc-123-..." },
    { "name": "policy.pdf", "chunks": 45, "id": "def-456-..." }
  ],
  "total_vectors": 87
}
```

---

## `DELETE /api/documents`

Wipe the entire vector store. All indexed documents are removed and the FAISS index file is deleted from disk. You'll need to re-upload PDFs after this.

**Response 200:**
```json
{ "message": "All documents cleared." }
```

---

## Status codes summary

| Code | When |
|---|---|
| `200` | Success |
| `201` | Successfully created (upload) |
| `400` | Bad request — invalid file, no question, no docs indexed |
| `422` | Validation error — malformed JSON or missing required field |
| `500` | Server error — Ollama unreachable, embedding failed, etc. |
