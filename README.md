# PDF Ingestion & Semantic Search API (FastAPI + Qdrant + Embedder + Streamlit)

A small microservice-based system to **ingest PDFs**, **chunk + embed their text**, store vectors in **Qdrant**, and run **semantic search** over the indexed chunks.

## Architecture (4 services)

- **Frontend (Streamlit)**: UI for ingestion + search
- **Backend (FastAPI)**: ingestion/search API, extraction, chunking, Qdrant operations
- **Embedder (FastAPI)**: embedding microservice (FastEmbed)
- **Vector DB (Qdrant)**: stores embeddings + payload metadata

Ports (docker-compose):
- Frontend: `8501`
- Backend: `8000`
- Embedder: `8001`
- Qdrant: `6333`

## Quickstart (Docker Compose)

From repo root:

~~~bash
./orchestrate.sh --action start
# or: docker compose up --build -d
~~~

Open:
- UI: http://localhost:8501
- Backend: http://localhost:8000
- Qdrant: http://localhost:6333

Stop + remove volumes:
~~~bash
./orchestrate.sh --action terminate
# or: docker compose down -v
~~~

### Upload directory (for path ingestion)
Compose mounts `./data/uploads` into the backend container at `/data/uploads`.
Drop PDFs there if you want to ingest by directory path.

## Core workflows

### 1) Ingestion (PDF → chunks → embeddings → Qdrant)

The backend supports **two ingestion modes** through the same endpoint:

**A) Upload ingest (multipart)**  
Send one or more `input=@file.pdf` fields.

**B) Directory-path ingest (server-side path)**  
Send `input=/data/uploads` to ingest all `*.pdf` files in that directory (non-recursive).

Pipeline (what happens internally):
1. **Validate input**: only filenames ending with `.pdf` are accepted.
2. **Extract text**:
   - First attempt: **PyPDF** page extraction.
   - If PyPDF fails or yields empty output: **fallback** to decoding the raw PDF bytes as UTF-8 (`errors="ignore"`).
3. **Replace-by-filename (dedupe policy)**:
   - Before inserting new vectors, if a document with the **same filename** already exists, the backend **deletes all existing chunks** for that filename from Qdrant to avoid duplicates.
4. **Chunking with tokenizer-aware sizing**:
   - Chunking targets a **token length**, not just characters.
   - When available, a HuggingFace tokenizer is used to count tokens so chunks match the intended token budget (plus overlap).
   - If the tokenizer can’t be loaded, the system falls back to a word-based approximation.
5. **Batch embedding to protect the embedder**:
   - Chunks are embedded by calling the embedder service over HTTP.
   - If there are many chunks, the backend embeds them in **batches** (e.g., 64 per request) to avoid overloading/crashing the embedder service.
6. **Upsert into Qdrant**:
   - Each chunk is stored as a point with:
     - `vector`: the embedding
     - `payload`: `{ document: <filename>, chunk_id: <index>, content: <chunk_text> }`
   - Point IDs are deterministic (derived from `<filename>:<chunk_index>`), so the same filename+chunk index maps to the same point ID.

Key backend modules:
- Text extraction: `services/backend/app/core/text_extract.py`
- Chunking: `services/backend/app/core/chunking.py`
- Tokenizer loader: `services/backend/app/core/tokenizer_provider.py`
- Embedder HTTP client: `services/backend/app/core/embedder_client.py`
- Qdrant wrapper: `services/backend/app/core/vector_store.py`
- Orchestration: `services/backend/app/services/ingestion_service.py`

### 2) Search (query → embedding → Qdrant → results)

1. Client sends a query to the backend.
2. Backend embeds the query via the embedder service.
3. Backend queries Qdrant with `top_k` and `with_payload=True`.
4. Response returns matches containing `document`, `score`, and the matching chunk `content`.

Key backend modules:
- Search orchestration: `services/backend/app/services/search_service.py`
- Qdrant search: `services/backend/app/core/vector_store.py`

## Backend API

### Health
`GET /health`

Response:
~~~json
{ "status": "ok" }
~~~

### Ingest
`POST /ingest/` (form)

**Upload example (multipart)**
~~~bash
curl -X POST "http://localhost:8000/ingest/" \
  -H "x-request-id: demo-1" \
  -F "input=@./data/uploads/a.pdf" \
  -F "input=@./data/uploads/b.pdf"
~~~

**Directory-path example (form-urlencoded)**
~~~bash
curl -X POST "http://localhost:8000/ingest/" \
  -H "x-request-id: demo-2" \
  -d "input=/data/uploads"
~~~

Success:
~~~json
{
  "message": "Successfully ingested N PDF documents.",
  "files": ["a.pdf", "b.pdf"]
}
~~~

Errors (typical shape):
~~~json
{ "error": "..." }
~~~

### Search
`POST /search/` (JSON)

Request:
~~~json
{ "query": "your question here" }
~~~

Example:
~~~bash
curl -X POST "http://localhost:8000/search/" \
  -H "Content-Type: application/json" \
  -H "x-request-id: demo-3" \
  -d '{"query":"your question here"}'
~~~

Success:
~~~json
{
  "results": [
    { "document": "a.pdf", "score": 0.83, "content": "..." }
  ]
}
~~~

Errors:
~~~json
{ "error": "..." }
~~~

## Embedder API (internal service)

The backend calls the embedder service over HTTP.

- `GET /healthz`
- `POST /embed_documents` (JSON: `{ "texts": ["...","..."] }` → `{ "vectors": [[...],[...]] }`)
- `POST /embed_query` (JSON: `{ "text": "..." }` → `{ "vector": [...] }`)

Source: `services/embedder/app/main.py`

## Frontend (Streamlit)

The UI provides:
- **Ingest**:
  - Upload PDFs (multipart `input` fields)
  - Ingest a directory path (form field `input=/data/uploads`)
- **Search**:
  - Query input → renders top matches + scores

Source: `services/frontend/website/streamlit_app.py`

## Configuration (env vars)

Compose sets defaults; you can override via environment variables.

Backend (`services/backend/app/settings.py`):
- `QDRANT_URL` (default `http://qdrant:6333`)
- `QDRANT_COLLECTION` (default `pdf_chunks`)
- `EMBEDDER_URL` (default `http://embedder:8001`)
- `EMBEDDING_MODEL` (used for tokenizer and embedder)
- `TOP_K` (default `5`)
- Chunking:
  - `CHUNK_TARGET_TOKENS`
  - `CHUNK_OVERLAP_TOKENS`
  - `CHUNK_MAX_TOKENS`
- `UPLOAD_DIR` (default `/data/uploads`)
- `LOG_LEVEL`

Frontend:
- `BACKEND_URL` (default `http://app:8000` in compose)

Embedder:
- `EMBEDDING_MODEL`
- `LOG_LEVEL`

## Observability

- Backend and embedder emit JSON logs to stdout.
- Optional `x-request-id` can be supplied by clients; the backend uses it for correlation in logs.
- Backend logging helper: `services/backend/app/logging_utils.py`

## Notes / current behavior worth knowing

- **Deduplication is filename-based**: re-ingesting `same_name.pdf` deletes old chunks for that filename before inserting new ones.
- Text extraction uses **PyPDF first**, then **byte decode fallback** if needed.
- Embeddings are computed in **batches** to protect the embedder from large payloads.
- Directory-path ingestion reads PDFs from inside the backend container (use with care if exposing publicly).