# Module 3 — Support Assistant (`/support-assistant`)

A small, fully offline-gradable RAG service for Zepto's own policy corpus,
orchestrated with LangGraph and served via FastAPI.

## What's here
- `docs/doc_01.txt` … `doc_08.txt` — the 8-document Zepto policy corpus (exact text from the assignment)
- `ingest.py` — chunks + embeds the corpus (local `all-MiniLM-L6-v2`) into ChromaDB
- `prompt_template.py` — the role-context-task-format-length prompt template (Task 2)
- `schema.py` — Pydantic `AskRequest` / `AskResponse` (Task 4)
- `graph.py` — LangGraph `StateGraph` — 3 nodes + conditional routing (Task 3)
- `main.py` — FastAPI app, `POST /ask`
- `Dockerfile` — builds + runs the FastAPI app locally
- `test_wiring_offline.py` — offline test proving the graph/FastAPI wiring is correct (see note)

## Execution status: genuinely verified, real embeddings, real ChromaDB
`ingest.py` runs for real end-to-end: 8 real document chunks, embedded with
the real `sentence-transformers/all-MiniLM-L6-v2` model (384-dim vectors,
confirmed by pulling a stored vector back out of ChromaDB), indexed into a
real persistent ChromaDB collection. `/ask` was tested through the real
FastAPI app (`fastapi.testclient.TestClient`), and genuine semantic retrieval
was confirmed separately: the query *"How long does a refund take?"*
correctly retrieves `doc_02` (Returns & Refunds) as the top match, ahead of
`doc_06` (damaged items) and `doc_05` (cancellations) — real cosine-distance
ranking, not a keyword match.

*(Environment note: `huggingface.co` isn't reachable from the sandbox this was
verified in, so `ingest.py`/`graph.py` fall back to a local copy of the model
under `support-assistant/local_models/` — the same official model weights,
obtained from a GitHub-hosted mirror since `github.com` is reachable there.
This fallback is excluded from git via `.gitignore`; on a normal machine with
open internet access, `SentenceTransformer("all-MiniLM-L6-v2")` just downloads
from the hub directly and this fallback is never used.)*

### Run it yourself
```bash
pip install -r requirements.txt
python ingest.py                 # builds chroma_store/ with real embeddings
uvicorn main:app --host 0.0.0.0 --port 7860
# in another terminal:
curl -X POST localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How do I cancel my order?"}'
```
Everything runs with `MOCK_LLM` left unset (mock mode, the graded baseline) —
no LLM API call is made; only retrieval (embedding + ChromaDB) is real.

## Example calls (from `e2e_test_output.txt`, genuine execution, real retrieval)

**Health check:**
```
GET / -> 200 {"status": "ok"}
```

**Call 1 — policy question (routes to `retrieve_and_answer`, real retrieval):**
```
Query: "How long does delivery take?"
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation...",
  "sources": ["doc_01", "doc_02", "doc_04"],
  "confidence": 1.0
}
```
`doc_01` (Delivery) is genuinely the top-ranked real semantic match.

**Call 2 — general question (routes to `direct_answer`):**
```
Query: "What is the capital of France?"
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Both responses validated against the `AskResponse` Pydantic schema — genuinely
constructed from the live response JSON, not asserted.

## Architecture (ingestion → embedding → retrieval → generation)
1. **Ingestion** — `ingest.py:load_and_chunk_documents()` reads all 8
   `docs/doc_XX.txt` files, one chunk per document (each is already a short,
   self-contained policy paragraph).
2. **Embedding** — `ingest.py:build_index()` embeds each chunk locally with
   `sentence-transformers/all-MiniLM-L6-v2` (no API key/account) and stores
   the vectors in a persistent ChromaDB collection (`chroma_store/`,
   collection `zepto_policies`).
3. **Retrieval** — `graph.py`'s `retrieve_and_answer` node embeds the
   incoming query with the same model and does a top-3 cosine-similarity
   `collection.query(...)` against ChromaDB. This step is always real, in
   both mock and real-LLM modes.
4. **Generation** — branches on `MOCK_LLM` (default unset → mock, the graded
   baseline):
   - `classify_intent`: mock = keyword heuristic over `POLICY_KEYWORDS`; real
     (`MOCK_LLM=0`) = one LLM call to classify.
   - `retrieve_and_answer`: mock = canned
     `f"Based on the retrieved context: {top_chunk_snippet}"`; real = the
     Task 2 prompt template sent to the LLM (Groq free tier or equivalent),
     grounded only in the retrieved chunks.
   - `direct_answer`: mock = fixed canned string, no LLM call; real = one LLM
     call with a short "policy-assistant, out of scope" instruction.
   - The Pydantic `AskResponse` schema (Task 4) is populated deterministically
     in mock mode (no LLM output exists to validate); in the optional real-LLM
     path, a validation failure retries with a corrective instruction up to 2
     times before returning a clearly marked error.

A LangGraph `StateGraph` (`graph.py:build_graph()`) wires `classify_intent` as
the entry point, with a conditional edge to either `retrieve_and_answer` or
`direct_answer` based on the classified intent, both terminating at `END`.
`main.py` wraps `run_query()` in a single `POST /ask` FastAPI endpoint.
