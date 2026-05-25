# AI System Design Coach

> Built an evaluated RAG system for AI engineering education — grounded, cited, measured.

---

## Status

| | |
|---|---|
| **Current Phase** | Phase 1 — Basic RAG |
| **Phase Status** | Implementation complete 2026-05-26 |
| **Pipeline** | End-to-end working: ingest → chunk → embed → retrieve → generate → cite |
| **UI** | Minimal React/TS chat with clickable citations + Evaluation panel |
| **Formal Eval Harness** | Phase 2 (golden dataset + Recall@k / MRR / faithfulness benchmark) |

---

## Preliminary Numbers *(2026-05-26 — not the gating benchmark)*

These are sanity-check figures from live ingestion and local-embedding retrieval. The formal scored eval (Recall@5, MRR, faithfulness) lands in Phase 2.

| Metric | Value | Notes |
|---|---|---|
| Corpus | 14 sources / 285 chunks | Ingested into Qdrant `ai_coach_docs` |
| Embed model | `BAAI/bge-small-en-v1.5` (384-d, local) | No API key needed for retrieval |
| Top-5 cosine — RAG definition query | 0.82 / 0.81 / 0.80 / 0.76 / 0.74 | All hits from LangChain RAG tutorial |
| Top-5 cosine — faithfulness metric query | 0.75 / 0.72 / 0.70 / … | Top 3 hits from RAGAS source |
| Formal Recall@5 / MRR / Faithfulness | **TBD — Phase 2** | Gating targets: ≥85% / ≥0.70 / ≥0.90 |

**Preliminary (local bge-small); formal Recall@k / MRR / faithfulness benchmark is Phase 2.**

---

## Eval-First Philosophy

Every answer is grounded in retrieved source material and refuses when context is insufficient — it never fills gaps with parametric knowledge. Every claim is cited with inline `[n]` markers backed by a specific chunk; the citation validator downgrades unsupported answers to the canonical refusal. Every `/query` request is logged as a structured JSONL record (query, retrieved chunks + scores, citations, latency, cost, refused flag). This log is the substrate the Phase 2–4 eval harness reads to compute Recall@k, faithfulness, correctness, and citation accuracy. The differentiator is not the chat UI; it is the measurement infrastructure around it.

---

## Quickstart

### Prerequisites

- Python 3.12
- Node 22
- Docker (for Qdrant)

### 1 — Start Qdrant

```powershell
docker compose up -d
```

### 2 — Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env          # then open .env and set OPENAI_API_KEY
python scripts/ingest.py        # fetches docs, chunks, embeds, upserts to Qdrant
uvicorn app.main:app --reload
```

### 3 — Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 4 — Phase 1 Acceptance Smoke

Run all 20 sample questions through the full pipeline (requires Qdrant running with ingested corpus and `OPENAI_API_KEY` set in `backend/.env`):

```powershell
cd backend
python scripts/run_samples.py
```

Expected: grounded cited answers on AI-engineering questions, with ≥2 correct refusals for off-topic questions.

---

## Tests

Tests are hermetic — no API keys, no network, no running Qdrant required.

```powershell
# Backend
cd backend
pytest
# → 43 passed, 1 deselected (deselected: @pytest.mark.slow real-fastembed download)

# Frontend
cd frontend
npm run test
# → 2 passed
```

---

## Required Environment Variables

Source: `backend/.env.example`. Copy to `backend/.env` before running.

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | — | Yes (generator) | OpenAI key for generation; not needed for tests |
| `LLM_PROVIDER` | `openai` | No | `openai` \| `fake` |
| `LLM_MODEL` | `gpt-4o-mini` | No | Generator model; swap to `gpt-4o` for quality |
| `EMBED_PROVIDER` | `fastembed` | No | `fastembed` (local) \| `openai` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | No | Embedding model |
| `QDRANT_URL` | `http://localhost:6333` | No | Qdrant connection |
| `QDRANT_COLLECTION` | `ai_coach_docs` | No | Collection name |
| `TOP_K` | `5` | No | Retrieval depth |
| `CHUNK_SIZE` | `1000` | No | Chunk size in chars |
| `CHUNK_OVERLAP` | `150` | No | Chunk overlap in chars |
| `SCORE_THRESHOLD` | `0.30` | No | Refusal gate (untuned; Phase 3) |
| `REWRITE_ENABLED` | `true` | No | Query rewriting toggle |
| `LANGCHAIN_TRACING_V2` | `false` | No | LangSmith tracing (optional) |
| `LANGCHAIN_API_KEY` | — | No | LangSmith key (optional) |
| `LANGCHAIN_PROJECT` | `ai-system-design-coach` | No | LangSmith project name |

---

## Architecture

### Ingest Path (offline, run once)

```
docs/sources.yaml
  → Fetcher (httpx + trafilatura HTML→Markdown, retry + skip on error)
  → Chunker (LangChain RecursiveCharacterTextSplitter, 1000/150 chars)
  → EmbeddingProvider (fastembed bge-small local, or OpenAI via env)
  → Qdrant upsert (collection: ai_coach_docs, stable doc_id point IDs)
```

### Query Path (per request)

```
POST /query
  → QueryRewriter (LLM-based abbreviation/ambiguity expansion; toggleable)
  → RefusalGate (short-circuit if no chunk clears SCORE_THRESHOLD)
  → Retriever (embed query → Qdrant top-k vector search → RetrievedChunk[])
  → Generator (LLM prompt with numbered chunks [1..k] → answer + [n] citations)
  → CitationValidator (validate [n] ranges; downgrade violations to refusal)
  → RequestLogger (append JSONL to reports/eval_runs/requests-YYYYMMDD.jsonl)
  → AnswerResult (answer, citations, chunks, scores, latency, cost, refused)
```

### Stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript (Vite, strict mode) |
| Backend | FastAPI + Python 3.12 (async, typed) |
| Vector DB | Qdrant (Docker) |
| Embeddings | `fastembed` `BAAI/bge-small-en-v1.5` (local, 384-d); OpenAI swap in Phase 3 |
| Orchestration | Thin typed orchestrator; LangChain for text splitting only |
| Tracing | LangSmith (`@traceable`, env-gated, off by default) |
| Eval | RAGAS + custom pytest (Phase 2+) |
| LLM | OpenAI `gpt-4o-mini` default; configurable via env |

---

## Repo Layout

```
ai-system-design-coach/
  backend/
    app/
      main.py                  # FastAPI app, POST /query, GET /health
      rag/                     # ingest, chunker, retriever, generator, citation_checker, orchestrator
      evals/                   # eval runners and metrics (Phase 2+)
    scripts/
      ingest.py                # offline ingestion entrypoint
      run_samples.py           # Phase 1 acceptance smoke (20 questions)
    tests/                     # pytest unit + integration (hermetic, no keys)
    .env.example               # document all required vars
  frontend/
    src/
      pages/                   # chat page
      components/              # citation renderer, evaluation panel
  docs/
    raw/                       # fetched source docs (gitignored)
    superpowers/specs/         # per-phase design specs
  reports/
    eval_runs/                 # JSONL request logs + eval HTML reports
  HANDOVERS/                   # one file per completed task (CLAUDE Rule 2)
  docker-compose.yml           # Qdrant
  PRD.md                       # live product requirements (CLAUDE Rule 4)
  CLAUDE.md                    # working agreement
```

---

## Documentation

- **Latest handover:** [`HANDOVERS/2026-05-26_0046_phase1-basic-rag.md`](HANDOVERS/2026-05-26_0046_phase1-basic-rag.md)
- **PRD:** [`PRD.md`](PRD.md)
- **Phase 1 Design Spec:** [`docs/superpowers/specs/2026-05-25-phase1-basic-rag-design.md`](docs/superpowers/specs/2026-05-25-phase1-basic-rag-design.md)
