# AI System Design Coach

> Built an evaluated RAG system for AI engineering education — grounded, cited, measured.

---

## Status

| | |
|---|---|
| **Current Phase** | Phase 2 — Golden Eval Dataset |
| **Phase Status** | Complete 2026-05-26 |
| **Pipeline** | End-to-end working: ingest → chunk → embed → retrieve → generate → cite |
| **UI** | Minimal React/TS chat with clickable citations + Evaluation panel |
| **Corpus** | 18 sources / 373 chunks (Option A corpus fix included) |
| **Eval Harness** | Deterministic 100-question benchmark + HTML/JSON score reports |

---

## Latest Eval Scores *(2026-05-26 — after eval-driven prompt fix)*

The table below shows the before/after from the Phase 2 benchmark run. The **eval-first workflow** — measure → identify failure mode → fix → re-measure — is the project's core thesis.

| Metric | Before fix | After fix | PRD Target |
|---|---|---|---|
| refusal_accuracy | 0.51 | **0.82** | ≥ 0.95 — not yet met (Phase 3) |
| answer_rate (grounded Qs) | 0.36 | **0.77** | — |
| source_recall@5 | 0.63 | **0.61** | ≥ 0.85 — not yet met (Phase 3) |
| pass_rate | 0.45 | **0.62** | ≥ 0.80 |
| citation_validity | 1.00 | **1.00** | — |

The first benchmark run exposed severe over-refusal on grounded questions. A single eval-driven revision of the generation prompt cut false refusals dramatically without breaking genuine refusals (adversarial questions still ~24/25 correct). The remaining gap to PRD targets is a retrieval/corpus problem — the Phase 3 lever (better chunking, reranker, corpus coverage).

---

## Eval-First Philosophy

Every answer is grounded in retrieved source material and refuses when context is insufficient — it never fills gaps with parametric knowledge. Every claim is cited with inline `[n]` markers backed by a specific chunk; the citation validator downgrades unsupported answers to the canonical refusal. Every `/query` request is logged as a structured JSONL record (query, retrieved chunks + scores, citations, latency, cost, refused flag). This log is the substrate the Phase 2–4 eval harness reads to compute Recall@k, faithfulness, correctness, and citation accuracy. The differentiator is not the chat UI; it is the measurement infrastructure around it.

---

## Evaluation Workflow

**Corpus:** 18 sources / 373 chunks ingested into Qdrant `ai_coach_docs`.

**Golden dataset:** `backend/app/evals/golden/dataset.yaml` — 100 questions across 7 types (definition, comparison, architecture, troubleshooting, multi_hop, refusal, adversarial).

**Metrics (deterministic, no LLM judge):** refusal_accuracy, answer_rate, source_recall@5, citation_validity, pass_rate. RAGAS/faithfulness deferred to Phase 4.

```powershell
# (Re)build the corpus — Qdrant must be running
cd backend
python scripts/ingest.py

# Run the full benchmark — produces an HTML+JSON report
python scripts/run_evals.py
# → backend/reports/eval_runs/eval-<run_id>.{html,json}

# Fast unit + harness suite (hermetic, no keys/network)
pytest
# → 79 passed, 2 deselected

# Gated regression guard — baseline floors enforced
pytest -m eval
```

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
# Backend — fast unit + eval harness suite
cd backend
pytest
# → 79 passed, 2 deselected (deselected: @pytest.mark.slow + @pytest.mark.eval)

# Backend — gated regression guard (requires Qdrant + OPENAI_API_KEY)
pytest -m eval

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
  → RequestLogger (append JSONL to backend/reports/eval_runs/requests-YYYYMMDD.jsonl)
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
      App.tsx                  # root component
      components/              # Chat, AnswerView, EvaluationPanel
      lib/                     # citations.ts tokenizer + citations.test.ts
      api/                     # client.ts (typed POST /query wrapper)
      types.ts                 # shared TypeScript types
  docs/
    raw/                       # fetched source docs (gitignored)
    superpowers/specs/         # per-phase design specs
  backend/reports/
    eval_runs/                 # JSONL request logs + eval HTML reports (gitignored)
  HANDOVERS/                   # one file per completed task (CLAUDE Rule 2)
  docker-compose.yml           # Qdrant
  PRD.md                       # live product requirements (CLAUDE Rule 4)
  CLAUDE.md                    # working agreement
```

---

## Documentation

- **Latest handover:** [`HANDOVERS/2026-05-26_1345_phase2-eval-and-corpus.md`](HANDOVERS/2026-05-26_1345_phase2-eval-and-corpus.md)
- **PRD:** [`PRD.md`](PRD.md)
- **Phase 1 Design Spec:** [`docs/superpowers/specs/2026-05-25-phase1-basic-rag-design.md`](docs/superpowers/specs/2026-05-25-phase1-basic-rag-design.md)
- **Phase 2 Plan:** [`docs/superpowers/plans/2026-05-26-phase2-eval-harness.md`](docs/superpowers/plans/2026-05-26-phase2-eval-harness.md)
