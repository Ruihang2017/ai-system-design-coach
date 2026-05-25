# Phase 1 Design — Basic RAG with Citations

**Date:** 2026-05-25
**Project:** AI System Design Coach (see `PRD.md`, `CLAUDE.md`)
**Phase:** 1 — Basic RAG (PRD §9)
**Status:** Approved (design); ready for implementation planning

---

## 1. Goal & Definition of Done

Build the end-to-end RAG pipeline that turns AI-engineering questions into **grounded, cited answers**, with the architectural boundaries the eval-first product (Phases 2–5) depends on.

**Phase 1 is done when:**

1. `python backend/scripts/run_samples.py` runs **20 sample questions** through the real pipeline and returns grounded answers with inline citations, including ≥2 questions that correctly **refuse** (off-topic / unsupported).
2. The unit + integration test suite (`pytest backend/tests`) is green **without** API keys or network (uses fakes + in-memory Qdrant).
3. The React UI renders an answer with **clickable citations** and an expandable **Evaluation panel** (retrieved chunks + scores, latency breakdown, cost, refused flag).
4. Every `/query` writes one structured JSONL request-log record (the substrate Phases 2–4 read).

---

## 2. Locked Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Phase sequencing | Phase 1 only, then checkpoint | Smallest reviewable increment (PRD §9) |
| Embeddings | **Local** `fastembed` `BAAI/bge-small-en-v1.5` (384-d) default | Zero cost; OpenAI `text-embedding-3-large` is the Phase-3 A/B swap |
| Generator LLM | **OpenAI**, default `gpt-4o-mini` | User has OpenAI (no Anthropic key); under <$0.02/query target; `gpt-4o` documented as quality swap; Claude added when key exists |
| Corpus | **Fetch real docs from the web** (curated URL allowlist) | User choice; authentic content; cached for reproducibility |
| Frontend | **Minimal React/TS UI in Phase 1** | Demonstrable end-to-end; "Compare runs" deferred to Phase 5 |
| Orchestration | **Approach A — thin typed orchestrator** | LangChain for components only; best isolation/testability + per-step instrumentation for the eval log |
| Tracing | **LangSmith wired now**, env-gated | User has the account; cheap to add early; dashboards stay in Phase 5 |

---

## 3. Architecture

```
                    ┌─────────────── RAG Orchestrator ───────────────┐
POST /query ──▶ QueryRewriter ─▶ Retriever ─▶ Generator ─▶ CitationValidator ─▶ AnswerResult
                    │              │ (Qdrant)   │ (LLM)       │ (+ refusal gate)    │
                    └──────────────┴────────────┴─────────────┴──── RequestLogger ──┘
                                                                     (JSONL eval log)

Ingestion (offline entrypoint, not in request path):
  docs/sources.yaml ─▶ Fetcher ─▶ HTML→Markdown ─▶ Chunker ─▶ EmbeddingProvider ─▶ Qdrant upsert
```

Each module is a focused file with a typed interface. Providers (embeddings, LLM) sit behind `Protocol`s so the orchestrator depends only on interfaces. This is what lets Phases 3–4 A/B swap (embedding model, chunk size, reranker on/off, judge model) purely by config.

### Component responsibilities

- **QueryRewriter** (`rag/query_rewriter.py`) — light LLM-based rewrite (expand abbreviations, clarify). Toggleable via config (`REWRITE_ENABLED`, default on). No-op fallback path so it can be disabled for A/B in Phase 3.
- **Retriever** (`rag/retriever.py`) — embeds query, runs Qdrant top-k vector search, returns `RetrievedChunk[]` with scores. `top_k` configurable.
- **Generator** (`rag/generator.py`) — builds the prompt from numbered chunks `[1..k]`, calls `LLMProvider`, returns raw answer text + `TokenUsage`. Parses inline `[n]` markers.
- **CitationValidator** (`rag/citation_checker.py`) — validates every `[n]` is in range and any non-refusal answer carries ≥1 citation; downgrades violations to the canonical refusal. (Claim-level faithfulness is Phase 4 — not stubbed here.)
- **RefusalGate** — part of the orchestrator: if no retrieved chunk clears `SCORE_THRESHOLD`, short-circuit to the exact PRD refusal string before any LLM generation call.
- **Orchestrator** (`rag/orchestrator.py`) — composes the above, captures per-step latency, assembles `AnswerResult`, and hands a `RequestLog` to the logger.
- **RequestLogger** (`eval/request_logger.py`) — appends one JSON line per request to `reports/eval_runs/requests-YYYYMMDD.jsonl`.

### Ingestion components

- **Fetcher** (`rag/fetcher.py`) — fetch each allowlisted URL with polite UA + retry/timeout; extract main content with `trafilatura` (HTML→markdown); cache raw + processed under `docs/raw/`.
- **Chunker** (`rag/chunker.py`) — LangChain `RecursiveCharacterTextSplitter`, config-driven `chunk_size`/`overlap`.
- **ingest.py** — orchestrates fetch→chunk→embed→Qdrant upsert; idempotent (stable `doc_id`/point IDs); payload `{text, source_url, title, doc_id, chunk_index}`.

---

## 4. Data models (`rag/models.py`, pydantic)

- `Chunk` — `{id, text, source_url, title, doc_id, chunk_index}`
- `RetrievedChunk` — `Chunk` + `score: float` + ordinal `n: int` (citation number)
- `TokenUsage` — `{prompt_tokens, completion_tokens, total_tokens}`
- `AnswerResult` — `{request_id, answer, citations: list[int], chunks: list[RetrievedChunk], refused: bool, latency_ms: dict[str,float], usage: TokenUsage, cost_usd: float, model: str}`
- `RequestLog` — `AnswerResult` fields + `{ts, query, rewritten_query, config_snapshot}` (serialized to JSONL)

---

## 5. Answer contract (PRD §7.2)

The assistant MUST: answer only from retrieved context; cite each non-trivial claim with `[n]`; refuse with the exact string when context is insufficient:

> *"I don't have enough information in the provided sources to answer that confidently."*

The assistant MUST NOT: use parametric knowledge to fill gaps; cite unsupporting chunks; obey adversarial "ignore the documents" instructions.

**Enforcement (Phase 1):** two-layer refusal (retrieval score gate + prompt instruction) and the citation validator. Prompts live in versioned files under `rag/prompts/` — changing a prompt is a code change requiring a re-run of the sample smoke (CLAUDE conventions).

---

## 6. Request log schema (foundation for Phases 2–4)

One JSON object per line:

```json
{
  "request_id": "uuid", "ts": "ISO-8601",
  "query": "...", "rewritten_query": "...",
  "retrieved": [{"chunk_id": "...", "score": 0.71, "source_url": "...", "n": 1}],
  "answer": "...", "citations": [1, 3], "refused": false,
  "latency_ms": {"rewrite": 120, "retrieve": 35, "generate": 1400, "total": 1560},
  "usage": {"prompt_tokens": 1800, "completion_tokens": 220, "total_tokens": 2020},
  "cost_usd": 0.0009, "model": "gpt-4o-mini",
  "config_snapshot": {"top_k": 5, "chunk_size": 1000, "embed_model": "bge-small", "rerank": false}
}
```

Cost = token usage × a per-model pricing table in `config.py`.

---

## 7. Backend API (FastAPI)

- `POST /query` — body `{query: str}` → `AnswerResult`. Retry+timeout on retrieval/LLM calls; structured logs; CORS for dev frontend.
- `GET /health` — liveness + Qdrant reachability.

App wiring uses FastAPI dependency injection so tests can override the orchestrator with a fake-backed one.

---

## 8. Frontend (Vite + React + TS, strict)

Single-page chat. Answers render `[n]` markers as **clickable** elements that scroll/highlight the matching chunk. Each answer has an expandable **Evaluation panel**: retrieved chunks (text/source/score), latency breakdown, cost, refused flag. Talks to `POST /query`. Built with the `frontend-design` skill to avoid generic AI aesthetics. Component-level tests where logic exists (citation parsing/rendering); manual verification steps documented in the handover otherwise (CLAUDE Rule 1).

---

## 9. Testing strategy (CLAUDE Rule 1 — TDD)

**Unit/integration — no keys, no network:**
- `FakeLLMProvider` (canned output) and `qdrant-client` in-memory mode (`:memory:`).
- Coverage: fetcher extraction (mocked HTTP via `pytest-httpx`), chunker determinism, embedding dim, retriever top-k ordering, generator prompt-build + `[n]` parsing, citation validator case table (valid / out-of-range / no-citation / refusal), refusal gate (low score → refuses), orchestrator end-to-end (fakes), API via `TestClient` with overridden deps, request-logger JSONL shape.

**Acceptance smoke — needs OpenAI key:**
- `backend/scripts/run_samples.py` over the 20 sample questions = Phase 1 DoD. If no key in an environment, reported under "Untested" per CLAUDE Rule 1; the unit suite still proves correctness via fakes.

---

## 10. Configuration & secrets

`config.py` (pydantic-settings) reads `.env`. Documented in `.env.example` and README:

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | generator (required for real runs) |
| `LLM_PROVIDER` | `openai` | `openai` \| `fake` |
| `LLM_MODEL` | `gpt-4o-mini` | generator model |
| `EMBED_PROVIDER` | `fastembed` | `fastembed` \| `openai` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model |
| `QDRANT_URL` | `http://localhost:6333` | vector DB |
| `QDRANT_COLLECTION` | `ai_coach_docs` | collection name |
| `TOP_K` | `5` | retrieval depth |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | chunking (chars) |
| `SCORE_THRESHOLD` | `0.30` | refusal gate (tune in Phase 3) |
| `REWRITE_ENABLED` | `true` | query rewriting toggle |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | off | LangSmith (optional) |

Secrets never committed; `.env` gitignored.

---

## 11. Dependencies

**Backend:** `fastapi, uvicorn[standard], qdrant-client, fastembed, langchain, langchain-community, langchain-openai, langchain-text-splitters, openai, trafilatura, tiktoken, pydantic-settings, pyyaml, langsmith, httpx, tenacity`.
**Dev:** `pytest, pytest-asyncio, pytest-httpx, ruff, black`.
**Frontend:** `vite, react, react-dom, typescript`.
**Infra:** Qdrant via `docker-compose`.

---

## 12. Autonomous assumptions (made without HITL — flagged for morning review)

1. **Source URLs:** I'll curate ~15–20 specific public documentation pages (LangChain, LangGraph, Qdrant, RAGAS, FastAPI, Docker, etc.) into `docs/sources.yaml`. List is reviewable/editable post-hoc.
2. **`gpt-4o-mini` default** for cost; quality swap to `gpt-4o` documented. If model naming has changed, it's a one-line `.env` edit.
3. **`SCORE_THRESHOLD = 0.30`** is a placeholder pending empirical tuning in Phase 3; Phase 1 leans on prompt-level refusal + genuinely off-topic sample questions.
4. **Chunking defaults** `1000/150` chars; Phase 3 A/Bs them.
5. **20 sample questions** authored to span PRD question types (definition, comparison, architecture, troubleshooting, refusal, adversarial); full 100-question golden set is Phase 2.
6. **No live network/keys in the test suite** — fetcher and LLM are faked/mocked; the real pipeline is exercised only by the manual smoke script.

---

## 13. Out of scope (deferred per PRD)

Golden dataset & `pytest evals/` (P2); retrieval metrics + A/B chunking + reranker (P3); RAGAS faithfulness/correctness (P4); LangSmith dashboards, regression alerts, "Compare runs" UI (P5); auth/billing/multi-tenant, mobile, fine-tuning (non-goals).
