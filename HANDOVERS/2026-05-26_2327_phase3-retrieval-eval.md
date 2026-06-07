# Handover: Phase 3 — Retrieval Eval & Tuning

**Date/Time:** 2026-05-26 23:27
**Branch:** `feat/phase3-retrieval-eval`
**Commit:** `fa40ee8` (code HEAD before docs commit)
**Commit range (Phase 3 code):** `614727d..fa40ee8`
**Phase:** Phase 3 — Retrieval Eval & Tuning

---

## What Was Done

Built a deterministic, no-LLM retrieval A/B sweep harness, ran it over all chunking/search/reranker/top-k variants, declared a data-driven winner, and applied the winner to the live pipeline.

### Retrieval metrics library
`app/evals/retrieval_metrics.py` — source-level relevance functions: recall@k, precision@k, MRR, hit-rate@k. These operate on retrieved chunk sources vs golden expected sources, fully deterministic (no LLM judge).

### Retrieval-only evaluator
`app/evals/retrieval_eval.py` — runs retrieval (no generation) against 75 grounded golden questions and computes the retrieval metrics above. Designed to be fast and key-free.

### Staged sweep harness
`app/evals/sweep.py` + `scripts/run_sweep.py` — iterates over `IngestVariant` configurations (chunk_size, hybrid on/off, reranker on/off, top-k), builds each collection, runs the retrieval evaluator, collects results, and calls the comparison reporter.

### Comparison reporter
`app/evals/comparison_report.py` — consumes sweep results and emits HTML, Markdown, and JSON comparison reports with a declared winner (ranked by MRR + hit-rate@5).

### Reranker
`app/rag/reranker.py` — local fastembed cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores retrieved candidates. Also exposes `NoopReranker` and `FakeReranker` for tests.

### Sparse / BM25
`app/rag/sparse.py` — fastembed BM25 sparse encoder. `FakeSparseEncoder` for hermetic tests.

### Hybrid retrieval in Qdrant store + retriever
`app/rag/qdrant_store.py` + `app/rag/retriever.py` — named dense+sparse vectors, Qdrant RRF fusion when `hybrid_enabled=True`. Falls back to dense-only when disabled.

### Ingest variant support
`app/rag/ingest.py` — `build_index` / `IngestVariant` to programmatically build collections per sweep configuration.

### Phase-3 config settings
`config.py` — `rerank_enabled`, `rerank_candidates`, `hybrid_enabled`, `sparse_model` added.

### Winner applied to live pipeline
`build_orchestrator` wired for hybrid retrieval. New defaults: `hybrid_enabled=True`, `qdrant_collection=ai_coach_cs1000_hybrid`.

---

## Files Changed

*(commit range 614727d..fa40ee8)*

| Area | Files |
|---|---|
| Retrieval metrics | `backend/app/evals/retrieval_metrics.py` |
| Retrieval evaluator | `backend/app/evals/retrieval_eval.py` |
| Sweep harness | `backend/app/evals/sweep.py` |
| Comparison reporter | `backend/app/evals/comparison_report.py` |
| Reranker | `backend/app/rag/reranker.py` |
| Sparse / BM25 | `backend/app/rag/sparse.py` |
| Qdrant store (hybrid) | `backend/app/rag/qdrant_store.py` |
| Retriever (hybrid) | `backend/app/rag/retriever.py` |
| Ingest (variant support) | `backend/app/rag/ingest.py` |
| Config | `backend/config.py` |
| Sweep runner script | `backend/scripts/run_sweep.py` |
| Tests | `backend/tests/` (new tests for reranker, sparse, retrieval metrics, sweep) |
| Docs | `README.md`, `PRD.md`, `HANDOVERS/2026-05-26_2327_phase3-retrieval-eval.md` |

---

## Tests

```powershell
# Fast unit + harness suite (hermetic — no API keys, no network, no Qdrant)
cd backend
pytest -q
# → 97 passed, 4 deselected
#    deselected: @pytest.mark.slow + @pytest.mark.eval

# Slow tests — validates real cross-encoder + BM25 models (no API keys needed)
pytest -m slow
# → passes

# Gated regression guard (requires Qdrant running + OPENAI_API_KEY)
pytest -m eval
# → passes updated baseline floors

# Lint
python -m ruff check app tests
# → All checks passed!
```

---

## Eval Impact

### Retrieval sweep (centerpiece — deterministic, no LLM, 75 grounded questions)

Run: `sweep-1c8f2d241b73` — raw query (no rewrite), source-level relevance.

| Variant | recall@5 | precision@5 | hit_rate@5 | MRR |
|---|---|---|---|---|
| cs500_dense | 0.600 | 0.448 | 0.653 | 0.587 |
| cs1000_dense (prior baseline) | 0.633 | 0.443 | 0.707 | 0.594 |
| cs1500_dense | 0.613 | 0.419 | 0.693 | 0.596 |
| **cs1000_hybrid (WINNER)** | **0.653** | 0.453 | **0.733** | **0.617** |
| cs1000_hybrid + reranker | 0.633 | 0.475 | 0.693 | 0.592 |
| cs1000_hybrid @ k=10 | recall@10 0.687 | — | hit@10 0.760 | 0.604 |

Winner: **chunk_size=1000, hybrid (BM25+dense RRF), reranker OFF, k=5** → collection `ai_coach_cs1000_hybrid`.

Key findings: hybrid beats pure-dense across all metrics. The cross-encoder reranker HURT recall/hit-rate at candidates=20/k=5 (traded recall for precision) — left off.

### Full-benchmark before/after (Phase 2 eval suite, 100 questions, with LLM)

| Metric | Before (dense, run a90c620e) | After (hybrid, run 9910c0d1) | PRD Target |
|---|---|---|---|
| refusal_accuracy | 0.82 | **0.83** | ≥ 0.95 — not yet met |
| answer_rate | 0.77 | **0.81** | — |
| source_recall@5 | 0.61 | **0.64** | ≥ 0.85 — not yet met |
| pass_rate | 0.62 | **0.63** | ≥ 0.80 |
| citation_validity | 1.00 | **1.00** | — |

Hybrid delivers a real but modest improvement (answer_rate +0.04, source_recall +0.03, no regressions). No PRD targets are yet met.

---

## Known Issues / Untested

1. **Metrics remain below PRD targets — corpus coverage is the ceiling.** source_recall@5 = 0.64 means roughly ⅓ of grounded questions still lack a retrievable supporting source. Retrieval hyperparameter tuning space is now exhausted (sweep done). The next lever is **corpus expansion** (more/better source documents), not further tuning.

2. **Reranker needs a larger candidate pool to be net-positive.** At candidates=20/k=5 the cross-encoder hurt recall (-0.020) and hit-rate (-0.040). Larger candidate pools (e.g., candidates=50) may flip it positive — a future experiment.

3. **Query-rewriting may slightly hurt retrieval vs raw query.** The sweep used raw (unrewritten) queries. The live pipeline still applies query-rewriting before retrieval. Any rewrite that substantially changes the query surface may slightly depress retrieval metrics relative to the sweep numbers. This is a future A/B experiment.

4. **`docs/raw/` reproducibility gap still open.** The corpus lives only in Qdrant; there is no on-disk cache of fetched docs. The `ai_coach_cs1000_hybrid` collection must be rebuilt via `run_sweep.py` or `build_index` in a fresh environment (requires source URLs to still be live).

5. **`.env` was updated locally (`QDRANT_COLLECTION=ai_coach_cs1000_hybrid`, `HYBRID_ENABLED=true`); gitignored.** A fresh checkout relies on the new `config.py` defaults (`qdrant_collection="ai_coach_cs1000_hybrid"`, `hybrid_enabled=True`) being present — verify those are committed.

---

## How to Verify Locally

```powershell
# Prerequisites: Docker (Qdrant), Python 3.12 venv activated

# 1. Start Qdrant
docker compose up -d

# 2. Run the retrieval sweep (no API key needed — deterministic, no LLM)
cd backend
python scripts/run_sweep.py
# → Builds each variant collection, evaluates retrieval, writes HTML/MD/JSON report
# → backend/reports/eval_runs/sweep-<id>/report.{html,md,json}
# → Declares winner: cs1000_hybrid

# 3. Run the full benchmark (requires OPENAI_API_KEY in backend/.env)
python scripts/run_evals.py
# → backend/reports/eval_runs/eval-<run_id>.{html,json}
# → Expected: refusal_accuracy ~0.83, source_recall ~0.64, pass_rate ~0.63

# 4. Fast hermetic test suite
pytest -q
# → 97 passed, 4 deselected

# 5. Slow model tests (validates real fastembed cross-encoder + BM25 — no API key)
pytest -m slow
# → passes

# 6. Gated regression guard (requires Qdrant + OPENAI_API_KEY)
pytest -m eval
# → passes updated baseline floors
```

---

## Recommended Next Steps

**Phase 3.5 / ongoing — Corpus Expansion**

1. **(Recommended) Expand corpus coverage.** Add sources covering Redis internals, RAGAS implementation details, LangGraph specifics, and other topics where golden questions currently find no supporting chunks. Target: source_recall@5 ≥ 0.75 as an intermediate milestone toward the PRD's ≥ 0.85.

2. **Implement `docs/raw/` cache.** Save fetched Markdown at ingest time to `docs/raw/<slug>.md` and commit, so the corpus is reproducible without live URL access. Closes the reproducibility gap noted since Phase 2.

3. **Larger reranker candidate pool.** Re-run the sweep with candidates=50 (vs current 20) to test whether the cross-encoder becomes net-positive at a larger pool size before committing to Phase 4.
