# Handover: Phase 2 — Golden Eval Dataset (+ Option A Corpus Fix)

**Date/Time:** 2026-05-26 13:45
**Branch:** `feat/phase1-basic-rag`
**Commit range:** `bde0ffc..HEAD` (prior code HEAD before this session: `84027a5`)
**Phase:** Phase 2 — Golden Eval Dataset (+ Option A corpus fix)

---

## What Was Done

### Option A — Corpus Fix (commit bde0ffc)
Added four missing Wikipedia / AWS sources to `docs/sources.yaml`:
- Wikipedia RAG article
- AWS "What is RAG?"
- Wikipedia Vector Database
- Wikipedia Word Embedding

Corpus grew from **14 sources / 285 chunks** to **18 sources / 373 chunks**. Verified live: "What is RAG?" and "What is a vector database?" now return grounded, cited answers (previously refused due to insufficient retrieval).

### Phase 2 — Eval Harness (commit 7b3ac20)
Deterministic eval harness in `backend/app/evals/`:
- `models.py` — `GoldenQuestion`, `QuestionOutcome`, `EvalReport` Pydantic models
- `dataset.py` — YAML loader (`load_dataset`)
- `metrics.py` — `build_outcome` + `compute_metrics` (refusal_accuracy, answer_rate, source_recall_at_k, citation_validity, pass_rate)
- `runner.py` — `evaluate(questions, orchestrator, settings)` → `EvalReport`
- `report.py` — `write_report` → HTML + JSON files in `backend/reports/eval_runs/`

RAGAS/faithfulness deferred to Phase 4. MRR/chunking A-B deferred to Phase 3.

### Phase 2 — Golden Dataset (commit 6319c0d)
`backend/app/evals/golden/dataset.yaml`: **100 questions** across 7 types:

| Type | Count |
|---|---|
| definition | 20 |
| comparison | 15 |
| architecture | 15 |
| troubleshooting | 15 |
| multi_hop | 10 |
| refusal | 13 |
| adversarial | 12 |
| **Total** | **100** |

25 questions are refusal-type (should_refuse=True).

### Phase 2 — Runner + Gated Guard (commit a41b55b)
- `backend/scripts/run_evals.py` — runs the real benchmark end-to-end → writes `eval-<run_id>.{html,json}` to `backend/reports/eval_runs/`
- `pytest -m eval` gated integration test — asserts baseline floors; deselected by default (`pytest` excludes it)

### Phase 2 — Eval-Driven Prompt Fix (commit 84027a5)
First benchmark run (run `99cf20be`) exposed severe over-refusal: 64% of grounded questions were incorrectly refused. Revised `backend/app/rag/prompts/generate_answer.md` to reduce false refusals. Re-ran benchmark (run `a90c620e`) — large measured improvement (see Eval Impact below).

### This session — Code Quality + Documentation
- `report.py`: added `import html` and wrapped model-derived strings (`o.id`, `o.type`, `truncated_q`, `citations_str`) with `html.escape()` to prevent XSS in generated HTML reports.
- Removed `review-notes/q4-source-storage.md` (finding folded into Known Issues here).
- Updated `README.md`: Phase 2 status, before/after eval table, Evaluation Workflow subsection, corrected test counts.
- Updated `PRD.md`: Phase 2 marked complete in §9 roadmap, §6 baseline annotation, §10 corpus-reproducibility risk row added.

---

## Files Changed

*(commit range bde0ffc..HEAD, plus this session's doc + fix commits)*

| Area | Files |
|---|---|
| Corpus config | `docs/sources.yaml` |
| Eval harness | `backend/app/evals/models.py`, `dataset.py`, `metrics.py`, `runner.py`, `report.py` |
| Golden dataset | `backend/app/evals/golden/dataset.yaml` |
| Eval runner | `backend/scripts/run_evals.py` |
| Gated guard | `backend/tests/test_eval_guard.py` (or equivalent `pytest -m eval` test) |
| Prompt | `backend/app/rag/prompts/generate_answer.md` |
| Tests | `backend/tests/test_evals.py` |
| Plan doc | `docs/superpowers/plans/2026-05-26-phase2-eval-harness.md` |
| Docs | `README.md`, `PRD.md`, `HANDOVERS/2026-05-26_1345_phase2-eval-and-corpus.md` |

---

## Tests

```
# Fast unit + harness suite (hermetic — no keys, no network, no Qdrant)
cd backend
.venv/Scripts/python.exe -m pytest -q
→ 79 passed, 2 deselected

# Gated regression guard (requires Qdrant running + OPENAI_API_KEY)
.venv/Scripts/python.exe -m pytest -m eval
→ passes baseline floors

# Lint
.venv/Scripts/python.exe -m ruff check app tests
→ All checks passed!
```

---

## Eval Impact

The centerpiece of Phase 2: a single eval-driven prompt revision produced the following measured change.

| Metric | Before (run 99cf20be) | After (run a90c620e) | PRD Target |
|---|---|---|---|
| refusal_accuracy | 0.51 | **0.82** | ≥ 0.95 — not yet met |
| answer_rate (grounded) | 0.36 | **0.77** | — |
| source_recall@5 | 0.63 | **0.61** | ≥ 0.85 — not yet met |
| pass_rate | 0.45 | **0.62** | ≥ 0.80 |
| citation_validity | 1.00 | **1.00** | — |

Per-type pass rate at initial run (before fix):

| Type | Pass rate |
|---|---|
| refusal | 100% |
| adversarial | 92% |
| definition | 60% |
| comparison | 27% |
| architecture | 20% |
| multi_hop | 10% |
| troubleshooting | 7% |

The prompt fix cut false refusals on grounded questions WITHOUT breaking genuine/adversarial refusals (~24/25 correct on the refusal subset). The remaining gap to PRD targets is now a **retrieval/corpus problem** (source_recall ~0.61) — addressable in Phase 3 via better chunking, a reranker, and corpus coverage expansion.

---

## Known Issues / Untested

1. **refusal_accuracy 0.82 < PRD target 0.95; source_recall 0.61 < target 0.85.** The remaining gap is a retrieval/corpus problem. The Phase-3 levers are: better chunking strategy, cross-encoder reranker, and broader corpus coverage. This is expected and planned.

2. **docs/raw reproducibility discrepancy (spec-vs-code gap).** The Phase 1 design spec states that fetched raw docs are "cached for reproducibility under `docs/raw/`", but this was never implemented. `fetcher.py` fetches in memory and discards; only chunk text + metadata persists inside Qdrant. Consequently, re-ingest depends on source URLs staying live — 4 URLs already required swaps due to 403s during Phase 1. **Recommendation for Phase 3:** either implement the `docs/raw/` cache (save fetched Markdown to `docs/raw/<slug>.md` and commit), or explicitly remove the claim from the design spec.

3. **Answer completeness can be terse.** Some multi-hop and comparison questions receive technically correct but short answers. This is a prompt/Phase-4 answer-quality concern, not a retrieval bug.

4. **Thin corpus coverage.** During dataset authoring, sparse retrieval was noted for Redis internals, RAGAS implementation details, and LangGraph specifics. These topics need additional source URLs in Phase 3.

5. **No RAGAS/LLM-judge faithfulness score yet.** Faithfulness and answer correctness scoring are deferred to Phase 4; current metrics are all deterministic.

---

## How to Verify Locally

```powershell
# 1. Start Qdrant
docker compose up -d

# 2. (Re)build the corpus
cd backend
python scripts/ingest.py
# Expected: 18 sources fetched, 373 chunks upserted

# 3. Run the benchmark
python scripts/run_evals.py
# Expected: eval-<run_id>.html and eval-<run_id>.json in backend/reports/eval_runs/
# Open the HTML report in a browser to see per-question outcomes

# 4. Fast test suite
.venv/Scripts/python.exe -m pytest -q
# Expected: 79 passed, 2 deselected

# 5. Gated regression guard
.venv/Scripts/python.exe -m pytest -m eval
# Expected: passes baseline floors (requires Qdrant + OPENAI_API_KEY)
```

---

## Recommended Next Steps

**Phase 3 — Retrieval Eval Dashboard**

1. **(Recommended) Implement `docs/raw/` cache** — save fetched Markdown at ingest time so corpus is reproducible from local files independent of source URL availability.
2. **Chunking A/B** — compare current 1000/150 chars vs 500/100 and 1500/200; measure source_recall@5 for each.
3. **Cross-encoder reranker** — add a reranker step post-retrieval; target source_recall@5 ≥ 0.85 to close the PRD gap.
