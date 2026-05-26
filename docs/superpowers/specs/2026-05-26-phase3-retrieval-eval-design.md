# Phase 3 Design — Retrieval Eval & Tuning

**Date:** 2026-05-26
**Project:** AI System Design Coach (see `PRD.md`, `CLAUDE.md`)
**Phase:** 3 — Retrieval Eval Dashboard (PRD §9)
**Status:** Approved (design); ready for implementation planning
**Predecessors:** Phase 1 (Basic RAG), Phase 2 (Golden Eval Dataset) — both complete.

---

## 1. Goal & Definition of Done

Phase 2 measured retrieval at **`source_recall@5 = 0.61`** (PRD target ≥ 0.85) and showed
the remaining answer-quality gap is retrieval-bound, not generation-bound. Phase 3
**improves retrieval and proves the improvement with data.**

**Phase 3 is done when:**

1. A reproducible **retrieval A/B sweep** compares chunking, top-k, dense-vs-hybrid, and
   reranker-on/off against the 100-question golden set on deterministic retrieval metrics.
2. The sweep emits a **side-by-side HTML/Markdown comparison report** (archived in
   `reports/eval_runs/`) that declares a **winner** by a primary metric.
3. The winning config is **applied to the live `Settings` defaults**, the production
   collection is re-ingested with it, and the **Phase 2 full benchmark is re-run** to show
   a before/after — the PRD's "chunking strategies compared quantitatively, with a winner."
4. The unit/harness test suite stays green and **key-free / network-free** (real reranker
   and sparse models are exercised only under `@pytest.mark.slow`).

**Target:** move `source_recall@5` toward ≥ 0.85 and, via better-grounded answers, lift
`refusal_accuracy` toward the PRD's 0.95. (These are aspirations for the winner, not a
hard gate on Phase 3 completion — the deliverable is the measured comparison + applied
winner, even if a target isn't fully reached.)

---

## 2. Locked Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| A/B levers | **All four:** cross-encoder reranker on/off, top-k sweep, chunk-size/overlap sweep, hybrid (dense+sparse) vs dense |
| Reranker | **Local cross-encoder** via fastembed `TextCrossEncoder` (no API key, ONNX, no torch) |
| Hybrid | **In scope this phase** (dense + sparse BM25/SPLADE, fused in Qdrant Query API) — heaviest item |
| Metrics | **Deterministic only**: Recall@k, Precision@k, MRR, hit-rate@k from `required_sources`. LLM-judged context precision deferred to Phase 4 |
| Output | **HTML/Markdown comparison report** (extend `report.py`); React "Compare runs" UI deferred |
| Sweep style | **Staged** (tune one lever, lock the winner, proceed) over a configurable matrix; full-grid optional |
| Approach | **A** — config-matrix sweep with named collections + a retrieval-only evaluator |

---

## 3. Architecture

```
                         ┌──────────── A/B Sweep (no LLM) ────────────┐
golden dataset ──▶ RetrievalEvaluator ──▶ retrieval_metrics ──▶ Comparison report (HTML/MD)
(grounded Qs)         │  uses a Retriever bound to a variant:        │   ranks variants, picks winner
                      │   • collection (chunk_size, dense|hybrid)    │
                      │   • query cfg (top_k, rerank on/off)         │
                      └──────────────────────────────────────────────┘
Index variants (built once, reused):
  sources.yaml ─▶ ingest(variant) ─▶ Qdrant collection  ai_coach_cs{N}_{dense|hybrid}
                                       (hybrid = named dense + sparse vectors)
Live pipeline (unchanged shape, now retrieval-configurable):
  /query ─▶ orchestrator ─▶ Retriever[dense|hybrid → candidate pool → rerank → top-k] ─▶ generator …
```

The **sweep and the live pipeline share the same `Retriever`/`Reranker` code paths** — the
sweep just instantiates them with different configs. Applying the winner = setting
`Settings` defaults; no separate "eval-only retrieval" implementation.

**Separation of variant cost classes:**
- **Ingest-time variants** (chunk size/overlap, dense-vs-hybrid) → a distinct Qdrant
  collection each; built once.
- **Query-time variants** (top-k, reranker on/off) → same collection, varied params.

---

## 4. Components & file map

```
backend/app/
  rag/
    reranker.py            # NEW: Reranker protocol, LocalCrossEncoderReranker (fastembed), NoopReranker, factory
    retriever.py           # EXTEND: candidate pool (retrieve_n), optional rerank, optional hybrid (sparse+RRF fusion)
    qdrant_store.py        # EXTEND: hybrid collection schema (named dense+sparse vectors), sparse upsert, collection naming
    ingest.py              # EXTEND: build a collection for a given IngestVariant (chunk size, hybrid)
    sparse.py              # NEW: SparseEmbeddingProvider (fastembed SparseTextEmbedding) + Fake for tests
  evals/
    retrieval_metrics.py   # NEW: recall_at_k, precision_at_k, mrr, hit_rate_at_k (pure, no LLM)
    retrieval_eval.py      # NEW: RetrievalEvaluator.evaluate(questions, retriever) -> RetrievalReport
    sweep.py               # NEW: variant matrix, staged sweep driver, winner selection
    comparison_report.py   # NEW (or extend report.py): side-by-side HTML/MD of variants + winner
  config.py                # EXTEND: rerank_*, hybrid_*, sparse_model settings
backend/scripts/
  run_sweep.py             # NEW: CLI — run the sweep, write the comparison report
backend/tests/
  test_retrieval_metrics.py, test_retrieval_eval.py, test_reranker.py, test_hybrid.py, test_sweep.py
```

Each unit has one job and a typed interface; fakes exist for the network/model-bound ones
so the default suite is hermetic.

---

## 5. Retrieval metrics (`retrieval_metrics.py`)

Pure functions over a single question's **ranked retrieved source list** and its
`required_sources` set, then averaged across grounded questions. Definitions:

- **hit_rate@k** — 1 if any required source appears in the top-k retrieved, else 0.
  (Identical to Phase 2's `source_recall_at_k`; carried for continuity.)
- **recall@k** — (# distinct required sources present in top-k) / (# required sources).
- **precision@k** — (# of top-k retrieved chunks whose source ∈ required_sources) / k.
- **mrr** — 1 / (rank of the first retrieved chunk whose source ∈ required_sources); 0 if none.

**Relevance definition (explicit):** a retrieved chunk is "relevant" iff its `source_url`
∈ `required_sources`. This is **source-level** relevance — a deterministic, reproducible
proxy. It does NOT judge whether the chunk's *text* actually answers the question (that
needs an LLM/human judge → Phase 4). The spec states this limitation so the numbers aren't
over-read.

Empty-input safety (no ZeroDivisionError); all ratios rounded to 4 dp; functions take
plain lists so they're trivially unit-testable with synthetic rankings.

---

## 6. Retrieval-only evaluator (`retrieval_eval.py`)

`RetrievalEvaluator.evaluate(questions, retriever, label) -> RetrievalReport` where:
- `questions` = grounded golden questions (those with `required_sources`).
- `retriever` = any object with `.search(query) -> list[RetrievedChunk]` (the real
  `Retriever` bound to a variant, or a fake in tests).
- Per question: run `retriever.search`, take the ranked `source_url`s, compute per-question
  metrics; aggregate to a `RetrievalReport{label, config, n_questions, metrics, per_question}`.
- **No orchestrator, no generator, no LLM.** Pure retrieval. Fast and free.

This is the engine the sweep calls once per (variant × query-config).

---

## 7. Retriever extensions (real product code)

### 7.1 Reranker (`reranker.py`)
- `Reranker` Protocol: `rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]`.
- `LocalCrossEncoderReranker(model_name)` — fastembed `TextCrossEncoder`; scores (query, chunk.text)
  pairs, sorts desc, returns top-k (rewriting `n` to the new 1-based rank). Default model
  candidate: `Xenova/ms-marco-MiniLM-L-6-v2` (verify fastembed support at impl time; fall
  back to another supported cross-encoder, e.g. `jinaai/jina-reranker-v1-tiny-en`).
- `NoopReranker` — returns the first top-k unchanged (used when rerank disabled, and as a
  test double base).
- `get_reranker(settings)` factory: returns Local or Noop by `settings.rerank_enabled`.

### 7.2 Retriever flow (`retriever.py`)
With reranking enabled: embed query → retrieve **`rerank_candidates`** (e.g. 20) by dense
(or hybrid) → `reranker.rerank(query, candidates, top_k)` → return top-k. With reranking
disabled: retrieve top-k directly (current behavior). The `Retriever` is constructed with
an optional reranker and an optional hybrid mode; both default off so existing behavior and
tests are unchanged unless configured.

### 7.3 Hybrid search (`sparse.py`, `qdrant_store.py`, `retriever.py`)
- `SparseEmbeddingProvider` — fastembed `SparseTextEmbedding` (default `Qdrant/bm25`;
  SPLADE optional), returns sparse vectors (indices+values). `FakeSparseProvider` for tests.
- Hybrid collections use Qdrant **named vectors**: a dense vector (`"dense"`) + a sparse
  vector (`"sparse"`). `ensure_collection` gains a `hybrid: bool` path with the named-vector
  schema; `upsert_chunks` writes both vectors when hybrid.
- Hybrid query uses Qdrant's **Query API with prefetch + `FusionQuery(RRF)`**: prefetch
  dense and sparse candidate lists, fuse with Reciprocal Rank Fusion, take the fused top-N
  (then optional rerank → top-k).

---

## 8. Index variant management (`qdrant_store.py`, `ingest.py`)

- An `IngestVariant{chunk_size, overlap, hybrid}` maps to a deterministic collection name,
  e.g. `ai_coach_cs1000_dense`, `ai_coach_cs500_hybrid`.
- `build_index(variant, sources, ...)` = re-chunk at the variant's size, embed (dense, +
  sparse if hybrid), upsert into the variant's collection; idempotent (stable point IDs).
- Collections are built once and reused across the sweep. The sweep skips a collection that
  already exists with the right point count (cheap re-runs).

---

## 9. Sweep driver, matrix & comparison report

### 9.1 Default staged sweep (`sweep.py`)
Run levers in stages, locking each winner before the next (cheaper, clearer narrative):
1. **Chunk size** (dense, no rerank, k=5): {500, 1000, 1500} → pick best by Recall@5.
2. **Retrieval mode** at the winning chunk size: dense vs hybrid → pick best.
3. **Reranker**: off vs on (candidates=20) at the winning index → pick best.
4. **top-k**: {5, 10} (mostly informational; Recall rises with k) → record.

The matrix is declarative and configurable (a full grid is possible by supplying the full
cross-product). Primary metric = **Recall@5**, tie-break **MRR**.

### 9.2 Comparison report (`comparison_report.py`)
A self-contained HTML page (+ a Markdown sibling) with one row per evaluated variant
(label, config, hit_rate@5, recall@5, precision@5, MRR, n_questions, build/eval time), the
**winner highlighted**, and a short "stage-by-stage winner" summary. Archived as
`reports/eval_runs/sweep-<run_id>.{html,md,json}`.

### 9.3 CLI (`scripts/run_sweep.py`)
Builds the needed collections (Qdrant up; local models download once), runs the staged
sweep over the grounded golden questions, writes the comparison report, prints the winner.
No OpenAI key required (retrieval-only).

---

## 10. Closing the loop (apply winner → re-measure)

1. Set the winning config in `Settings` defaults (chunk size, hybrid on/off, rerank on/off,
   top-k, candidates) and point `QDRANT_COLLECTION` at the winning collection (or re-ingest
   the canonical `ai_coach_docs` with the winning ingest config).
2. Re-run `scripts/run_evals.py` (Phase 2 full benchmark, needs key) for a **before/after**:
   prior baseline (refusal_accuracy 0.82, source_recall 0.61, pass_rate 0.62) vs the tuned
   pipeline. Update the Phase 2 regression-guard floors and README/PRD scores intentionally.
3. This produces the PRD Phase-3 artifact ("compared quantitatively, with a winner") plus
   the gating-metric movement.

---

## 11. New configuration (`config.py`)

| Var | Default | Purpose |
|---|---|---|
| `RERANK_ENABLED` | `false` | turn the cross-encoder reranker on |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed cross-encoder |
| `RERANK_CANDIDATES` | `20` | candidate pool size before rerank |
| `HYBRID_ENABLED` | `false` | dense+sparse hybrid retrieval |
| `SPARSE_MODEL` | `Qdrant/bm25` | fastembed sparse model |

Defaults keep the **current behavior** (dense, no rerank) until the winner is applied, so
nothing changes for existing tests/users until we deliberately flip them in §10.

---

## 12. Testing strategy (CLAUDE Rule 1)

- **Pure metrics** (`test_retrieval_metrics.py`): synthetic ranked-source lists → known
  Recall/Precision/MRR/hit-rate values (hand-computed fractions).
- **Retrieval evaluator** (`test_retrieval_eval.py`): a `FakeRetriever` returning canned
  ranked chunks → assert aggregated metrics.
- **Reranker** (`test_reranker.py`): a deterministic fake reranker reorders predictably;
  `NoopReranker` passthrough; the **real** `LocalCrossEncoderReranker` under `@pytest.mark.slow`.
- **Hybrid** (`test_hybrid.py`): in-memory Qdrant with named dense+sparse vectors +
  `FakeSparseProvider` → assert fused results return; real sparse model under `slow`.
- **Sweep** (`test_sweep.py`): drive the staged sweep with fakes (in-memory Qdrant + fake
  providers) → assert it produces a ranked comparison + a winner; report writer to tmp_path.
- Default `pytest` remains key-free, network-free, fast; `slow`/`eval` markers gate the
  model-downloading / real-pipeline tests.

---

## 13. Dependencies

`fastembed` (already installed) provides both `SparseTextEmbedding` (BM25/SPLADE) and
`TextCrossEncoder` (rerankers) in current versions — **verify the exact model names are
supported at implementation time**; if a chosen model isn't available, fall back to another
fastembed-supported one and note it. `qdrant-client` 1.18 (installed) supports named
vectors + sparse + the Query API fusion. No new top-level dependencies expected.

---

## 14. Out of scope (deferred)

- **LLM-judged context precision / RAGAS faithfulness/correctness** → Phase 4.
- **React "Compare runs" UI** (PRD §7.4) → later; Phase 3 ships HTML/MD reports.
- **Embedding-model sweep** (bge-small vs bge-base vs OpenAI) — *available* via existing
  config but not a primary Phase-3 axis unless requested.
- **LangSmith dashboards / regression alerts** → Phase 5.

---

## 15. Autonomous assumptions (flagged for review)

1. **Model names** (`Xenova/ms-marco-MiniLM-L-6-v2`, `Qdrant/bm25`) verified at impl time;
   fall back to alternates if unsupported by the installed fastembed.
2. **Default sweep matrix**: chunk_size {500, 1000, 1500} × {dense, hybrid}, reranker
   on/off (candidates=20), k∈{5,10}, run **staged**; primary metric Recall@5, tie-break MRR.
3. **Source-level relevance** is the deterministic relevance definition (not chunk-content
   relevance) — true content relevance is Phase 4.
4. **Re-ingest cost**: building 3–4 extra collections re-fetches/re-embeds the corpus
   (local, free, a few minutes each); sweeps skip already-built collections.
5. Winner applied by updating `Settings` defaults + re-ingesting the canonical collection;
   Phase 2 baseline floors updated intentionally to the improved numbers.
