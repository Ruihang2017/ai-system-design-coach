# Phase 3 — Retrieval Eval & Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve retrieval quality and prove it with data — a deterministic, no-LLM retrieval A/B sweep over chunk-size, dense-vs-hybrid, reranker on/off, and top-k against the 100-question golden set, emitting a comparison report with a winner, then applying the winner and re-running the Phase 2 benchmark for a before/after.

**Architecture:** A retrieval-only evaluator scores `Retriever` outputs against `required_sources` (source-level relevance, no LLM). The `Retriever` gains a candidate-pool → optional cross-encoder rerank → top-k flow and an optional dense+sparse hybrid (Qdrant RRF fusion). Ingest-time variants (chunk size, hybrid) build distinctly-named Qdrant collections; query-time variants (top-k, rerank) vary params on the same collection. A staged sweep drives the matrix and writes an HTML/MD/JSON comparison report.

**Tech Stack:** Python 3.12, fastembed (`TextCrossEncoder` reranker, `SparseTextEmbedding` BM25), Qdrant (named dense+sparse vectors, Query API fusion), pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-05-26-phase3-retrieval-eval-design.md`

---

## Conventions (every task)
- Commands run from `backend/` with the venv: `.venv/Scripts/python.exe -m pytest <args>` / `-m ruff check app tests`.
- All unit tests pass with **no keys, no network, no Qdrant server** (use fakes + `QdrantClient(location=":memory:")`). Real fastembed reranker/sparse models are exercised only under `@pytest.mark.slow` (deselected by default).
- TDD: write the failing test, see it fail, implement, see it pass, commit. Conventional commits; **no `git push`**; trailer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`. Never stage `.env`, `qdrant_storage/`, generated reports, or `CLAUDE.md`.
- **API-verification callouts:** three calls depend on the installed library versions and are marked **⚠️ VERIFY** — confirm the exact API against `fastembed`/`qdrant-client` at implementation time and adapt if needed (do not weaken tests; fakes make the suite hermetic regardless).

## File map
```
backend/app/
  config.py                  # MODIFY: rerank_* + hybrid_* + sparse_model settings
  rag/
    reranker.py              # CREATE: Reranker protocol, NoopReranker, LocalCrossEncoderReranker, FakeReranker, get_reranker
    sparse.py                # CREATE: SparseVec, SparseEmbeddingProvider, FakeSparseProvider, get_sparse_provider
    qdrant_store.py          # MODIFY: collection_name(), hybrid ensure_collection, sparse-aware upsert_chunks
    retriever.py             # MODIFY: candidate pool + rerank + hybrid fusion
    ingest.py                # MODIFY: IngestVariant, _ingest_sources helper, build_index(); run_ingest reuses helper
  evals/
    models.py                # MODIFY: add RetrievalReport, SweepReport
    retrieval_metrics.py     # CREATE: recall_at_k, precision_at_k, mrr, hit_rate_at_k
    retrieval_eval.py        # CREATE: evaluate_retrieval()
    sweep.py                 # CREATE: SweepContext, evaluate_variant(), run_staged_sweep()
    comparison_report.py     # CREATE: write_comparison_report()
backend/scripts/
  run_sweep.py               # CREATE: CLI to run the staged sweep + write report
backend/tests/
  test_retrieval_metrics.py test_retrieval_eval.py test_reranker.py test_sparse.py
  test_qdrant_hybrid.py test_retriever_rerank.py test_ingest_variant.py test_sweep.py test_config_phase3.py
```

## Dependency / parallelization notes
- **T1 (config), T2 (metrics)** independent, do first. **T3 (models+evaluator)** needs T2. **T4 (reranker), T5 (sparse)** independent.
- **T6 (qdrant_store+retriever hybrid/rerank)** needs T4, T5. **T7 (ingest variant)** needs T5, T6. **T8 (sweep+report)** needs T3, T6, T7.
- **T9 (run real sweep)** needs all code + Qdrant + local models (controller runs it). **T10 (apply winner + re-measure + docs)** is last (controller, data-dependent).

---

### Task 1: Phase 3 config settings

**Files:** Modify `backend/app/config.py`; Test `backend/tests/test_config_phase3.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_config_phase3.py`
```python
from app.config import Settings


def test_phase3_defaults_preserve_current_behavior():
    s = Settings(_env_file=None)
    assert s.rerank_enabled is False
    assert s.hybrid_enabled is False
    assert s.rerank_candidates == 20
    assert s.rerank_model and s.sparse_model
```

- [ ] **Step 2: Run → FAIL** `... -m pytest tests/test_config_phase3.py -v` (AttributeError).

- [ ] **Step 3: Implement** — add to `Settings` in `app/config.py` (after the `rewrite_enabled` line):
```python
    # Retrieval tuning (Phase 3). Defaults keep current behavior (dense, no rerank).
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    hybrid_enabled: bool = False
    sparse_model: str = "Qdrant/bm25"
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit** `git add backend/app/config.py backend/tests/test_config_phase3.py && git commit -m "feat: add Phase 3 retrieval tuning settings"`

---

### Task 2: Retrieval metrics (pure, no LLM)

**Files:** Create `backend/app/evals/retrieval_metrics.py`; Test `backend/tests/test_retrieval_metrics.py`

- [ ] **Step 1: Write the failing test**
```python
from app.evals.retrieval_metrics import hit_rate_at_k, recall_at_k, precision_at_k, mrr


def test_hit_rate():
    assert hit_rate_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], ["z"], 3) == 0.0
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 2) == 0.0  # c is rank 3, outside k=2


def test_recall_at_k():
    # 1 of 2 required sources present in top-3
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
    assert recall_at_k(["a", "b"], [], 3) == 0.0  # no required -> 0


def test_precision_at_k():
    # 1 relevant in top-3 -> 1/3
    assert precision_at_k(["a", "b", "c"], ["b"], 3) == 0.3333
    assert precision_at_k(["a", "b", "c"], ["a", "b"], 2) == 1.0
    assert precision_at_k([], ["a"], 0) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], ["b"]) == 0.5      # first relevant at rank 2
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    assert mrr(["a", "b", "c"], ["z"]) == 0.0
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** `backend/app/evals/retrieval_metrics.py`
```python
"""Deterministic, source-level retrieval metrics (no LLM).

A retrieved chunk is "relevant" iff its source_url is in the question's
required_sources. Inputs are the ranked list of retrieved source_urls (rank order,
may contain duplicates) and the list of required source_urls.
"""


def hit_rate_at_k(ranked_sources: list[str], required: list[str], k: int) -> float:
    req = set(required)
    return 1.0 if any(s in req for s in ranked_sources[:k]) else 0.0


def recall_at_k(ranked_sources: list[str], required: list[str], k: int) -> float:
    req = set(required)
    if not req:
        return 0.0
    found = {s for s in ranked_sources[:k] if s in req}
    return round(len(found) / len(req), 4)


def precision_at_k(ranked_sources: list[str], required: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    req = set(required)
    relevant = sum(1 for s in ranked_sources[:k] if s in req)
    return round(relevant / k, 4)


def mrr(ranked_sources: list[str], required: list[str]) -> float:
    req = set(required)
    for rank, s in enumerate(ranked_sources, start=1):
        if s in req:
            return round(1.0 / rank, 4)
    return 0.0
```

- [ ] **Step 4: Run → PASS** (note `0.3333` is `round(1/3, 4)`).

- [ ] **Step 5: Commit** `git add backend/app/evals/retrieval_metrics.py backend/tests/test_retrieval_metrics.py && git commit -m "feat: add deterministic retrieval metrics (recall/precision/mrr/hit-rate)"`

---

### Task 3: Retrieval report models + retrieval-only evaluator

**Files:** Modify `backend/app/evals/models.py`; Create `backend/app/evals/retrieval_eval.py`; Test `backend/tests/test_retrieval_eval.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_retrieval_eval.py`
```python
from app.evals.models import GoldenQuestion, RetrievalReport
from app.evals.retrieval_eval import evaluate_retrieval
from app.rag.models import Chunk, RetrievedChunk


class FakeRetriever:
    """Returns canned chunks keyed by question text."""
    def __init__(self, mapping):
        self._mapping = mapping  # question -> list[source_url]
    def search(self, query, top_k=None):
        srcs = self._mapping.get(query, [])
        return [
            RetrievedChunk(
                chunk=Chunk(id=f"x::{i}", text="t", source_url=u, title="T", doc_id="x", chunk_index=i),
                score=1.0 - i * 0.1, n=i + 1,
            )
            for i, u in enumerate(srcs)
        ]


def test_evaluate_retrieval_aggregates_metrics():
    qs = [
        GoldenQuestion(id="q1", type="definition", question="A?", required_sources=["s1"]),
        GoldenQuestion(id="q2", type="definition", question="B?", required_sources=["s2"]),
        GoldenQuestion(id="q3", type="refusal", question="R?", should_refuse=True),  # skipped (no required)
    ]
    retriever = FakeRetriever({"A?": ["s1", "sx"], "B?": ["sy", "sz"]})  # q1 hits, q2 misses
    report = evaluate_retrieval(qs, retriever, k=5, label="test", config={"foo": 1})
    assert isinstance(report, RetrievalReport)
    assert report.n_questions == 2  # only grounded
    assert report.metrics["hit_rate_at_5"] == 0.5
    assert report.metrics["recall_at_5"] == 0.5
    assert report.label == "test" and report.config == {"foo": 1}
    assert len(report.per_question) == 2
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Add models** — append to `backend/app/evals/models.py`:
```python
class RetrievalReport(BaseModel):
    label: str
    config: dict = Field(default_factory=dict)
    n_questions: int
    metrics: dict[str, float] = Field(default_factory=dict)
    per_question: list[dict] = Field(default_factory=list)


class SweepReport(BaseModel):
    run_id: str
    ts: str
    variants: list[RetrievalReport] = Field(default_factory=list)
    stage_winners: dict = Field(default_factory=dict)
    winner_label: str = ""
    winner_config: dict = Field(default_factory=dict)
```

- [ ] **Step 4: Implement** `backend/app/evals/retrieval_eval.py`
```python
"""Retrieval-only evaluator: score a Retriever against the golden set (no LLM)."""

from typing import Any

from app.evals.models import GoldenQuestion, RetrievalReport
from app.evals.retrieval_metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k


def evaluate_retrieval(
    questions: list[GoldenQuestion],
    retriever: Any,
    k: int = 5,
    label: str = "",
    config: dict | None = None,
) -> RetrievalReport:
    grounded = [q for q in questions if q.required_sources]
    per_q: list[dict] = []
    for q in grounded:
        chunks = retriever.search(q.question)
        ranked = [c.chunk.source_url for c in chunks]
        per_q.append({
            "id": q.id,
            "type": q.type,
            "recall": recall_at_k(ranked, q.required_sources, k),
            "precision": precision_at_k(ranked, q.required_sources, k),
            "hit": hit_rate_at_k(ranked, q.required_sources, k),
            "mrr": mrr(ranked, q.required_sources),
            "ranked_sources": ranked[:k],
        })

    def _mean(key: str) -> float:
        return round(sum(d[key] for d in per_q) / len(per_q), 4) if per_q else 0.0

    metrics = {
        f"recall_at_{k}": _mean("recall"),
        f"precision_at_{k}": _mean("precision"),
        f"hit_rate_at_{k}": _mean("hit"),
        "mrr": _mean("mrr"),
    }
    return RetrievalReport(
        label=label, config=config or {}, n_questions=len(grounded),
        metrics=metrics, per_question=per_q,
    )
```

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit** `git add backend/app/evals/models.py backend/app/evals/retrieval_eval.py backend/tests/test_retrieval_eval.py && git commit -m "feat: add retrieval report models and retrieval-only evaluator"`

---

### Task 4: Cross-encoder reranker

**Files:** Create `backend/app/rag/reranker.py`; Test `backend/tests/test_reranker.py`

- [ ] **Step 1: Write the failing test**
```python
import pytest
from app.config import Settings
from app.rag.models import Chunk, RetrievedChunk
from app.rag.reranker import NoopReranker, FakeReranker, get_reranker


def _chunks(texts):
    return [
        RetrievedChunk(chunk=Chunk(id=f"d::{i}", text=t, source_url="u", title="T", doc_id="d", chunk_index=i), score=1.0 - i * 0.1, n=i + 1)
        for i, t in enumerate(texts)
    ]


def test_noop_returns_top_k_renumbered():
    out = NoopReranker().rerank("q", _chunks(["a", "b", "c"]), top_k=2)
    assert [c.chunk.text for c in out] == ["a", "b"]
    assert [c.n for c in out] == [1, 2]


def test_fake_reranker_reorders_by_text_length():
    out = FakeReranker().rerank("q", _chunks(["aa", "bbbb", "c"]), top_k=2)
    assert [c.chunk.text for c in out] == ["bbbb", "aa"]
    assert [c.n for c in out] == [1, 2]


def test_factory_returns_noop_when_disabled():
    assert isinstance(get_reranker(Settings(_env_file=None, rerank_enabled=False)), NoopReranker)


@pytest.mark.slow
def test_local_cross_encoder_reranks():
    from app.rag.reranker import LocalCrossEncoderReranker
    r = LocalCrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    out = r.rerank("what is hybrid search?", _chunks([
        "Bananas are yellow fruit.",
        "Hybrid search combines dense and sparse retrieval.",
    ]), top_k=1)
    assert out[0].chunk.text.startswith("Hybrid search")
```

- [ ] **Step 2: Run → FAIL** (default suite; slow test deselected).

- [ ] **Step 3: Implement** `backend/app/rag/reranker.py`
```python
"""Cross-encoder rerankers. Local fastembed model; Noop/Fake for the no-rerank path & tests."""

from typing import Protocol, runtime_checkable

from app.config import Settings
from app.rag.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


def _renumber(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [c.model_copy(update={"n": i + 1}) for i, c in enumerate(chunks)]


class NoopReranker:
    """No reranking: keep retrieval order, take top_k, renumber n."""
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return _renumber(chunks[:top_k])


class FakeReranker:
    """Deterministic test double: rank by descending text length."""
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ranked = sorted(chunks, key=lambda c: len(c.chunk.text), reverse=True)[:top_k]
        return _renumber(ranked)


class LocalCrossEncoderReranker:
    """fastembed TextCrossEncoder. Downloads the model on first use."""
    def __init__(self, model_name: str) -> None:
        from fastembed import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        # ⚠️ VERIFY fastembed API: TextCrossEncoder.rerank(query, documents) -> iterable[float]
        scores = list(self._model.rerank(query, [c.chunk.text for c in chunks]))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
        ranked = [chunks[i].model_copy(update={"score": float(scores[i])}) for i in order]
        return _renumber(ranked)


def get_reranker(settings: Settings) -> Reranker:
    if settings.rerank_enabled:
        return LocalCrossEncoderReranker(settings.rerank_model)
    return NoopReranker()
```

- [ ] **Step 4: Run → PASS** (3 passed, 1 slow deselected).

- [ ] **Step 5: Commit** `git add backend/app/rag/reranker.py backend/tests/test_reranker.py && git commit -m "feat: add cross-encoder reranker (local fastembed, noop/fake for tests)"`

---

### Task 5: Sparse embedding provider

**Files:** Create `backend/app/rag/sparse.py`; Test `backend/tests/test_sparse.py`

- [ ] **Step 1: Write the failing test**
```python
import pytest
from app.config import Settings
from app.rag.sparse import SparseVec, FakeSparseProvider, get_sparse_provider


def test_fake_sparse_is_deterministic_and_shaped():
    p = FakeSparseProvider()
    v1 = p.embed_query("hybrid search retrieval")
    v2 = p.embed_query("hybrid search retrieval")
    assert isinstance(v1, SparseVec)
    assert v1.indices == v2.indices and v1.values == v2.values
    assert len(v1.indices) == len(v1.values) and len(v1.indices) > 0
    docs = p.embed_documents(["a b", "c d e"])
    assert len(docs) == 2 and all(isinstance(d, SparseVec) for d in docs)


def test_factory_returns_fake():
    assert isinstance(get_sparse_provider(Settings(_env_file=None, sparse_model="fake")), FakeSparseProvider)


@pytest.mark.slow
def test_real_bm25_sparse():
    from app.rag.sparse import SparseEmbeddingProvider
    p = SparseEmbeddingProvider("Qdrant/bm25")
    v = p.embed_query("retrieval augmented generation")
    assert len(v.indices) == len(v.values) and len(v.indices) > 0
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** `backend/app/rag/sparse.py`
```python
"""Sparse embeddings for hybrid retrieval. Local fastembed BM25/SPLADE; Fake for tests."""

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.config import Settings


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


@runtime_checkable
class SparseProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVec]: ...
    def embed_query(self, text: str) -> SparseVec: ...


class FakeSparseProvider:
    """Deterministic hashed bag-of-words sparse vectors for tests."""
    def __init__(self, dim: int = 10000) -> None:
        self._dim = dim

    def _vec(self, text: str) -> SparseVec:
        counts: dict[int, float] = {}
        for tok in text.lower().split():
            idx = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self._dim
            counts[idx] = counts.get(idx, 0.0) + 1.0
        items = sorted(counts.items())
        return SparseVec(indices=[i for i, _ in items], values=[v for _, v in items])

    def embed_query(self, text: str) -> SparseVec:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._vec(t) for t in texts]


class SparseEmbeddingProvider:
    """fastembed SparseTextEmbedding (default BM25). Downloads on first use."""
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name)
        self._model_name = model_name

    @staticmethod
    def _to_vec(emb) -> SparseVec:
        return SparseVec(indices=[int(i) for i in emb.indices], values=[float(v) for v in emb.values])

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        # ⚠️ VERIFY fastembed API: SparseTextEmbedding.embed(texts) -> iterable with .indices/.values
        return [self._to_vec(e) for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVec:
        # BM25 provides query_embed; fall back to embed for models without it.
        embed_fn = getattr(self._model, "query_embed", self._model.embed)
        return self._to_vec(next(iter(embed_fn([text]))))


def get_sparse_provider(settings: Settings) -> SparseProvider:
    if settings.sparse_model == "fake":
        return FakeSparseProvider()
    return SparseEmbeddingProvider(settings.sparse_model)
```

- [ ] **Step 4: Run → PASS** (2 passed, 1 slow deselected).

- [ ] **Step 5: Commit** `git add backend/app/rag/sparse.py backend/tests/test_sparse.py && git commit -m "feat: add sparse embedding provider (fastembed bm25, fake for tests)"`

---

### Task 6: Qdrant hybrid store + retriever rerank/hybrid flow

**Files:** Modify `backend/app/rag/qdrant_store.py`, `backend/app/rag/retriever.py`; Tests `backend/tests/test_qdrant_hybrid.py`, `backend/tests/test_retriever_rerank.py`

- [ ] **Step 1: Write failing test** `backend/tests/test_retriever_rerank.py`
```python
from qdrant_client import QdrantClient

from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.models import Chunk
from app.rag.qdrant_store import ensure_collection, upsert_chunks
from app.rag.reranker import FakeReranker, NoopReranker
from app.rag.retriever import Retriever


def _seed(n=5):
    client = QdrantClient(location=":memory:")
    emb = FakeEmbeddingProvider(dim=8)
    ensure_collection(client, "c", emb.dim)
    chunks = [Chunk(id=f"d::{i}", text="x" * (i + 1), source_url=f"u{i}", title=f"T{i}", doc_id="d", chunk_index=i) for i in range(n)]
    upsert_chunks(client, "c", chunks, emb.embed_documents([c.text for c in chunks]))
    return client, emb


def test_dense_retriever_without_reranker_unchanged():
    client, emb = _seed()
    r = Retriever(client, "c", emb, top_k=3)  # NoopReranker by default
    out = r.search("xxx")
    assert len(out) == 3 and out[0].n == 1


def test_reranker_reorders_candidates():
    client, emb = _seed(n=5)
    # FakeReranker ranks by longest text; with candidates>=5 the longest ("xxxxx") wins.
    r = Retriever(client, "c", emb, top_k=2, reranker=FakeReranker(), rerank_candidates=5)
    out = r.search("anything")
    assert out[0].chunk.text == "xxxxx" and out[0].n == 1
```

- [ ] **Step 2: Run → FAIL** (Retriever has no `reranker`/`rerank_candidates` params).

- [ ] **Step 3: Modify `qdrant_store.py`** — replace its contents with (adds `collection_name`, hybrid `ensure_collection`, sparse-aware `upsert_chunks`):
```python
"""Qdrant collection management and upserts (dense + optional hybrid sparse)."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, SparseVector, SparseVectorParams, VectorParams,
)

from app.constants import QDRANT_NAMESPACE
from app.rag.models import Chunk
from app.rag.sparse import SparseVec


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(QDRANT_NAMESPACE, chunk_id))


def collection_name(chunk_size: int, hybrid: bool) -> str:
    return f"ai_coach_cs{chunk_size}_{'hybrid' if hybrid else 'dense'}"


def ensure_collection(client: QdrantClient, collection: str, dim: int, hybrid: bool = False) -> None:
    if client.collection_exists(collection):
        return
    if hybrid:
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVec] | None = None,
) -> None:
    points = []
    for i, (c, dv) in enumerate(zip(chunks, dense_vectors)):
        if sparse_vectors is not None:
            sv = sparse_vectors[i]
            vector = {"dense": dv, "sparse": SparseVector(indices=sv.indices, values=sv.values)}
        else:
            vector = dv
        points.append(PointStruct(id=point_id(c.id), vector=vector, payload=c.model_dump()))
    if points:
        client.upsert(collection_name=collection, points=points)
```

- [ ] **Step 4: Modify `retriever.py`** — replace its contents with:
```python
"""Vector retrieval from Qdrant: dense or hybrid, with optional reranking."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.providers.embeddings import EmbeddingProvider
from app.rag.models import Chunk, RetrievedChunk
from app.rag.reranker import NoopReranker, Reranker


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: EmbeddingProvider,
        top_k: int = 5,
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
        hybrid: bool = False,
        sparse_embedder: Any = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._top_k = top_k
        self._reranker = reranker or NoopReranker()
        self._rerank_candidates = rerank_candidates
        self._hybrid = hybrid
        self._sparse = sparse_embedder

    def _fetch_n(self, k: int) -> int:
        return max(k, self._rerank_candidates) if not isinstance(self._reranker, NoopReranker) else k

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self._top_k
        fetch_n = self._fetch_n(k)
        dense = self._embedder.embed_query(query)

        if self._hybrid:
            sv = self._sparse.embed_query(query)
            # ⚠️ VERIFY qdrant-client API: prefetch + FusionQuery(RRF) over named "dense"/"sparse".
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    qm.Prefetch(query=dense, using="dense", limit=fetch_n),
                    qm.Prefetch(
                        query=qm.SparseVector(indices=sv.indices, values=sv.values),
                        using="sparse", limit=fetch_n,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=fetch_n,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection, query=dense, limit=fetch_n, with_payload=True,
            )

        candidates = [
            RetrievedChunk(chunk=Chunk(**p.payload), score=p.score, n=i + 1)
            for i, p in enumerate(response.points)
        ]
        return self._reranker.rerank(query, candidates, k)
```

- [ ] **Step 5: Run → PASS** `... -m pytest tests/test_retriever_rerank.py tests/test_retriever.py -v` (the original `test_retriever.py` must still pass — defaults preserve behavior).

- [ ] **Step 6: Write hybrid test** `backend/tests/test_qdrant_hybrid.py`
```python
from qdrant_client import QdrantClient

from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.models import Chunk
from app.rag.qdrant_store import collection_name, ensure_collection, upsert_chunks
from app.rag.retriever import Retriever
from app.rag.sparse import FakeSparseProvider


def test_collection_name():
    assert collection_name(1000, False) == "ai_coach_cs1000_dense"
    assert collection_name(500, True) == "ai_coach_cs500_hybrid"


def test_hybrid_upsert_and_fused_search():
    client = QdrantClient(location=":memory:")
    emb = FakeEmbeddingProvider(dim=8)
    sparse = FakeSparseProvider()
    ensure_collection(client, "h", emb.dim, hybrid=True)
    chunks = [Chunk(id=f"d::{i}", text=t, source_url=f"u{i}", title=f"T{i}", doc_id="d", chunk_index=i)
              for i, t in enumerate(["alpha retrieval", "beta chunking"])]
    upsert_chunks(client, "h", chunks, emb.embed_documents([c.text for c in chunks]),
                  sparse_vectors=sparse.embed_documents([c.text for c in chunks]))
    r = Retriever(client, "h", emb, top_k=2, hybrid=True, sparse_embedder=sparse)
    out = r.search("alpha retrieval")
    assert len(out) >= 1
    assert any(c.chunk.text == "alpha retrieval" for c in out)
```

- [ ] **Step 7: Run → PASS.** ⚠️ If the in-memory client rejects the fusion/prefetch API or named-vector query, this is the spot to adapt to the installed qdrant-client (1.18) Query API. Confirm with `.venv/Scripts/python.exe -c "import qdrant_client; print(qdrant_client.__version__)"` and the `qdrant_client.models` names (`Prefetch`, `FusionQuery`, `Fusion`, `SparseVector`, `SparseVectorParams`). Do not weaken the test — fix the call.

- [ ] **Step 8: Commit** `git add backend/app/rag/qdrant_store.py backend/app/rag/retriever.py backend/tests/test_qdrant_hybrid.py backend/tests/test_retriever_rerank.py && git commit -m "feat: hybrid qdrant store + retriever candidate-pool/rerank/fusion flow"`

---

### Task 7: Ingest variants (build_index)

**Files:** Modify `backend/app/rag/ingest.py`; Test `backend/tests/test_ingest_variant.py`

- [ ] **Step 1: Write failing test**
```python
from qdrant_client import QdrantClient

from app.config import Settings
from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.ingest import IngestVariant, build_index
from app.rag.sparse import FakeSparseProvider


def _sources():
    return [{"url": "https://x/a", "title": "A"}, {"url": "https://x/b", "title": "B"}]


def _fetch(url):
    return "word " * 400


def test_build_dense_index():
    client = QdrantClient(location=":memory:")
    coll, total = build_index(Settings(_env_file=None), IngestVariant(chunk_size=500, overlap=50, hybrid=False),
                              _sources(), client=client, dense_embedder=FakeEmbeddingProvider(dim=8), fetch=_fetch)
    assert coll == "ai_coach_cs500_dense" and total > 0
    assert client.count(collection_name=coll).count == total


def test_build_hybrid_index():
    client = QdrantClient(location=":memory:")
    coll, total = build_index(Settings(_env_file=None), IngestVariant(chunk_size=1000, overlap=100, hybrid=True),
                              _sources(), client=client, dense_embedder=FakeEmbeddingProvider(dim=8),
                              sparse_embedder=FakeSparseProvider(), fetch=_fetch)
    assert coll == "ai_coach_cs1000_hybrid" and total > 0
    assert client.count(collection_name=coll).count == total
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — modify `backend/app/rag/ingest.py`. Add imports + `IngestVariant` + `_ingest_sources` helper + `build_index`, and refactor `run_ingest` to reuse the helper. New/changed top of file and functions:
```python
from pydantic import BaseModel

from app.rag.qdrant_store import collection_name, ensure_collection, upsert_chunks
from app.rag.sparse import SparseProvider, get_sparse_provider
# (keep existing imports: logging, re, Path, Callable, yaml, QdrantClient, Settings/default_settings,
#  EmbeddingProvider/get_embedding_provider, chunk_document, fetch_url)


class IngestVariant(BaseModel):
    chunk_size: int = 1000
    overlap: int = 150
    hybrid: bool = False


def _ingest_sources(
    client, collection, sources, chunk_size, overlap, dense_embedder,
    sparse_embedder=None, fetch=fetch_url,
) -> int:
    total = 0
    for src in sources:
        url, title = src["url"], src.get("title", src["url"])
        try:
            text = fetch(url)
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort ingestion
            logger.warning("Skipping (fetch error): %s (%s)", url, exc)
            continue
        if not text:
            logger.warning("Skipping (no content): %s", url)
            continue
        chunks = chunk_document(text, _slug(url), url, title, chunk_size, overlap)
        if not chunks:
            continue
        dense = dense_embedder.embed_documents([c.text for c in chunks])
        sparse = sparse_embedder.embed_documents([c.text for c in chunks]) if sparse_embedder else None
        upsert_chunks(client, collection, chunks, dense, sparse_vectors=sparse)
        total += len(chunks)
        logger.info("Ingested %d chunks from %s", len(chunks), url)
    return total


def build_index(
    settings, variant: IngestVariant, sources, client=None,
    dense_embedder=None, sparse_embedder: SparseProvider | None = None, fetch=fetch_url,
) -> tuple[str, int]:
    dense_embedder = dense_embedder or get_embedding_provider(settings)
    client = client or QdrantClient(url=settings.qdrant_url)
    coll = collection_name(variant.chunk_size, variant.hybrid)
    ensure_collection(client, coll, dense_embedder.dim, hybrid=variant.hybrid)
    if variant.hybrid and sparse_embedder is None:
        sparse_embedder = get_sparse_provider(settings)
    total = _ingest_sources(
        client, coll, sources, variant.chunk_size, variant.overlap,
        dense_embedder, sparse_embedder if variant.hybrid else None, fetch,
    )
    return coll, total
```
And change the body of the existing `run_ingest(...)` so that after `ensure_collection(client, settings.qdrant_collection, embedder.dim)` it delegates the per-source loop to the helper:
```python
    return _ingest_sources(
        client, settings.qdrant_collection, sources, settings.chunk_size,
        settings.chunk_overlap, embedder, None, fetch,
    )
```
(Removing the old inline loop — the helper has identical behavior, so `test_ingest.py` stays green.)

- [ ] **Step 4: Run → PASS** `... -m pytest tests/test_ingest_variant.py tests/test_ingest.py -v` (both new and original ingest tests).

- [ ] **Step 5: Commit** `git add backend/app/rag/ingest.py backend/tests/test_ingest_variant.py && git commit -m "feat: add ingest variants (build_index) and share the ingest loop"`

---

### Task 8: Staged sweep + comparison report

**Files:** Create `backend/app/evals/sweep.py`, `backend/app/evals/comparison_report.py`; Test `backend/tests/test_sweep.py`

- [ ] **Step 1: Write failing test** `backend/tests/test_sweep.py`
```python
from pathlib import Path

from app.config import Settings
from app.evals.models import GoldenQuestion
from app.evals.sweep import SweepContext, run_staged_sweep
from app.evals.comparison_report import write_comparison_report
from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.reranker import FakeReranker
from app.rag.sparse import FakeSparseProvider
from qdrant_client import QdrantClient


def _questions():
    return [
        GoldenQuestion(id="q1", type="definition", question="alpha?", required_sources=["https://x/a"]),
        GoldenQuestion(id="q2", type="definition", question="beta?", required_sources=["https://x/b"]),
    ]


def _ctx(tmp_path):
    return SweepContext(
        settings=Settings(_env_file=None),
        sources=[{"url": "https://x/a", "title": "A"}, {"url": "https://x/b", "title": "B"}],
        questions=_questions(),
        client=QdrantClient(location=":memory:"),
        dense_embedder=FakeEmbeddingProvider(dim=8),
        sparse_embedder=FakeSparseProvider(),
        reranker=FakeReranker(),
        fetch=lambda url: ("alpha " * 200) if url.endswith("a") else ("beta " * 200),
        chunk_sizes=[500, 1000],
        ks=[5],
    )


def test_staged_sweep_produces_variants_and_winner(tmp_path):
    report = run_staged_sweep(_ctx(tmp_path))
    assert len(report.variants) >= 3  # >=2 chunk sizes + dense/hybrid + rerank stages
    assert report.winner_label
    assert "chunk_size" in report.winner_config
    paths = write_comparison_report(report, tmp_path)
    assert Path(paths["html"]).exists() and Path(paths["json"]).exists() and Path(paths["md"]).exists()
    assert report.winner_label in Path(paths["md"]).read_text(encoding="utf-8")
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** `backend/app/evals/sweep.py`
```python
"""Staged retrieval A/B sweep: tune one lever at a time, lock the winner, proceed."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.evals.models import GoldenQuestion, RetrievalReport, SweepReport
from app.evals.retrieval_eval import evaluate_retrieval
from app.rag.ingest import IngestVariant, build_index
from app.rag.reranker import NoopReranker
from app.rag.retriever import Retriever

PRIMARY = "recall_at_5"


@dataclass
class SweepContext:
    settings: Any
    sources: list[dict]
    questions: list[GoldenQuestion]
    client: Any
    dense_embedder: Any
    sparse_embedder: Any
    reranker: Any                       # real LocalCrossEncoder in prod, FakeReranker in tests
    fetch: Any = None
    chunk_sizes: list[int] = field(default_factory=lambda: [500, 1000, 1500])
    ks: list[int] = field(default_factory=lambda: [5, 10])
    _built: dict = field(default_factory=dict)


def _build(ctx: SweepContext, chunk_size: int, hybrid: bool) -> str:
    key = (chunk_size, hybrid)
    if key not in ctx._built:
        kwargs = {"client": ctx.client, "dense_embedder": ctx.dense_embedder}
        if ctx.fetch is not None:
            kwargs["fetch"] = ctx.fetch
        if hybrid:
            kwargs["sparse_embedder"] = ctx.sparse_embedder
        coll, _ = build_index(ctx.settings, IngestVariant(chunk_size=chunk_size, hybrid=hybrid),
                              ctx.sources, **kwargs)
        ctx._built[key] = coll
    return ctx._built[key]


def _eval(ctx: SweepContext, *, chunk_size: int, hybrid: bool, rerank: bool, k: int) -> RetrievalReport:
    coll = _build(ctx, chunk_size, hybrid)
    retriever = Retriever(
        ctx.client, coll, ctx.dense_embedder, top_k=k,
        reranker=ctx.reranker if rerank else NoopReranker(),
        rerank_candidates=ctx.settings.rerank_candidates,
        hybrid=hybrid, sparse_embedder=ctx.sparse_embedder if hybrid else None,
    )
    cfg = {"chunk_size": chunk_size, "hybrid": hybrid, "rerank": rerank, "k": k}
    label = f"cs{chunk_size}_{'hybrid' if hybrid else 'dense'}_rr{int(rerank)}_k{k}"
    return evaluate_retrieval(ctx.questions, retriever, k=k, label=label, config=cfg)


def run_staged_sweep(ctx: SweepContext) -> SweepReport:
    variants: list[RetrievalReport] = []
    stage_winners: dict = {}
    k0 = ctx.ks[0]

    def best(reports: list[RetrievalReport]) -> RetrievalReport:
        return max(reports, key=lambda r: (r.metrics.get(PRIMARY, 0.0), r.metrics.get("mrr", 0.0)))

    # Stage 1: chunk size (dense, no rerank, k0)
    s1 = [_eval(ctx, chunk_size=cs, hybrid=False, rerank=False, k=k0) for cs in ctx.chunk_sizes]
    variants += s1
    best_cs = best(s1).config["chunk_size"]
    stage_winners["chunk_size"] = best_cs

    # Stage 2: dense vs hybrid at best chunk size
    s2 = [_eval(ctx, chunk_size=best_cs, hybrid=h, rerank=False, k=k0) for h in (False, True)]
    variants += [r for r in s2 if r.label not in {v.label for v in variants}]
    best_hybrid = best(s2).config["hybrid"]
    stage_winners["hybrid"] = best_hybrid

    # Stage 3: reranker off vs on at the winning index
    s3 = [_eval(ctx, chunk_size=best_cs, hybrid=best_hybrid, rerank=rr, k=k0) for rr in (False, True)]
    variants += [r for r in s3 if r.label not in {v.label for v in variants}]
    best_rerank = best(s3).config["rerank"]
    stage_winners["rerank"] = best_rerank

    # Stage 4: top-k (informational only — recall trivially rises with k, so we do NOT
    # pick a "winner" by metric across different k. Record the extra k variants; keep k0
    # as the operating point.)
    s4 = [_eval(ctx, chunk_size=best_cs, hybrid=best_hybrid, rerank=best_rerank, k=k)
          for k in ctx.ks if k != k0]
    variants += [r for r in s4 if r.label not in {v.label for v in variants}]
    best_k = k0

    winner_cfg = {"chunk_size": best_cs, "hybrid": best_hybrid, "rerank": best_rerank, "k": best_k}
    winner_label = f"cs{best_cs}_{'hybrid' if best_hybrid else 'dense'}_rr{int(best_rerank)}_k{best_k}"
    return SweepReport(
        run_id=uuid.uuid4().hex[:12], ts=datetime.now(timezone.utc).isoformat(),
        variants=variants, stage_winners=stage_winners,
        winner_label=winner_label, winner_config=winner_cfg,
    )
```

- [ ] **Step 4: Implement** `backend/app/evals/comparison_report.py`
```python
"""Side-by-side comparison report for a retrieval sweep (HTML + Markdown + JSON)."""

import html
from pathlib import Path

from app.evals.models import SweepReport

_METRIC_COLS = ["recall_at_5", "precision_at_5", "hit_rate_at_5", "mrr"]


def _rows(report: SweepReport):
    for v in report.variants:
        yield v.label, v.config, [v.metrics.get(m, v.metrics.get(m.replace("_5", "_10"), 0.0)) for m in _METRIC_COLS]


def _markdown(report: SweepReport) -> str:
    lines = [f"# Retrieval Sweep {report.run_id}", "", f"- ts: {report.ts}",
             f"- **winner: `{report.winner_label}`** {report.winner_config}",
             f"- stage winners: {report.stage_winners}", "",
             "| variant | " + " | ".join(_METRIC_COLS) + " | n |", "|" + "---|" * (len(_METRIC_COLS) + 2)]
    for label, _cfg, vals in _rows(report):
        mark = " **<-- winner**" if label == report.winner_label else ""
        vstr = " | ".join(f"{v:.4f}" for v in vals)
        n = next((x.n_questions for x in report.variants if x.label == label), 0)
        lines.append(f"| `{label}`{mark} | {vstr} | {n} |")
    return "\n".join(lines) + "\n"


def _html(report: SweepReport) -> str:
    head = ("<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:6px 10px}tr.win{background:#e6ffe6;font-weight:600}"
            "code{background:#f4f4f4;padding:1px 4px}</style>")
    rows = []
    for label, _cfg, vals in _rows(report):
        cls = " class='win'" if label == report.winner_label else ""
        tds = "".join(f"<td>{v:.4f}</td>" for v in vals)
        rows.append(f"<tr{cls}><td><code>{html.escape(label)}</code></td>{tds}</tr>")
    th = "".join(f"<th>{m}</th>" for m in _METRIC_COLS)
    return (f"<html><head>{head}</head><body>"
            f"<h1>Retrieval Sweep {report.run_id}</h1>"
            f"<p>Winner: <code>{html.escape(report.winner_label)}</code> — {html.escape(str(report.winner_config))}</p>"
            f"<table><tr><th>variant</th>{th}</tr>{''.join(rows)}</table></body></html>")


def write_comparison_report(report: SweepReport, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"sweep-{report.run_id}"
    base.with_suffix(".json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    base.with_suffix(".html").write_text(_html(report), encoding="utf-8")
    return {"json": str(base.with_suffix(".json")), "md": str(base.with_suffix(".md")), "html": str(base.with_suffix(".html"))}
```

- [ ] **Step 5: Run → PASS** `... -m pytest tests/test_sweep.py -v`.

- [ ] **Step 6: Full suite + lint** `... -m pytest -q` (all pass; slow/eval deselected) and `... -m ruff check app tests`.

- [ ] **Step 7: Commit** `git add backend/app/evals/sweep.py backend/app/evals/comparison_report.py backend/tests/test_sweep.py && git commit -m "feat: add staged retrieval sweep and comparison report"`

---

### Task 9: Sweep CLI + run the real sweep (controller)

**Files:** Create `backend/scripts/run_sweep.py`

- [ ] **Step 1: Create `backend/scripts/run_sweep.py`**
```python
"""Run the staged retrieval sweep against the real corpus and write a comparison report.

Prereqs: Qdrant running. No OpenAI key needed (retrieval-only). Local models download once.
Run from backend/:  python scripts/run_sweep.py
"""

from pathlib import Path

from qdrant_client import QdrantClient

from app.config import settings
from app.evals.comparison_report import write_comparison_report
from app.evals.dataset import load_dataset
from app.evals.sweep import SweepContext, run_staged_sweep
from app.logging_config import configure_logging
from app.providers.embeddings import get_embedding_provider
from app.rag.ingest import load_sources
from app.rag.reranker import LocalCrossEncoderReranker
from app.rag.sparse import get_sparse_provider


def main() -> None:
    configure_logging()
    root = Path(__file__).resolve().parents[1]
    questions = load_dataset(root / "app" / "evals" / "golden" / "dataset.yaml")
    sources = load_sources(root / "docs" / "sources.yaml")
    ctx = SweepContext(
        settings=settings,
        sources=sources,
        questions=questions,
        client=QdrantClient(url=settings.qdrant_url),
        dense_embedder=get_embedding_provider(settings),
        sparse_embedder=get_sparse_provider(settings),
        reranker=LocalCrossEncoderReranker(settings.rerank_model),
        chunk_sizes=[500, 1000, 1500],
        ks=[5, 10],
    )
    report = run_staged_sweep(ctx)
    paths = write_comparison_report(report, settings.log_dir)
    print("\n=== Retrieval Sweep ===")
    print("winner:", report.winner_label, report.winner_config)
    print("stage winners:", report.stage_winners)
    for v in report.variants:
        print(f"  {v.label}: {v.metrics}")
    print("report:", paths["html"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 (controller-run):** with Qdrant up, run `python scripts/run_sweep.py` (long timeout; builds up to 4 collections + downloads the reranker + BM25 models once). Capture the winner + per-variant metrics + report path. If the fastembed/qdrant ⚠️ VERIFY calls error, fix them in `reranker.py`/`sparse.py`/`retriever.py` and re-run.

- [ ] **Step 3: Commit** `git add backend/scripts/run_sweep.py && git commit -m "feat: add retrieval sweep CLI"` (the generated `sweep-*.{json,html,md}` are gitignored runtime artifacts — confirm `.gitignore` covers `**/reports/eval_runs/sweep-*` via the existing `*.json|*.html` patterns; add `*.md` ignore for `reports/eval_runs/` if needed).

---

### Task 10: Apply the winner, re-measure, update docs (controller)

**Files:** Modify `backend/app/config.py` (winning defaults), `backend/tests/test_eval_benchmark.py` (baseline floors), `README.md`, `PRD.md`; Create a handover.

- [ ] **Step 0 — wire the winner into the live pipeline** (`backend/app/rag/orchestrator.py`, `build_orchestrator`). The current `build_orchestrator` builds `Retriever(client, s.qdrant_collection, embedder, s.top_k)` with no reranker/hybrid. Change it to thread the Phase-3 settings through:
```python
    from app.rag.reranker import get_reranker
    from app.rag.sparse import get_sparse_provider
    reranker = get_reranker(s)
    sparse = get_sparse_provider(s) if s.hybrid_enabled else None
    retriever = Retriever(
        client, s.qdrant_collection, embedder, s.top_k,
        reranker=reranker, rerank_candidates=s.rerank_candidates,
        hybrid=s.hybrid_enabled, sparse_embedder=sparse,
    )
```
With defaults (`rerank_enabled=False`, `hybrid_enabled=False`) this is behaviorally identical to today, so existing tests stay green. Run `pytest -q` to confirm.

- [ ] **Step 1 — point at the winning collection.** Set the winning config as `Settings` defaults (`chunk_size`, `rerank_enabled`, `hybrid_enabled`, `top_k`) AND set `qdrant_collection` to the sweep-built winning collection name (`collection_name(winner.chunk_size, winner.hybrid)`, e.g. `ai_coach_cs500_hybrid`) — that collection already exists in the live Qdrant from the sweep (Task 9), so **no re-ingest is needed**. (Alternative for a fresh environment: `build_index(settings, IngestVariant(chunk_size=winner.chunk_size, hybrid=winner.hybrid), sources)`.)

- [ ] **Step 2:** Re-run the Phase 2 full benchmark: `python scripts/run_evals.py` (needs OpenAI key). Capture before/after vs the prior baseline (refusal_accuracy 0.82, source_recall 0.61, pass_rate 0.62). Update the `BASELINE` floors in `test_eval_benchmark.py` intentionally to the new numbers, and run `pytest -m eval` to confirm it passes.

- [ ] **Step 3:** Update `README.md` (latest eval scores table → new numbers, dated; add the retrieval-sweep workflow `python scripts/run_sweep.py`) and `PRD.md` (mark **Phase 3 ✅ complete** in §9 with the winner + before/after; record retrieval metrics vs targets in §6).

- [ ] **Step 4:** Write `HANDOVERS/2026-05-26_<HHMM>_phase3-retrieval-eval.md` (CLAUDE Rule 2 template): the sweep, the winner with data, the before/after benchmark, known issues (e.g. corpus reproducibility still open; any ⚠️ VERIFY adaptations made), how to verify (`run_sweep.py`, `run_evals.py`, `pytest`/`pytest -m eval`).

- [ ] **Step 5: Commit** the config + baseline + docs together: `git commit -m "eval: apply Phase 3 retrieval winner; before/after benchmark; docs"`.

---

## Final verification (end of plan)
- [ ] From `backend/`: `pytest -q` → all non-slow/non-eval green; `ruff check app tests` clean.
- [ ] `pytest -m slow` (optional, downloads models) → real reranker + BM25 + hybrid pass.
- [ ] `python scripts/run_sweep.py` → comparison report with a winner.
- [ ] `python scripts/run_evals.py` (winner applied) → before/after recorded; `pytest -m eval` green on updated floors.
- [ ] No secrets/reports committed; PRD Phase 3 marked complete with the winner + numbers.
