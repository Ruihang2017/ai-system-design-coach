# Phase 2 Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic (no-LLM-judge) scoring harness that loads a golden YAML dataset, runs it through an orchestrator, computes metrics, and writes an HTML+JSON report.

**Architecture:** Five focused modules under `app/evals/`: `models.py` (Pydantic data classes), `dataset.py` (YAML loader), `metrics.py` (pure metric functions), `runner.py` (orchestrates eval loop), `report.py` (writes HTML+JSON). All tested with fakes — no API keys, no Qdrant, no network.

**Tech Stack:** Python 3.10+, Pydantic v2, PyYAML (already a dependency), pytest, ruff.

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `backend/app/evals/models.py` | CREATE | `GoldenQuestion`, `QuestionOutcome`, `EvalReport` Pydantic models |
| `backend/app/evals/dataset.py` | CREATE | `load_dataset(path)` – YAML → `list[GoldenQuestion]` |
| `backend/app/evals/metrics.py` | CREATE | `build_outcome`, `compute_metrics` pure functions |
| `backend/app/evals/runner.py` | CREATE | `evaluate(questions, orchestrator, settings)` → `EvalReport` |
| `backend/app/evals/report.py` | CREATE | `write_report(report, out_dir)` → `{"json": path, "html": path}` |
| `backend/tests/test_evals.py` | CREATE | All unit tests (written first, TDD) |
| `backend/pyproject.toml` | MODIFY | Register `eval` marker, update `addopts` |

---

### Task 1: Update pyproject.toml and write ALL failing tests

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/tests/test_evals.py`

- [ ] **Step 1: Update pyproject.toml markers and addopts**

In `backend/pyproject.toml`, change `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-q -m 'not slow and not eval'"
markers = [
  "slow: requires network, model download, or API keys",
  "eval: runs the full golden benchmark against the real pipeline (needs Qdrant + OpenAI key)",
]
```

- [ ] **Step 2: Write the complete test file**

Create `backend/tests/test_evals.py`:
```python
"""Tests for Phase 2 eval harness (models, dataset, metrics, runner, report).

All tests run with zero API keys, zero network, zero Qdrant — fakes only.
"""

import json
from pathlib import Path

import pytest

from app.rag.models import AnswerResult, Chunk, RetrievedChunk, TokenUsage


# ---------------------------------------------------------------------------
# Helpers — build minimal AnswerResult fixtures
# ---------------------------------------------------------------------------

def _chunk(source_url: str, n: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=f"doc::{n}",
            text="some text",
            source_url=source_url,
            title=f"Title {n}",
            doc_id="doc",
            chunk_index=n,
        ),
        score=0.9,
        n=n,
    )


def _answered(source_urls: list[str], citations: list[int] | None = None) -> AnswerResult:
    if citations is None:
        citations = list(range(1, len(source_urls) + 1))
    return AnswerResult(
        request_id="req-1",
        answer="Some answer [1].",
        citations=citations,
        chunks=[_chunk(url, i + 1) for i, url in enumerate(source_urls)],
        refused=False,
        latency_ms={"total": 123.4},
        usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        cost_usd=0.0005,
        model="gpt-4o-mini",
    )


def _refused() -> AnswerResult:
    return AnswerResult(
        request_id="req-2",
        answer="I cannot answer that.",
        citations=[],
        chunks=[],
        refused=True,
        latency_ms={"total": 50.0},
        usage=TokenUsage(),
        cost_usd=0.0,
        model="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# 1. GoldenQuestion validation
# ---------------------------------------------------------------------------

class TestGoldenQuestion:
    def test_valid_type_accepted(self):
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(id="q1", type="definition", question="What is RAG?")
        assert q.type == "definition"

    def test_invalid_type_raises(self):
        from app.evals.models import GoldenQuestion
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GoldenQuestion(id="q1", type="banana", question="What is RAG?")

    def test_defaults(self):
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(id="q1", type="refusal", question="irrelevant")
        assert q.should_refuse is False
        assert q.required_sources == []
        assert q.expected_points == []


# ---------------------------------------------------------------------------
# 2. load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_loads_yaml_fixture(self, tmp_path: Path):
        from app.evals.dataset import load_dataset
        yaml_content = """
questions:
  - id: q1
    type: definition
    question: What is CAP theorem?
    should_refuse: false
    required_sources:
      - https://example.com/cap
    expected_points:
      - consistency
      - availability
  - id: q2
    type: refusal
    question: How do I bake bread?
    should_refuse: true
"""
        fixture = tmp_path / "golden.yaml"
        fixture.write_text(yaml_content, encoding="utf-8")

        questions = load_dataset(fixture)

        assert len(questions) == 2
        assert questions[0].id == "q1"
        assert questions[0].type == "definition"
        assert questions[0].required_sources == ["https://example.com/cap"]
        assert questions[1].should_refuse is True

    def test_accepts_path_object(self, tmp_path: Path):
        from app.evals.dataset import load_dataset
        yaml_content = "questions:\n  - id: q1\n    type: architecture\n    question: Q?\n"
        f = tmp_path / "g.yaml"
        f.write_text(yaml_content, encoding="utf-8")
        qs = load_dataset(f)
        assert len(qs) == 1

    def test_accepts_str_path(self, tmp_path: Path):
        from app.evals.dataset import load_dataset
        yaml_content = "questions:\n  - id: q1\n    type: architecture\n    question: Q?\n"
        f = tmp_path / "g.yaml"
        f.write_text(yaml_content, encoding="utf-8")
        qs = load_dataset(str(f))
        assert len(qs) == 1


# ---------------------------------------------------------------------------
# 3. build_outcome
# ---------------------------------------------------------------------------

class TestBuildOutcome:
    def test_grounded_answered_cited_source_hit__passed(self):
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(
            id="q1", type="definition", question="What is X?",
            should_refuse=False,
            required_sources=["https://example.com/x"],
        )
        result = _answered(["https://example.com/x"], citations=[1])
        outcome = build_outcome(q, result)
        assert outcome.passed is True
        assert outcome.source_hit is True
        assert outcome.citation_valid is True
        assert outcome.refusal_correct is True
        assert outcome.latency_ms_total == 123.4
        assert outcome.cost_usd == 0.0005

    def test_grounded_but_refused__passed_false(self):
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(
            id="q2", type="definition", question="What is X?",
            should_refuse=False,
            required_sources=["https://example.com/x"],
        )
        outcome = build_outcome(q, _refused())
        assert outcome.passed is False
        assert outcome.refusal_correct is False  # should_refuse=False, refused=True

    def test_grounded_answered_source_not_retrieved__passed_false(self):
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(
            id="q3", type="definition", question="What is X?",
            should_refuse=False,
            required_sources=["https://required.com/doc"],
        )
        result = _answered(["https://other.com/doc"], citations=[1])
        outcome = build_outcome(q, result)
        assert outcome.source_hit is False
        assert outcome.passed is False

    def test_should_refuse_and_refused__passed_true(self):
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(
            id="q4", type="refusal", question="Bake bread?",
            should_refuse=True,
        )
        outcome = build_outcome(q, _refused())
        assert outcome.passed is True
        assert outcome.refusal_correct is True

    def test_should_refuse_but_answered__passed_false(self):
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(
            id="q5", type="refusal", question="Bake bread?",
            should_refuse=True,
        )
        result = _answered(["https://example.com/bread"], citations=[1])
        outcome = build_outcome(q, result)
        assert outcome.passed is False
        assert outcome.refusal_correct is False

    def test_empty_required_sources__source_hit_false(self):
        """required_sources=[] → source_hit is always False (adversarial/refusal)."""
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(
            id="q6", type="adversarial", question="Adversarial?",
            should_refuse=False,
            required_sources=[],
        )
        result = _answered(["https://example.com/x"], citations=[1])
        outcome = build_outcome(q, result)
        assert outcome.source_hit is False

    def test_citation_valid_false_when_answered_no_citations(self):
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(
            id="q7", type="definition", question="What?",
            should_refuse=False,
            required_sources=["https://example.com/x"],
        )
        result = _answered(["https://example.com/x"], citations=[])  # no citations!
        outcome = build_outcome(q, result)
        assert outcome.citation_valid is False

    def test_citation_valid_true_when_refused(self):
        """citation_valid = True when refused (vacuously — no citation requirement)."""
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion
        q = GoldenQuestion(id="q8", type="refusal", question="Q?", should_refuse=True)
        outcome = build_outcome(q, _refused())
        assert outcome.citation_valid is True


# ---------------------------------------------------------------------------
# 4. compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_outcomes(self):
        """
        4 outcomes chosen for clear fractions:
          out0: grounded, passed (answered + cited + source_hit)
          out1: grounded, failed (answered + cited but source_hit False)
          out2: refusal, correct (should_refuse=True, refused=True)   → passed
          out3: refusal, wrong  (should_refuse=True, refused=False)   → passed=False
        """
        from app.evals.metrics import build_outcome
        from app.evals.models import GoldenQuestion

        q0 = GoldenQuestion(id="q0", type="definition", question="Q0",
                            should_refuse=False, required_sources=["https://a.com"])
        q1 = GoldenQuestion(id="q1", type="definition", question="Q1",
                            should_refuse=False, required_sources=["https://required.com"])
        q2 = GoldenQuestion(id="q2", type="refusal", question="Q2", should_refuse=True)
        q3 = GoldenQuestion(id="q3", type="refusal", question="Q3", should_refuse=True)

        out0 = build_outcome(q0, _answered(["https://a.com"], [1]))
        out1 = build_outcome(q1, _answered(["https://other.com"], [1]))
        out2 = build_outcome(q2, _refused())
        out3 = build_outcome(q3, _answered(["https://x.com"], [1]))

        return [out0, out1, out2, out3]

    def test_refusal_accuracy(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        # q2 correct (refused), q3 wrong (should refuse but answered) → 3/4 wrong?
        # q0: should_refuse=False, refused=False → refusal_correct=True
        # q1: should_refuse=False, refused=False → refusal_correct=True
        # q2: should_refuse=True,  refused=True  → refusal_correct=True
        # q3: should_refuse=True,  refused=False → refusal_correct=False
        # refusal_accuracy = 3/4 = 0.75
        assert m["refusal_accuracy"] == pytest.approx(0.75, abs=1e-4)

    def test_answer_rate(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        # grounded: q0 (not refused), q1 (not refused) → answer_rate = 2/2 = 1.0
        assert m["answer_rate"] == pytest.approx(1.0, abs=1e-4)

    def test_source_recall_at_k(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        # grounded with required_sources: q0 (hit), q1 (miss) → 1/2 = 0.5
        assert m["source_recall_at_k"] == pytest.approx(0.5, abs=1e-4)

    def test_pass_rate(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        # out0=True, out1=False, out2=True, out3=False → 2/4 = 0.5
        assert m["pass_rate"] == pytest.approx(0.5, abs=1e-4)

    def test_counts(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        assert m["n_total"] == 4.0
        assert m["n_grounded"] == 2.0
        assert m["n_refusal"] == 2.0

    def test_empty_list_returns_zeros(self):
        from app.evals.metrics import compute_metrics
        m = compute_metrics([])
        assert m["refusal_accuracy"] == 0.0
        assert m["answer_rate"] == 0.0
        assert m["source_recall_at_k"] == 0.0
        assert m["citation_validity"] == 0.0
        assert m["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# 5. evaluate end-to-end with FakeOrchestrator
# ---------------------------------------------------------------------------

class FakeOrchestrator:
    """Returns canned AnswerResults based on query substring matching."""

    def answer(self, query: str) -> AnswerResult:
        if "bake" in query.lower():
            return _refused()
        return _answered(["https://example.com/rag"], citations=[1])


class TestEvaluate:
    def _questions(self):
        from app.evals.models import GoldenQuestion
        return [
            GoldenQuestion(id="q1", type="definition", question="What is RAG?",
                           should_refuse=False, required_sources=["https://example.com/rag"]),
            GoldenQuestion(id="q2", type="refusal", question="How do I bake bread?",
                           should_refuse=True),
        ]

    def test_n_questions(self):
        from app.evals.runner import evaluate
        report = evaluate(self._questions(), FakeOrchestrator())
        assert report.n_questions == 2

    def test_run_id_and_ts_populated(self):
        from app.evals.runner import evaluate
        report = evaluate(self._questions(), FakeOrchestrator())
        assert len(report.run_id) > 0
        assert "T" in report.ts  # ISO format

    def test_metrics_populated(self):
        from app.evals.runner import evaluate
        report = evaluate(self._questions(), FakeOrchestrator())
        assert "pass_rate" in report.metrics
        assert "refusal_accuracy" in report.metrics

    def test_both_questions_pass(self):
        from app.evals.runner import evaluate
        report = evaluate(self._questions(), FakeOrchestrator())
        # q1: answered, cited, source hit → pass; q2: refused correctly → pass
        assert report.metrics["pass_rate"] == pytest.approx(1.0, abs=1e-4)

    def test_with_settings_snapshot(self):
        from app.config import Settings
        from app.evals.runner import evaluate
        s = Settings(_env_file=None, top_k=7, score_threshold=0.25, rewrite_enabled=False,
                     llm_model="gpt-4o-mini", embed_model="BAAI/bge-small-en-v1.5")
        report = evaluate(self._questions(), FakeOrchestrator(), settings=s)
        assert report.config_snapshot.get("top_k") == 7
        assert report.config_snapshot.get("score_threshold") == 0.25

    def test_outcomes_length(self):
        from app.evals.runner import evaluate
        report = evaluate(self._questions(), FakeOrchestrator())
        assert len(report.outcomes) == 2


# ---------------------------------------------------------------------------
# 6. write_report
# ---------------------------------------------------------------------------

class TestWriteReport:
    def _report(self):
        from app.evals.models import EvalReport
        from app.evals.runner import evaluate
        from app.evals.models import GoldenQuestion
        questions = [
            GoldenQuestion(id="q1", type="definition", question="What is RAG?",
                           should_refuse=False, required_sources=["https://example.com/rag"]),
            GoldenQuestion(id="q2", type="refusal", question="Bread?", should_refuse=True),
        ]
        return evaluate(questions, FakeOrchestrator())

    def test_files_created(self, tmp_path: Path):
        from app.evals.report import write_report
        report = self._report()
        paths = write_report(report, tmp_path)
        assert Path(paths["json"]).exists()
        assert Path(paths["html"]).exists()

    def test_json_is_valid_report(self, tmp_path: Path):
        from app.evals.report import write_report
        report = self._report()
        paths = write_report(report, tmp_path)
        data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        assert data["run_id"] == report.run_id
        assert "pass_rate" in data["metrics"]
        assert isinstance(data["outcomes"], list)

    def test_html_contains_run_id(self, tmp_path: Path):
        from app.evals.report import write_report
        report = self._report()
        paths = write_report(report, tmp_path)
        html = Path(paths["html"]).read_text(encoding="utf-8")
        assert report.run_id in html

    def test_html_contains_metric_name(self, tmp_path: Path):
        from app.evals.report import write_report
        report = self._report()
        paths = write_report(report, tmp_path)
        html = Path(paths["html"]).read_text(encoding="utf-8")
        assert "pass_rate" in html

    def test_creates_out_dir_if_missing(self, tmp_path: Path):
        from app.evals.report import write_report
        new_dir = tmp_path / "nested" / "reports"
        report = self._report()
        paths = write_report(report, new_dir)
        assert Path(paths["json"]).exists()

    def test_filenames_include_run_id(self, tmp_path: Path):
        from app.evals.report import write_report
        report = self._report()
        paths = write_report(report, tmp_path)
        assert report.run_id in paths["json"]
        assert report.run_id in paths["html"]
```

- [ ] **Step 3: Run tests to confirm they all fail (modules don't exist yet)**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_evals.py -q 2>&1
```
Expected: many `ModuleNotFoundError` / `ImportError` for `app.evals.models`, etc.

---

### Task 2: Implement `app/evals/models.py`

**Files:**
- Create: `backend/app/evals/models.py`

- [ ] **Step 1: Write the models**

Create `backend/app/evals/models.py`:
```python
"""Pydantic data models for the Phase 2 eval harness."""

from pydantic import BaseModel, Field, field_validator

QUESTION_TYPES = {"definition", "comparison", "architecture", "troubleshooting", "refusal", "multi_hop", "adversarial"}


class GoldenQuestion(BaseModel):
    id: str
    type: str  # must be one of QUESTION_TYPES
    question: str
    should_refuse: bool = False
    required_sources: list[str] = Field(default_factory=list)
    expected_points: list[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        if v not in QUESTION_TYPES:
            raise ValueError(f"type must be one of {QUESTION_TYPES!r}, got {v!r}")
        return v


class QuestionOutcome(BaseModel):
    id: str
    type: str
    question: str
    should_refuse: bool
    refused: bool
    citations: list[int]
    retrieved_sources: list[str]
    required_sources: list[str]
    source_hit: bool
    citation_valid: bool
    refusal_correct: bool
    passed: bool
    latency_ms_total: float
    cost_usd: float


class EvalReport(BaseModel):
    run_id: str
    ts: str
    config_snapshot: dict = Field(default_factory=dict)
    n_questions: int
    metrics: dict[str, float] = Field(default_factory=dict)
    outcomes: list[QuestionOutcome] = Field(default_factory=list)
```

- [ ] **Step 2: Run model tests**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_evals.py::TestGoldenQuestion -q 2>&1
```
Expected: 3 passed.

---

### Task 3: Implement `app/evals/dataset.py`

**Files:**
- Create: `backend/app/evals/dataset.py`

- [ ] **Step 1: Write the loader**

Create `backend/app/evals/dataset.py`:
```python
"""Load a golden evaluation dataset from a YAML file."""

from pathlib import Path

import yaml

from app.evals.models import GoldenQuestion


def load_dataset(path: str | Path) -> list[GoldenQuestion]:
    """Load and validate a YAML golden dataset.

    The YAML file must have shape:
        questions:
          - id: ...
            type: ...
            question: ...
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [GoldenQuestion(**q) for q in data["questions"]]
```

- [ ] **Step 2: Run dataset tests**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_evals.py::TestLoadDataset -q 2>&1
```
Expected: 3 passed.

---

### Task 4: Implement `app/evals/metrics.py`

**Files:**
- Create: `backend/app/evals/metrics.py`

- [ ] **Step 1: Write build_outcome and compute_metrics**

Create `backend/app/evals/metrics.py`:
```python
"""Pure metric functions for the Phase 2 deterministic eval harness."""

from app.evals.models import GoldenQuestion, QuestionOutcome
from app.rag.models import AnswerResult


def build_outcome(q: GoldenQuestion, result: AnswerResult) -> QuestionOutcome:
    """Map a GoldenQuestion + AnswerResult → QuestionOutcome."""
    # Unique list of retrieved source URLs (preserve order)
    seen: set[str] = set()
    retrieved_sources: list[str] = []
    for c in result.chunks:
        url = c.chunk.source_url
        if url not in seen:
            seen.add(url)
            retrieved_sources.append(url)

    refused = result.refused
    citations = result.citations

    source_hit = (
        bool(q.required_sources)
        and any(src in retrieved_sources for src in q.required_sources)
    )
    citation_valid = True if refused else len(citations) > 0
    refusal_correct = q.should_refuse == refused

    if q.should_refuse:
        passed = refused
    else:
        passed = (not refused) and citation_valid and source_hit

    return QuestionOutcome(
        id=q.id,
        type=q.type,
        question=q.question,
        should_refuse=q.should_refuse,
        refused=refused,
        citations=citations,
        retrieved_sources=retrieved_sources,
        required_sources=q.required_sources,
        source_hit=source_hit,
        citation_valid=citation_valid,
        refusal_correct=refusal_correct,
        passed=passed,
        latency_ms_total=result.latency_ms.get("total", 0.0),
        cost_usd=result.cost_usd,
    )


def compute_metrics(outcomes: list[QuestionOutcome]) -> dict[str, float]:
    """Compute deterministic Phase 2 metrics from a list of outcomes."""
    if not outcomes:
        return {
            "refusal_accuracy": 0.0,
            "answer_rate": 0.0,
            "source_recall_at_k": 0.0,
            "citation_validity": 0.0,
            "pass_rate": 0.0,
            "n_total": 0.0,
            "n_grounded": 0.0,
            "n_refusal": 0.0,
        }

    n_total = len(outcomes)
    grounded = [o for o in outcomes if not o.should_refuse]
    refusal_qs = [o for o in outcomes if o.should_refuse]
    non_refused = [o for o in outcomes if not o.refused]
    grounded_with_sources = [o for o in grounded if o.required_sources]

    def _mean(values: list[bool]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    refusal_accuracy = _mean([o.refusal_correct for o in outcomes])
    answer_rate = _mean([not o.refused for o in grounded])
    source_recall_at_k = _mean([o.source_hit for o in grounded_with_sources])
    citation_validity = _mean([len(o.citations) > 0 for o in non_refused])
    pass_rate = _mean([o.passed for o in outcomes])

    return {
        "refusal_accuracy": refusal_accuracy,
        "answer_rate": answer_rate,
        "source_recall_at_k": source_recall_at_k,
        "citation_validity": citation_validity,
        "pass_rate": pass_rate,
        "n_total": float(n_total),
        "n_grounded": float(len(grounded)),
        "n_refusal": float(len(refusal_qs)),
    }
```

- [ ] **Step 2: Run metrics tests**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_evals.py::TestBuildOutcome tests/test_evals.py::TestComputeMetrics -q 2>&1
```
Expected: all passed.

---

### Task 5: Implement `app/evals/runner.py`

**Files:**
- Create: `backend/app/evals/runner.py`

- [ ] **Step 1: Write the runner**

Create `backend/app/evals/runner.py`:
```python
"""Phase 2 eval runner — drives any orchestrator through a golden dataset."""

import uuid
from datetime import datetime, timezone
from typing import Any

from app.evals.metrics import build_outcome, compute_metrics
from app.evals.models import EvalReport, GoldenQuestion


def evaluate(
    questions: list[GoldenQuestion],
    orchestrator: Any,
    settings: Any = None,
) -> EvalReport:
    """Run every question through orchestrator.answer and compute metrics.

    Args:
        questions: List of GoldenQuestion from load_dataset.
        orchestrator: Any object with .answer(query: str) -> AnswerResult.
        settings: Optional Settings instance; used to populate config_snapshot.

    Returns:
        A fully populated EvalReport.
    """
    run_id = uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc).isoformat()

    config_snapshot: dict = {}
    if settings is not None:
        config_snapshot = {
            "top_k": settings.top_k,
            "embed_model": settings.embed_model,
            "llm_model": settings.llm_model,
            "score_threshold": settings.score_threshold,
            "rewrite_enabled": settings.rewrite_enabled,
        }

    outcomes = []
    for q in questions:
        result = orchestrator.answer(q.question)
        outcomes.append(build_outcome(q, result))

    metrics = compute_metrics(outcomes)

    return EvalReport(
        run_id=run_id,
        ts=ts,
        config_snapshot=config_snapshot,
        n_questions=len(outcomes),
        metrics=metrics,
        outcomes=outcomes,
    )
```

- [ ] **Step 2: Run runner tests**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_evals.py::TestEvaluate -q 2>&1
```
Expected: 6 passed.

---

### Task 6: Implement `app/evals/report.py`

**Files:**
- Create: `backend/app/evals/report.py`

- [ ] **Step 1: Write the report writer**

Create `backend/app/evals/report.py`:
```python
"""Write HTML and JSON eval reports from an EvalReport."""

from pathlib import Path

from app.evals.models import EvalReport, QuestionOutcome


def _render_html(report: EvalReport) -> str:
    def fmt_metric(k: str, v: float) -> str:
        if k.startswith("n_"):
            return str(int(v))
        return f"{v * 100:.1f}%"

    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{fmt_metric(k, v)}</td></tr>"
        for k, v in sorted(report.metrics.items())
    )

    def outcome_row(o: QuestionOutcome) -> str:
        row_class = "pass" if o.passed else "fail"
        truncated_q = o.question[:80] + "…" if len(o.question) > 80 else o.question
        citations_str = ", ".join(str(c) for c in o.citations) or "—"
        return (
            f'<tr class="{row_class}">'
            f"<td>{o.id}</td>"
            f"<td>{o.type}</td>"
            f"<td>{truncated_q}</td>"
            f"<td>{'yes' if o.should_refuse else 'no'}</td>"
            f"<td>{'yes' if o.refused else 'no'}</td>"
            f"<td>{'PASS' if o.passed else 'FAIL'}</td>"
            f"<td>{citations_str}</td>"
            f"<td>{'yes' if o.source_hit else 'no'}</td>"
            f"</tr>"
        )

    outcome_rows = "".join(outcome_row(o) for o in report.outcomes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Report {report.run_id}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  p.meta {{ color: #555; font-size: 0.9rem; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.7rem; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr.pass td {{ background: #e6f4ea; }}
  tr.fail td {{ background: #fce8e6; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Eval Report</h1>
<p class="meta"><strong>run_id:</strong> {report.run_id} &nbsp;|&nbsp; <strong>ts:</strong> {report.ts} &nbsp;|&nbsp; <strong>questions:</strong> {report.n_questions}</p>

<h2>Metrics</h2>
<table>
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>{metric_rows}</tbody>
</table>

<h2>Question Outcomes</h2>
<table>
  <thead>
    <tr>
      <th>ID</th><th>Type</th><th>Question</th>
      <th>Should Refuse</th><th>Refused</th><th>Passed</th>
      <th>Citations</th><th>Source Hit</th>
    </tr>
  </thead>
  <tbody>{outcome_rows}</tbody>
</table>
</body>
</html>"""


def write_report(report: EvalReport, out_dir: str | Path) -> dict[str, str]:
    """Write JSON and HTML reports to out_dir.

    Returns:
        dict with keys "json" and "html" pointing to the written file paths (str).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"eval-{report.run_id}"
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")

    return {"json": str(json_path), "html": str(html_path)}
```

- [ ] **Step 2: Run report tests**

```
cd backend && .venv/Scripts/python.exe -m pytest tests/test_evals.py::TestWriteReport -q 2>&1
```
Expected: 6 passed.

---

### Task 7: Full test suite + ruff + commit

- [ ] **Step 1: Run all tests**

```
cd backend && .venv/Scripts/python.exe -m pytest -q 2>&1
```
Expected: 44 prior + new test_evals tests, 1 deselected (slow). 0 errors.

- [ ] **Step 2: Run ruff**

```
cd backend && .venv/Scripts/python.exe -m ruff check app tests 2>&1
```
Expected: clean (no output).

- [ ] **Step 3: Commit**

```bash
git add backend/app/evals/models.py backend/app/evals/dataset.py backend/app/evals/metrics.py backend/app/evals/runner.py backend/app/evals/report.py backend/tests/test_evals.py backend/pyproject.toml
git commit -m "feat: add Phase 2 eval harness (golden dataset loader, deterministic metrics, runner, HTML report)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```
