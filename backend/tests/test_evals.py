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
        """required_sources=[] -> source_hit is always False (adversarial/refusal)."""
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
        """citation_valid = True when refused (vacuously -- no citation requirement)."""
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
          out2: refusal, correct (should_refuse=True, refused=True)   -> passed
          out3: refusal, wrong  (should_refuse=True, refused=False)   -> passed=False
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
        # q0: should_refuse=False, refused=False -> refusal_correct=True
        # q1: should_refuse=False, refused=False -> refusal_correct=True
        # q2: should_refuse=True,  refused=True  -> refusal_correct=True
        # q3: should_refuse=True,  refused=False -> refusal_correct=False
        # refusal_accuracy = 3/4 = 0.75
        assert m["refusal_accuracy"] == pytest.approx(0.75, abs=1e-4)

    def test_answer_rate(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        # grounded: q0 (not refused), q1 (not refused) -> answer_rate = 2/2 = 1.0
        assert m["answer_rate"] == pytest.approx(1.0, abs=1e-4)

    def test_source_recall_at_k(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        # grounded with required_sources: q0 (hit), q1 (miss) -> 1/2 = 0.5
        assert m["source_recall_at_k"] == pytest.approx(0.5, abs=1e-4)

    def test_pass_rate(self):
        from app.evals.metrics import compute_metrics
        outcomes = self._make_outcomes()
        m = compute_metrics(outcomes)
        # out0=True, out1=False, out2=True, out3=False -> 2/4 = 0.5
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
        # q1: answered, cited, source hit -> pass; q2: refused correctly -> pass
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


def evaluate(questions, orchestrator, settings=None):
    """Local import helper used by TestWriteReport._report() before runner exists."""
    from app.evals.runner import evaluate as _evaluate
    return _evaluate(questions, orchestrator, settings)
