"""Regression-guard benchmark test — runs the full Phase-2 golden eval.

Marked @pytest.mark.eval so it is deselected by default (addopts excludes 'eval').
Run explicitly with:  pytest -m eval -q

Prereqs: Qdrant running + corpus ingested + OPENAI_API_KEY in .env.
"""

from pathlib import Path

import pytest

from app.config import settings
from app.evals.dataset import load_dataset
from app.evals.runner import evaluate
from app.rag.orchestrator import build_orchestrator

# Baseline observed on 2026-05-26 (first Phase-2 run, run_id 99cf20be4f10).
# These are regression FLOORS, not targets.
# Formula: floor = round_down_to_nearest_0.05(observed) - 0.05, clamped >= 0.
# PRD targets: refusal_accuracy >= 0.95 (the pipeline does not meet this yet;
# the low refusal_accuracy is caused by ~48/75 grounded questions being
# over-refused; that is a Phase-3 improvement area).
# Update these intentionally when the pipeline genuinely improves.
BASELINE = {
    "refusal_accuracy": 0.45,    # observed 0.51
    "answer_rate": 0.30,         # observed 0.36
    "source_recall_at_k": 0.55,  # observed 0.6267
    "pass_rate": 0.40,           # observed 0.45
    "citation_validity": 0.95,   # observed 1.0 (trivially perfect; floor at 0.95)
}


@pytest.mark.eval
def test_golden_benchmark_meets_baseline():
    dataset = Path(__file__).resolve().parents[1] / "app" / "evals" / "golden" / "dataset.yaml"
    questions = load_dataset(dataset)
    report = evaluate(questions, build_orchestrator(settings), settings)
    assert report.n_questions == 100
    m = report.metrics
    for key in ("refusal_accuracy", "answer_rate", "source_recall_at_k", "citation_validity", "pass_rate"):
        assert key in m, f"metric '{key}' missing from report"
    # Regression floors (observed baseline minus margin):
    for key, floor in BASELINE.items():
        assert m[key] >= floor, f"{key}={m[key]:.4f} regressed below floor {floor}"
