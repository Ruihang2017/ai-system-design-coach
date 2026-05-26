"""Phase 2 eval runner -- drives any orchestrator through a golden dataset."""

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
