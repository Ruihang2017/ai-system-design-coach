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
