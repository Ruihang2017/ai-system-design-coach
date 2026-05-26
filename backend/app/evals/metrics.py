"""Pure metric functions for the Phase 2 deterministic eval harness."""

from app.evals.models import GoldenQuestion, QuestionOutcome
from app.rag.models import AnswerResult


def build_outcome(q: GoldenQuestion, result: AnswerResult) -> QuestionOutcome:
    """Map a GoldenQuestion + AnswerResult -> QuestionOutcome."""
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

    source_hit = bool(q.required_sources) and any(
        src in retrieved_sources for src in q.required_sources
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
