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
