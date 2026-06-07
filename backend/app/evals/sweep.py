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
    reranker: Any
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

    s1 = [_eval(ctx, chunk_size=cs, hybrid=False, rerank=False, k=k0) for cs in ctx.chunk_sizes]
    variants += s1
    best_cs = best(s1).config["chunk_size"]
    stage_winners["chunk_size"] = best_cs

    s2 = [_eval(ctx, chunk_size=best_cs, hybrid=h, rerank=False, k=k0) for h in (False, True)]
    variants += [r for r in s2 if r.label not in {v.label for v in variants}]
    best_hybrid = best(s2).config["hybrid"]
    stage_winners["hybrid"] = best_hybrid

    s3 = [_eval(ctx, chunk_size=best_cs, hybrid=best_hybrid, rerank=rr, k=k0) for rr in (False, True)]
    variants += [r for r in s3 if r.label not in {v.label for v in variants}]
    best_rerank = best(s3).config["rerank"]
    stage_winners["rerank"] = best_rerank

    # Stage 4: top-k informational (recall trivially rises with k -> don't pick by metric).
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
