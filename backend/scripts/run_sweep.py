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
