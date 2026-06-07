from pathlib import Path

from qdrant_client import QdrantClient

from app.config import Settings
from app.evals.comparison_report import write_comparison_report
from app.evals.models import GoldenQuestion
from app.evals.sweep import SweepContext, run_staged_sweep
from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.reranker import FakeReranker
from app.rag.sparse import FakeSparseProvider


def _questions():
    return [
        GoldenQuestion(id="q1", type="definition", question="alpha?", required_sources=["https://x/a"]),
        GoldenQuestion(id="q2", type="definition", question="beta?", required_sources=["https://x/b"]),
    ]


def _ctx():
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
    report = run_staged_sweep(_ctx())
    assert len(report.variants) >= 3
    assert report.winner_label
    assert "chunk_size" in report.winner_config
    paths = write_comparison_report(report, tmp_path)
    assert Path(paths["html"]).exists() and Path(paths["json"]).exists() and Path(paths["md"]).exists()
    assert report.winner_label in Path(paths["md"]).read_text(encoding="utf-8")
