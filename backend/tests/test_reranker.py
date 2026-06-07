import pytest
from app.config import Settings
from app.rag.models import Chunk, RetrievedChunk
from app.rag.reranker import NoopReranker, FakeReranker, get_reranker


def _chunks(texts):
    return [
        RetrievedChunk(chunk=Chunk(id=f"d::{i}", text=t, source_url="u", title="T", doc_id="d", chunk_index=i), score=1.0 - i * 0.1, n=i + 1)
        for i, t in enumerate(texts)
    ]


def test_noop_returns_top_k_renumbered():
    out = NoopReranker().rerank("q", _chunks(["a", "b", "c"]), top_k=2)
    assert [c.chunk.text for c in out] == ["a", "b"]
    assert [c.n for c in out] == [1, 2]


def test_fake_reranker_reorders_by_text_length():
    out = FakeReranker().rerank("q", _chunks(["aa", "bbbb", "c"]), top_k=2)
    assert [c.chunk.text for c in out] == ["bbbb", "aa"]
    assert [c.n for c in out] == [1, 2]


def test_factory_returns_noop_when_disabled():
    assert isinstance(get_reranker(Settings(_env_file=None, rerank_enabled=False)), NoopReranker)


@pytest.mark.slow
def test_local_cross_encoder_reranks():
    from app.rag.reranker import LocalCrossEncoderReranker
    r = LocalCrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    out = r.rerank("what is hybrid search?", _chunks([
        "Bananas are yellow fruit.",
        "Hybrid search combines dense and sparse retrieval.",
    ]), top_k=1)
    assert out[0].chunk.text.startswith("Hybrid search")
