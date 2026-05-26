from qdrant_client import QdrantClient

from app.config import Settings
from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.ingest import IngestVariant, build_index
from app.rag.sparse import FakeSparseProvider


def _sources():
    return [{"url": "https://x/a", "title": "A"}, {"url": "https://x/b", "title": "B"}]


def _fetch(url):
    return "word " * 400


def test_build_dense_index():
    client = QdrantClient(location=":memory:")
    coll, total = build_index(Settings(_env_file=None), IngestVariant(chunk_size=500, overlap=50, hybrid=False),
                              _sources(), client=client, dense_embedder=FakeEmbeddingProvider(dim=8), fetch=_fetch)
    assert coll == "ai_coach_cs500_dense" and total > 0
    assert client.count(collection_name=coll).count == total


def test_build_hybrid_index():
    client = QdrantClient(location=":memory:")
    coll, total = build_index(Settings(_env_file=None), IngestVariant(chunk_size=1000, overlap=100, hybrid=True),
                              _sources(), client=client, dense_embedder=FakeEmbeddingProvider(dim=8),
                              sparse_embedder=FakeSparseProvider(), fetch=_fetch)
    assert coll == "ai_coach_cs1000_hybrid" and total > 0
    assert client.count(collection_name=coll).count == total
