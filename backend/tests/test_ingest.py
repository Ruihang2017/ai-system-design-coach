from qdrant_client import QdrantClient

from app.config import Settings
from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.ingest import run_ingest


def test_run_ingest_populates_collection():
    client = QdrantClient(location=":memory:")
    embedder = FakeEmbeddingProvider(dim=8)
    sources = [{"url": "https://x/doc1", "title": "Doc1"}, {"url": "https://x/doc2", "title": "Doc2"}]

    def fake_fetch(url: str):
        return "word " * 600  # ~3000 chars -> multiple chunks

    s = Settings(_env_file=None, qdrant_collection="c", chunk_size=1000, chunk_overlap=100)
    count = run_ingest(s, sources, client=client, embedder=embedder, fetch=fake_fetch)
    assert count > 0
    assert client.count(collection_name="c").count == count


def test_run_ingest_skips_failed_fetches():
    client = QdrantClient(location=":memory:")
    embedder = FakeEmbeddingProvider(dim=8)
    sources = [{"url": "https://x/ok", "title": "Ok"}, {"url": "https://x/bad", "title": "Bad"}]

    def fetch(url: str):
        return None if url.endswith("bad") else "word " * 600

    s = Settings(_env_file=None, qdrant_collection="c2")
    count = run_ingest(s, sources, client=client, embedder=embedder, fetch=fetch)
    assert count > 0  # only the good doc contributed
