from qdrant_client import QdrantClient

from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.models import Chunk
from app.rag.qdrant_store import ensure_collection, upsert_chunks
from app.rag.retriever import Retriever


def _chunk(i, text):
    return Chunk(id=f"d::{i}", text=text, source_url="u", title=f"T{i}", doc_id="d", chunk_index=i)


def _seed():
    client = QdrantClient(location=":memory:")
    embedder = FakeEmbeddingProvider(dim=8)
    ensure_collection(client, "c", embedder.dim)
    chunks = [_chunk(0, "alpha document about retrieval"), _chunk(1, "beta document about chunking")]
    upsert_chunks(client, "c", chunks, embedder.embed_documents([c.text for c in chunks]))
    return client, embedder


def test_top_k_ordering_returns_best_match_first():
    client, embedder = _seed()
    r = Retriever(client, "c", embedder, top_k=2)
    results = r.search("alpha document about retrieval")
    assert results[0].chunk.text == "alpha document about retrieval"
    assert results[0].score >= results[1].score
    assert results[0].n == 1 and results[1].n == 2


def test_payload_reconstructs_chunk():
    client, embedder = _seed()
    r = Retriever(client, "c", embedder, top_k=1)
    rc = r.search("beta document about chunking")[0]
    assert rc.chunk.doc_id == "d" and rc.chunk.source_url == "u"
