from qdrant_client import QdrantClient

from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.models import Chunk
from app.rag.qdrant_store import ensure_collection, upsert_chunks
from app.rag.reranker import FakeReranker
from app.rag.retriever import Retriever


def _seed(n=5):
    client = QdrantClient(location=":memory:")
    emb = FakeEmbeddingProvider(dim=8)
    ensure_collection(client, "c", emb.dim)
    chunks = [Chunk(id=f"d::{i}", text="x" * (i + 1), source_url=f"u{i}", title=f"T{i}", doc_id="d", chunk_index=i) for i in range(n)]
    upsert_chunks(client, "c", chunks, emb.embed_documents([c.text for c in chunks]))
    return client, emb


def test_dense_retriever_without_reranker_unchanged():
    client, emb = _seed()
    r = Retriever(client, "c", emb, top_k=3)  # NoopReranker by default
    out = r.search("xxx")
    assert len(out) == 3 and out[0].n == 1


def test_reranker_reorders_candidates():
    client, emb = _seed(n=5)
    r = Retriever(client, "c", emb, top_k=2, reranker=FakeReranker(), rerank_candidates=5)
    out = r.search("anything")
    assert out[0].chunk.text == "xxxxx" and out[0].n == 1  # FakeReranker = longest text first
