from qdrant_client import QdrantClient

from app.providers.embeddings import FakeEmbeddingProvider
from app.rag.models import Chunk
from app.rag.qdrant_store import collection_name, ensure_collection, upsert_chunks
from app.rag.retriever import Retriever
from app.rag.sparse import FakeSparseProvider


def test_collection_name():
    assert collection_name(1000, False) == "ai_coach_cs1000_dense"
    assert collection_name(500, True) == "ai_coach_cs500_hybrid"


def test_hybrid_upsert_and_fused_search():
    client = QdrantClient(location=":memory:")
    emb = FakeEmbeddingProvider(dim=8)
    sparse = FakeSparseProvider()
    ensure_collection(client, "h", emb.dim, hybrid=True)
    chunks = [Chunk(id=f"d::{i}", text=t, source_url=f"u{i}", title=f"T{i}", doc_id="d", chunk_index=i)
              for i, t in enumerate(["alpha retrieval", "beta chunking"])]
    upsert_chunks(client, "h", chunks, emb.embed_documents([c.text for c in chunks]),
                  sparse_vectors=sparse.embed_documents([c.text for c in chunks]))
    r = Retriever(client, "h", emb, top_k=2, hybrid=True, sparse_embedder=sparse)
    out = r.search("alpha retrieval")
    assert len(out) >= 1
    assert any(c.chunk.text == "alpha retrieval" for c in out)
