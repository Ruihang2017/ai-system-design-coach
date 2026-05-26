import pytest
from app.config import Settings
from app.rag.sparse import SparseVec, FakeSparseProvider, get_sparse_provider


def test_fake_sparse_is_deterministic_and_shaped():
    p = FakeSparseProvider()
    v1 = p.embed_query("hybrid search retrieval")
    v2 = p.embed_query("hybrid search retrieval")
    assert isinstance(v1, SparseVec)
    assert v1.indices == v2.indices and v1.values == v2.values
    assert len(v1.indices) == len(v1.values) and len(v1.indices) > 0
    docs = p.embed_documents(["a b", "c d e"])
    assert len(docs) == 2 and all(isinstance(d, SparseVec) for d in docs)


def test_factory_returns_fake():
    assert isinstance(get_sparse_provider(Settings(_env_file=None, sparse_model="fake")), FakeSparseProvider)


@pytest.mark.slow
def test_real_bm25_sparse():
    from app.rag.sparse import SparseEmbeddingProvider
    p = SparseEmbeddingProvider("Qdrant/bm25")
    v = p.embed_query("retrieval augmented generation")
    assert len(v.indices) == len(v.values) and len(v.indices) > 0
