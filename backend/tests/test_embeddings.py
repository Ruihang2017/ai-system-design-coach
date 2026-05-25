import pytest
from app.providers.embeddings import FakeEmbeddingProvider, get_embedding_provider
from app.config import Settings


def test_fake_provider_dim_and_determinism():
    p = FakeEmbeddingProvider(dim=8)
    assert p.dim == 8
    v1 = p.embed_query("hello")
    v2 = p.embed_query("hello")
    assert v1 == v2 and len(v1) == 8
    assert p.embed_query("hello") != p.embed_query("world")


def test_embed_documents_shape():
    p = FakeEmbeddingProvider(dim=4)
    vecs = p.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3 and all(len(v) == 4 for v in vecs)


def test_factory_returns_fake():
    p = get_embedding_provider(Settings(_env_file=None, embed_provider="fake"))
    assert isinstance(p, FakeEmbeddingProvider)


@pytest.mark.slow
def test_fastembed_real_model():
    from app.providers.embeddings import FastEmbedProvider
    p = FastEmbedProvider("BAAI/bge-small-en-v1.5")
    v = p.embed_query("retrieval augmented generation")
    assert len(v) == p.dim == 384
