"""Sparse embeddings for hybrid retrieval. Local fastembed BM25/SPLADE; Fake for tests."""

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.config import Settings


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


@runtime_checkable
class SparseProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVec]: ...
    def embed_query(self, text: str) -> SparseVec: ...


class FakeSparseProvider:
    """Deterministic hashed bag-of-words sparse vectors for tests."""

    def __init__(self, dim: int = 10000) -> None:
        self._dim = dim

    def _vec(self, text: str) -> SparseVec:
        counts: dict[int, float] = {}
        for tok in text.lower().split():
            idx = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self._dim
            counts[idx] = counts.get(idx, 0.0) + 1.0
        items = sorted(counts.items())
        return SparseVec(indices=[i for i, _ in items], values=[v for _, v in items])

    def embed_query(self, text: str) -> SparseVec:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._vec(t) for t in texts]


class SparseEmbeddingProvider:
    """fastembed SparseTextEmbedding (default BM25). Downloads on first use."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name)
        self._model_name = model_name

    @staticmethod
    def _to_vec(emb) -> SparseVec:
        return SparseVec(indices=[int(i) for i in emb.indices], values=[float(v) for v in emb.values])

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._to_vec(e) for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVec:
        embed_fn = getattr(self._model, "query_embed", self._model.embed)
        return self._to_vec(next(iter(embed_fn([text]))))


def get_sparse_provider(settings: Settings) -> SparseProvider:
    if settings.sparse_model == "fake":
        return FakeSparseProvider()
    return SparseEmbeddingProvider(settings.sparse_model)
