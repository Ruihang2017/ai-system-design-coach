"""Vector retrieval from Qdrant: dense or hybrid, with optional reranking."""

from typing import Any, Protocol, runtime_checkable

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.providers.embeddings import EmbeddingProvider
from app.rag.models import Chunk, RetrievedChunk
from app.rag.reranker import NoopReranker, Reranker


@runtime_checkable
class RetrieverProtocol(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]: ...


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: EmbeddingProvider,
        top_k: int = 5,
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
        hybrid: bool = False,
        sparse_embedder: Any = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._top_k = top_k
        self._reranker = reranker or NoopReranker()
        self._rerank_candidates = rerank_candidates
        self._hybrid = hybrid
        self._sparse = sparse_embedder

    def _fetch_n(self, k: int) -> int:
        return max(k, self._rerank_candidates) if not isinstance(self._reranker, NoopReranker) else k

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self._top_k
        fetch_n = self._fetch_n(k)
        dense = self._embedder.embed_query(query)

        if self._hybrid:
            sv = self._sparse.embed_query(query)
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    qm.Prefetch(query=dense, using="dense", limit=fetch_n),
                    qm.Prefetch(
                        query=qm.SparseVector(indices=sv.indices, values=sv.values),
                        using="sparse", limit=fetch_n,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=fetch_n,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection, query=dense, limit=fetch_n, with_payload=True,
            )

        candidates = [
            RetrievedChunk(chunk=Chunk(**p.payload), score=p.score, n=i + 1)
            for i, p in enumerate(response.points)
        ]
        return self._reranker.rerank(query, candidates, k)
