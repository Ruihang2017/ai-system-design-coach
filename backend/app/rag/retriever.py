"""Vector retrieval from Qdrant."""

from qdrant_client import QdrantClient

from app.providers.embeddings import EmbeddingProvider
from app.rag.models import Chunk, RetrievedChunk


class Retriever:
    def __init__(
        self, client: QdrantClient, collection: str, embedder: EmbeddingProvider, top_k: int = 5
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self._top_k
        vector = self._embedder.embed_query(query)
        response = self._client.query_points(
            collection_name=self._collection, query=vector, limit=k, with_payload=True
        )
        return [
            RetrievedChunk(chunk=Chunk(**point.payload), score=point.score, n=i + 1)
            for i, point in enumerate(response.points)
        ]
