"""Qdrant collection management and upserts."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.constants import QDRANT_NAMESPACE
from app.rag.models import Chunk


def point_id(chunk_id: str) -> str:
    """Stable UUID point id derived from the chunk's string id."""
    return str(uuid.uuid5(QDRANT_NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient, collection: str, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    points = [
        PointStruct(id=point_id(c.id), vector=vec, payload=c.model_dump())
        for c, vec in zip(chunks, vectors)
    ]
    if points:
        client.upsert(collection_name=collection, points=points)
