"""Qdrant collection management and upserts (dense + optional hybrid sparse)."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, SparseVector, SparseVectorParams, VectorParams,
)

from app.constants import QDRANT_NAMESPACE
from app.rag.models import Chunk
from app.rag.sparse import SparseVec


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(QDRANT_NAMESPACE, chunk_id))


def collection_name(chunk_size: int, hybrid: bool) -> str:
    return f"ai_coach_cs{chunk_size}_{'hybrid' if hybrid else 'dense'}"


def ensure_collection(client: QdrantClient, collection: str, dim: int, hybrid: bool = False) -> None:
    if client.collection_exists(collection):
        return
    if hybrid:
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVec] | None = None,
) -> None:
    points = []
    for i, (c, dv) in enumerate(zip(chunks, dense_vectors)):
        if sparse_vectors is not None:
            sv = sparse_vectors[i]
            vector = {"dense": dv, "sparse": SparseVector(indices=sv.indices, values=sv.values)}
        else:
            vector = dv
        points.append(PointStruct(id=point_id(c.id), vector=vector, payload=c.model_dump()))
    if points:
        client.upsert(collection_name=collection, points=points)
