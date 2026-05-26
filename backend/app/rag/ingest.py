"""Ingestion: fetch -> extract -> chunk -> embed -> upsert into Qdrant."""

import logging
import re
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.config import Settings, settings as default_settings
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.chunker import chunk_document
from app.rag.fetcher import fetch_url
from app.rag.qdrant_store import collection_name, ensure_collection, upsert_chunks
from app.rag.sparse import SparseProvider, get_sparse_provider

logger = logging.getLogger(__name__)


def _slug(url: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")


class IngestVariant(BaseModel):
    chunk_size: int = 1000
    overlap: int = 150
    hybrid: bool = False


def _ingest_sources(
    client,
    collection,
    sources,
    chunk_size,
    overlap,
    dense_embedder,
    sparse_embedder=None,
    fetch=fetch_url,
) -> int:
    total = 0
    for src in sources:
        url, title = src["url"], src.get("title", src["url"])
        try:
            text = fetch(url)
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort ingestion
            logger.warning("Skipping (fetch error): %s (%s)", url, exc)
            continue
        if not text:
            logger.warning("Skipping (no content): %s", url)
            continue
        chunks = chunk_document(text, _slug(url), url, title, chunk_size, overlap)
        if not chunks:
            continue
        dense = dense_embedder.embed_documents([c.text for c in chunks])
        sparse = sparse_embedder.embed_documents([c.text for c in chunks]) if sparse_embedder else None
        upsert_chunks(client, collection, chunks, dense, sparse_vectors=sparse)
        total += len(chunks)
        logger.info("Ingested %d chunks from %s", len(chunks), url)
    return total


def run_ingest(
    settings: Settings,
    sources: list[dict],
    client: QdrantClient | None = None,
    embedder: EmbeddingProvider | None = None,
    fetch: Callable[[str], str | None] = fetch_url,
) -> int:
    embedder = embedder or get_embedding_provider(settings)
    client = client or QdrantClient(url=settings.qdrant_url)
    ensure_collection(client, settings.qdrant_collection, embedder.dim)
    return _ingest_sources(
        client, settings.qdrant_collection, sources, settings.chunk_size,
        settings.chunk_overlap, embedder, None, fetch,
    )


def build_index(
    settings: Settings,
    variant: IngestVariant,
    sources: list[dict],
    client: QdrantClient | None = None,
    dense_embedder: EmbeddingProvider | None = None,
    sparse_embedder: SparseProvider | None = None,
    fetch: Callable[[str], str | None] = fetch_url,
) -> tuple[str, int]:
    dense_embedder = dense_embedder or get_embedding_provider(settings)
    client = client or QdrantClient(url=settings.qdrant_url)
    coll = collection_name(variant.chunk_size, variant.hybrid)
    ensure_collection(client, coll, dense_embedder.dim, hybrid=variant.hybrid)
    if variant.hybrid and sparse_embedder is None:
        sparse_embedder = get_sparse_provider(settings)
    total = _ingest_sources(
        client, coll, sources, variant.chunk_size, variant.overlap,
        dense_embedder, sparse_embedder if variant.hybrid else None, fetch,
    )
    return coll, total


def load_sources(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["sources"]


def main() -> None:
    from app.logging_config import configure_logging

    configure_logging()
    sources = load_sources(Path(__file__).resolve().parents[2] / "docs" / "sources.yaml")
    total = run_ingest(default_settings, sources)
    logger.info("Done. Total chunks ingested: %d", total)


if __name__ == "__main__":
    main()
