"""Ingestion: fetch -> extract -> chunk -> embed -> upsert into Qdrant."""

import logging
import re
from pathlib import Path
from typing import Callable

import yaml
from qdrant_client import QdrantClient

from app.config import Settings, settings as default_settings
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.chunker import chunk_document
from app.rag.fetcher import fetch_url
from app.rag.qdrant_store import ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)


def _slug(url: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")


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

    total = 0
    for src in sources:
        url, title = src["url"], src.get("title", src["url"])
        text = fetch(url)
        if not text:
            logger.warning("Skipping (no content): %s", url)
            continue
        chunks = chunk_document(text, _slug(url), url, title, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue
        vectors = embedder.embed_documents([c.text for c in chunks])
        upsert_chunks(client, settings.qdrant_collection, chunks, vectors)
        total += len(chunks)
        logger.info("Ingested %d chunks from %s", len(chunks), url)
    return total


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
