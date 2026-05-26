"""Cross-encoder rerankers. Local fastembed model; Noop/Fake for the no-rerank path & tests."""

from typing import Protocol, runtime_checkable

from app.config import Settings
from app.rag.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


def _renumber(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [c.model_copy(update={"n": i + 1}) for i, c in enumerate(chunks)]


class NoopReranker:
    """No reranking: keep retrieval order, take top_k, renumber n."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return _renumber(chunks[:top_k])


class FakeReranker:
    """Deterministic test double: rank by descending text length."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ranked = sorted(chunks, key=lambda c: len(c.chunk.text), reverse=True)[:top_k]
        return _renumber(ranked)


class LocalCrossEncoderReranker:
    """fastembed TextCrossEncoder. Downloads the model on first use."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = list(self._model.rerank(query, [c.chunk.text for c in chunks]))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
        ranked = [chunks[i].model_copy(update={"score": float(scores[i])}) for i in order]
        return _renumber(ranked)


def get_reranker(settings: Settings) -> Reranker:
    if settings.rerank_enabled:
        return LocalCrossEncoderReranker(settings.rerank_model)
    return NoopReranker()
