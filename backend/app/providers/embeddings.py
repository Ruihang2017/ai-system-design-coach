"""Embedding providers. Default is local fastembed; Fake is for tests."""

import hashlib
from typing import Protocol, runtime_checkable

from app.config import Settings


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FakeEmbeddingProvider:
    """Deterministic hash-based embeddings for fast, network-free tests."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [h[i % len(h)] / 255.0 for i in range(self._dim)]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class FastEmbedProvider:
    """Local ONNX embeddings via fastembed. Downloads the model on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dim = len(self.embed_query("dimension probe"))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(iter(self._model.embed([text])))))


class OpenAIEmbeddingProvider:
    """OpenAI embeddings (Phase-3 A/B swap)."""

    _DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}

    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._dim = self._DIMS.get(model, 3072)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embed_provider == "fake":
        return FakeEmbeddingProvider()
    if settings.embed_provider == "openai":
        return OpenAIEmbeddingProvider(settings.embed_model, settings.openai_api_key)
    return FastEmbedProvider(settings.embed_model)
