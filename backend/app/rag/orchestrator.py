"""Composes the RAG pipeline and produces a logged AnswerResult."""

import logging
import time
import uuid
from datetime import datetime, timezone

from app.config import Settings, settings as default_settings
from app.evals.request_logger import RequestLogger
from app.constants import REFUSAL_MESSAGE
from app.rag.citation_checker import check_and_enforce
from app.rag.generator import Generator
from app.rag.models import AnswerResult, RequestLog, add_usage, compute_cost
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import RetrieverProtocol

logger = logging.getLogger(__name__)

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - langsmith is a dependency
    def traceable(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RAGOrchestrator:
    def __init__(self, rewriter: QueryRewriter, retriever: RetrieverProtocol, generator: Generator,
                 request_logger: RequestLogger, settings: Settings) -> None:
        self._rewriter = rewriter
        self._retriever = retriever
        self._generator = generator
        self._logger = request_logger
        self._settings = settings

    @traceable(name="rag.answer")
    def answer(self, query: str) -> AnswerResult:
        request_id = str(uuid.uuid4())
        latency: dict[str, float] = {}
        overall = time.perf_counter()

        t = time.perf_counter()
        rewritten, rw_usage = self._rewriter.rewrite(query)
        latency["rewrite"] = _ms(t)

        t = time.perf_counter()
        chunks = self._retriever.search(rewritten)
        latency["retrieve"] = _ms(t)

        top_score = max((c.score for c in chunks), default=0.0)
        if not chunks or top_score < self._settings.score_threshold:
            answer_text, citations, refused, usage = REFUSAL_MESSAGE, [], True, rw_usage
            latency["generate"] = 0.0
        else:
            t = time.perf_counter()
            raw, gen_usage = self._generator.generate(rewritten, chunks)
            latency["generate"] = _ms(t)
            answer_text, citations, refused = check_and_enforce(raw, len(chunks))
            usage = add_usage(rw_usage, gen_usage)

        latency["total"] = _ms(overall)
        cost = compute_cost(self._settings.llm_model, usage)

        result = AnswerResult(
            request_id=request_id, answer=answer_text, citations=citations, chunks=chunks,
            refused=refused, latency_ms=latency, usage=usage, cost_usd=cost,
            model=self._settings.llm_model,
        )
        self._log(query, rewritten, result)
        return result

    def _log(self, query: str, rewritten: str, result: AnswerResult) -> None:
        record = RequestLog(
            request_id=result.request_id,
            ts=datetime.now(timezone.utc).isoformat(),
            query=query, rewritten_query=rewritten,
            retrieved=[{"chunk_id": c.chunk.id, "score": c.score, "source_url": c.chunk.source_url, "n": c.n} for c in result.chunks],
            answer=result.answer, citations=result.citations, refused=result.refused,
            latency_ms=result.latency_ms, usage=result.usage, cost_usd=result.cost_usd,
            model=result.model,
            config_snapshot={
                "top_k": self._settings.top_k, "chunk_size": self._settings.chunk_size,
                "embed_model": self._settings.embed_model, "score_threshold": self._settings.score_threshold,
                "rerank": False, "rewrite_enabled": self._settings.rewrite_enabled,
            },
        )
        self._logger.log(record)


def build_orchestrator(settings: Settings | None = None) -> RAGOrchestrator:
    """Wire the real pipeline (Qdrant + configured providers). Requires running Qdrant."""
    from pathlib import Path

    from qdrant_client import QdrantClient

    from app.providers.embeddings import get_embedding_provider
    from app.providers.llm import get_llm_provider
    from app.rag.retriever import Retriever

    s = settings or default_settings
    embedder = get_embedding_provider(s)
    client = QdrantClient(url=s.qdrant_url)
    retriever = Retriever(client, s.qdrant_collection, embedder, s.top_k)
    llm = get_llm_provider(s)
    prompts = Path(__file__).parent / "prompts"
    generator = Generator(llm, (prompts / "generate_answer.md").read_text(encoding="utf-8"))
    rewriter = QueryRewriter(llm, (prompts / "rewrite_query.md").read_text(encoding="utf-8"), s.rewrite_enabled)
    return RAGOrchestrator(rewriter, retriever, generator, RequestLogger(s.log_dir), s)
