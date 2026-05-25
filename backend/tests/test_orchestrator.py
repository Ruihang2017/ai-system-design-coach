from pathlib import Path

from app.config import Settings
from app.eval.request_logger import RequestLogger
from app.providers.llm import FakeLLMProvider
from app.rag.generator import Generator
from app.rag.models import Chunk, RetrievedChunk, TokenUsage
from app.rag.orchestrator import RAGOrchestrator
from app.rag.query_rewriter import QueryRewriter
from app.constants import REFUSAL_MESSAGE


class FakeRetriever:
    def __init__(self, results):
        self._results = results
    def search(self, query, top_k=None):
        return self._results


def _rc(n, score, text="grounded fact"):
    return RetrievedChunk(chunk=Chunk(id=f"d::{n}", text=text, source_url="u", title=f"T{n}", doc_id="d", chunk_index=n), score=score, n=n)


def _orchestrator(tmp_path, retriever, llm_response):
    s = Settings(_env_file=None, llm_provider="fake", score_threshold=0.30, rewrite_enabled=False)
    llm = FakeLLMProvider(response=llm_response, usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))
    rewriter = QueryRewriter(llm, "REWRITE", enabled=False)
    generator = Generator(llm, "SYSTEM")
    logger = RequestLogger(tmp_path)
    return RAGOrchestrator(rewriter, retriever, generator, logger, s)


def test_grounded_query_returns_cited_answer(tmp_path: Path):
    orch = _orchestrator(tmp_path, FakeRetriever([_rc(1, 0.8), _rc(2, 0.6)]), "Hybrid search helps [1].")
    result = orch.answer("how to retrieve well?")
    assert result.refused is False
    assert result.citations == [1]
    assert result.cost_usd > 0
    assert "total" in result.latency_ms
    assert len(list(tmp_path.glob("requests-*.jsonl"))) == 1  # logged


def test_low_score_triggers_refusal_without_calling_llm(tmp_path: Path):
    orch = _orchestrator(tmp_path, FakeRetriever([_rc(1, 0.05)]), "SHOULD NOT BE USED")
    result = orch.answer("totally off topic question")
    assert result.refused is True and result.answer == REFUSAL_MESSAGE
    assert result.usage.completion_tokens == 0  # generator never ran


def test_uncited_answer_downgrades_to_refusal(tmp_path: Path):
    orch = _orchestrator(tmp_path, FakeRetriever([_rc(1, 0.9)]), "Trust me, no citation here.")
    result = orch.answer("explain something")
    assert result.refused is True and result.answer == REFUSAL_MESSAGE


def test_rewrite_usage_counted_when_refusing(tmp_path):
    """When rewrite is enabled and the score gate refuses, the rewrite's token
    usage is still counted, but the generator is never called."""
    s = Settings(_env_file=None, llm_provider="fake", score_threshold=0.30, rewrite_enabled=True)
    llm = FakeLLMProvider(response="rewritten query", usage=TokenUsage(prompt_tokens=7, completion_tokens=5, total_tokens=12))
    rewriter = QueryRewriter(llm, "REWRITE", enabled=True)
    generator = Generator(llm, "SYSTEM")
    logger = RequestLogger(tmp_path)
    orch = RAGOrchestrator(rewriter, FakeRetriever([_rc(1, 0.05)]), generator, logger, s)

    result = orch.answer("rag?")

    assert result.refused is True
    assert result.usage.total_tokens == 12          # only the rewrite call
    assert result.usage.completion_tokens == 5       # generation was skipped
