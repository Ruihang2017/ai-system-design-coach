from app.providers.llm import FakeLLMProvider
from app.rag.generator import Generator, build_context, parse_citations
from app.rag.models import Chunk, RetrievedChunk, TokenUsage


def _rc(n, text):
    return RetrievedChunk(chunk=Chunk(id=f"d::{n}", text=text, source_url="u", title=f"T{n}", doc_id="d", chunk_index=n), score=0.9, n=n)


def test_parse_citations_dedup_sorted():
    assert parse_citations("foo [2] bar [1] baz [2]") == [1, 2]
    assert parse_citations("no citations here") == []


def test_build_context_numbers_sources():
    ctx = build_context([_rc(1, "alpha"), _rc(2, "beta")])
    assert "[1]" in ctx and "alpha" in ctx and "[2]" in ctx and "beta" in ctx


def test_generator_calls_llm_with_context_and_question():
    llm = FakeLLMProvider(response="Use hybrid search [1].", usage=TokenUsage(prompt_tokens=5, completion_tokens=4, total_tokens=9))
    gen = Generator(llm, system_prompt="SYSTEM RULES")
    text, usage = gen.generate("how to retrieve?", [_rc(1, "hybrid search helps")])
    assert text == "Use hybrid search [1]."
    assert usage.total_tokens == 9
    assert llm.last_system == "SYSTEM RULES"
    assert "hybrid search helps" in llm.last_user and "how to retrieve?" in llm.last_user
