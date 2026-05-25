from app.providers.llm import FakeLLMProvider
from app.rag.models import TokenUsage
from app.rag.query_rewriter import QueryRewriter


def test_rewrite_enabled_uses_llm():
    llm = FakeLLMProvider(response="retrieval augmented generation tradeoffs", usage=TokenUsage(prompt_tokens=4, completion_tokens=4, total_tokens=8))
    rw = QueryRewriter(llm, system_prompt="REWRITE", enabled=True)
    text, usage = rw.rewrite("RAG tradeoffs?")
    assert text == "retrieval augmented generation tradeoffs"
    assert usage.total_tokens == 8


def test_rewrite_disabled_returns_input_with_zero_usage():
    llm = FakeLLMProvider(response="SHOULD NOT BE USED")
    rw = QueryRewriter(llm, system_prompt="REWRITE", enabled=False)
    text, usage = rw.rewrite("what is a vector db?")
    assert text == "what is a vector db?" and usage.total_tokens == 0


def test_rewrite_empty_llm_output_falls_back_to_input():
    llm = FakeLLMProvider(response="   ")
    rw = QueryRewriter(llm, system_prompt="REWRITE", enabled=True)
    text, _ = rw.rewrite("keep me")
    assert text == "keep me"
