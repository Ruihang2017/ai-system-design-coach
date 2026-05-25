from app.rag.models import (
    Chunk, RetrievedChunk, TokenUsage, AnswerResult, RequestLog,
    add_usage, compute_cost,
)


def _chunk(i=0):
    return Chunk(id=f"d::{i}", text="t", source_url="u", title="T", doc_id="d", chunk_index=i)


def test_retrieved_chunk_roundtrip():
    rc = RetrievedChunk(chunk=_chunk(), score=0.9, n=1)
    assert rc.chunk.doc_id == "d" and rc.n == 1


def test_add_usage():
    a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    c = add_usage(a, b)
    assert (c.prompt_tokens, c.completion_tokens, c.total_tokens) == (13, 7, 20)


def test_compute_cost_known_model():
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    # gpt-4o-mini = (0.15, 0.60) per 1M
    assert compute_cost("gpt-4o-mini", usage) == 0.75


def test_compute_cost_unknown_model_is_zero():
    assert compute_cost("nonexistent", TokenUsage(prompt_tokens=100)) == 0.0


def test_answer_result_and_log_serialize():
    ar = AnswerResult(request_id="r", answer="hi [1]", citations=[1], chunks=[RetrievedChunk(chunk=_chunk(), score=0.5, n=1)], model="gpt-4o-mini")
    log = RequestLog(request_id="r", ts="2026-05-25T00:00:00Z", query="q", rewritten_query="q",
                     retrieved=[{"chunk_id": "d::0", "score": 0.5, "source_url": "u", "n": 1}],
                     answer="hi [1]", citations=[1], refused=False, latency_ms={"total": 1.0},
                     usage=TokenUsage(), cost_usd=0.0, model="gpt-4o-mini", config_snapshot={"top_k": 5})
    assert ar.model_dump()["citations"] == [1]
    assert "\n" not in log.model_dump_json()  # one JSONL line
