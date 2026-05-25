"""Pydantic data models and cost/usage helpers for the RAG pipeline."""

from pydantic import BaseModel, Field

from app.constants import MODEL_PRICING


class Chunk(BaseModel):
    id: str
    text: str
    source_url: str
    title: str
    doc_id: str
    chunk_index: int


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    n: int  # 1-based citation number assigned at retrieval time


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AnswerResult(BaseModel):
    request_id: str
    answer: str
    citations: list[int] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    refused: bool = False
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    model: str = ""


class RequestLog(BaseModel):
    request_id: str
    ts: str
    query: str
    rewritten_query: str
    retrieved: list[dict] = Field(default_factory=list)
    answer: str
    citations: list[int] = Field(default_factory=list)
    refused: bool = False
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    model: str = ""
    config_snapshot: dict = Field(default_factory=dict)


def add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )


def compute_cost(model: str, usage: TokenUsage) -> float:
    inp, out = MODEL_PRICING.get(model, (0.0, 0.0))
    return round(usage.prompt_tokens / 1e6 * inp + usage.completion_tokens / 1e6 * out, 6)
