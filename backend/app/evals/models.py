"""Pydantic data models for the Phase 2 eval harness."""

from pydantic import BaseModel, Field, field_validator

QUESTION_TYPES = {
    "definition",
    "comparison",
    "architecture",
    "troubleshooting",
    "refusal",
    "multi_hop",
    "adversarial",
}


class GoldenQuestion(BaseModel):
    id: str
    type: str  # must be one of QUESTION_TYPES
    question: str
    should_refuse: bool = False
    required_sources: list[str] = Field(default_factory=list)
    expected_points: list[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        if v not in QUESTION_TYPES:
            raise ValueError(f"type must be one of {QUESTION_TYPES!r}, got {v!r}")
        return v


class QuestionOutcome(BaseModel):
    id: str
    type: str
    question: str
    should_refuse: bool
    refused: bool
    citations: list[int]
    retrieved_sources: list[str]
    required_sources: list[str]
    source_hit: bool
    citation_valid: bool
    refusal_correct: bool
    passed: bool
    latency_ms_total: float
    cost_usd: float


class EvalReport(BaseModel):
    run_id: str
    ts: str
    config_snapshot: dict = Field(default_factory=dict)
    n_questions: int
    metrics: dict[str, float] = Field(default_factory=dict)
    outcomes: list[QuestionOutcome] = Field(default_factory=list)


class RetrievalReport(BaseModel):
    label: str
    config: dict = Field(default_factory=dict)
    n_questions: int
    metrics: dict[str, float] = Field(default_factory=dict)
    per_question: list[dict] = Field(default_factory=list)


class SweepReport(BaseModel):
    run_id: str
    ts: str
    variants: list[RetrievalReport] = Field(default_factory=list)
    stage_winners: dict = Field(default_factory=dict)
    winner_label: str = ""
    winner_config: dict = Field(default_factory=dict)
