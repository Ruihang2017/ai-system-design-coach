"""Project-wide constants."""

import uuid

REFUSAL_MESSAGE = (
    "I don't have enough information in the provided sources to answer that confidently."
)

DEFAULT_UA = "ai-system-design-coach/0.1 (+https://github.com/; educational RAG project)"

# Stable namespace so the same chunk id always maps to the same Qdrant point id.
QDRANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# USD per 1M tokens: (input, output). Update if pricing/model changes.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
