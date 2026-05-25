"""Answer generation: build numbered context, call the LLM, parse citations."""

import re

from app.providers.llm import LLMProvider
from app.rag.models import RetrievedChunk, TokenUsage

_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer: str) -> list[int]:
    return sorted({int(m) for m in _CITATION_RE.findall(answer)})


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.n}] ({c.chunk.title}) {c.chunk.text}" for c in chunks)


class Generator:
    def __init__(self, llm: LLMProvider, system_prompt: str) -> None:
        self._llm = llm
        self._system_prompt = system_prompt

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, TokenUsage]:
        context = build_context(chunks)
        user = f"Context:\n{context}\n\nQuestion: {query}"
        return self._llm.generate(self._system_prompt, user)
