"""Optional LLM-based query rewriting."""

from app.providers.llm import LLMProvider
from app.rag.models import TokenUsage


class QueryRewriter:
    def __init__(self, llm: LLMProvider, system_prompt: str, enabled: bool = True) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._enabled = enabled

    def rewrite(self, query: str) -> tuple[str, TokenUsage]:
        if not self._enabled:
            return query, TokenUsage()
        text, usage = self._llm.generate(self._system_prompt, query)
        cleaned = text.strip()
        return (cleaned or query), usage
