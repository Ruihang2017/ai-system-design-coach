"""LLM providers. Default OpenAI; Fake for tests."""

from typing import Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings
from app.rag.models import TokenUsage


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, system: str, user: str) -> tuple[str, TokenUsage]: ...


class FakeLLMProvider:
    """Returns a canned response; records the last prompt for assertions."""

    def __init__(self, response: str = "", usage: TokenUsage | None = None) -> None:
        self.response = response
        self.usage = usage or TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, system: str, user: str) -> tuple[str, TokenUsage]:
        self.last_system, self.last_user = system, user
        return self.response, self.usage


class OpenAILLMProvider:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def generate(self, system: str, user: str) -> tuple[str, TokenUsage]:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            timeout=60,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = TokenUsage(
            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, total_tokens=u.total_tokens
        )
        return text, usage


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        return FakeLLMProvider()
    return OpenAILLMProvider(settings.llm_model, settings.openai_api_key)
