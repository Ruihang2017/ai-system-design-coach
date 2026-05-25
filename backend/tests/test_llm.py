from app.providers.llm import FakeLLMProvider, get_llm_provider
from app.rag.models import TokenUsage
from app.config import Settings


def test_fake_llm_returns_canned_text_and_usage():
    llm = FakeLLMProvider(response="answer [1]", usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13))
    text, usage = llm.generate("system", "user")
    assert text == "answer [1]"
    assert usage.total_tokens == 13


def test_fake_llm_records_last_prompt():
    llm = FakeLLMProvider(response="x")
    llm.generate("SYS", "USER")
    assert llm.last_system == "SYS" and llm.last_user == "USER"


def test_factory_returns_fake():
    llm = get_llm_provider(Settings(_env_file=None, llm_provider="fake"))
    assert isinstance(llm, FakeLLMProvider)
