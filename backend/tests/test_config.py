from app.config import Settings
from app.constants import REFUSAL_MESSAGE, MODEL_PRICING


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.llm_model == "gpt-4o-mini"
    assert s.embed_provider == "fastembed"
    assert s.top_k == 5
    assert 0.0 <= s.score_threshold <= 1.0


def test_constants_present():
    assert "confidently" in REFUSAL_MESSAGE
    assert "gpt-4o-mini" in MODEL_PRICING
