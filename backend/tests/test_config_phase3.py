from app.config import Settings


def test_phase3_defaults_preserve_current_behavior():
    s = Settings(_env_file=None)
    assert s.rerank_enabled is False
    assert s.hybrid_enabled is False
    assert s.rerank_candidates == 20
    assert s.rerank_model and s.sparse_model
