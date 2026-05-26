from app.config import Settings


def test_phase3_defaults_apply_retrieval_winner():
    s = Settings(_env_file=None)
    assert s.hybrid_enabled is True            # Phase 3 winner: hybrid on
    assert s.rerank_enabled is False           # reranker hurt recall on the benchmark
    assert s.qdrant_collection == "ai_coach_cs1000_hybrid"
    assert s.rerank_candidates == 20 and s.rerank_model and s.sparse_model
