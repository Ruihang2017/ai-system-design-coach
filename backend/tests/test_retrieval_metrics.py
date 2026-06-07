from app.evals.retrieval_metrics import hit_rate_at_k, recall_at_k, precision_at_k, mrr


def test_hit_rate():
    assert hit_rate_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], ["z"], 3) == 0.0
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 2) == 0.0  # c is rank 3, outside k=2


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
    assert recall_at_k(["a", "b"], [], 3) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["b"], 3) == 0.3333
    assert precision_at_k(["a", "b", "c"], ["a", "b"], 2) == 1.0
    assert precision_at_k([], ["a"], 0) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], ["b"]) == 0.5
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    assert mrr(["a", "b", "c"], ["z"]) == 0.0
