from app.evals.models import GoldenQuestion, RetrievalReport
from app.evals.retrieval_eval import evaluate_retrieval
from app.rag.models import Chunk, RetrievedChunk


class FakeRetriever:
    def __init__(self, mapping):
        self._mapping = mapping  # question text -> list[source_url]
    def search(self, query, top_k=None):
        srcs = self._mapping.get(query, [])
        return [
            RetrievedChunk(
                chunk=Chunk(id=f"x::{i}", text="t", source_url=u, title="T", doc_id="x", chunk_index=i),
                score=1.0 - i * 0.1, n=i + 1,
            )
            for i, u in enumerate(srcs)
        ]


def test_evaluate_retrieval_aggregates_metrics():
    qs = [
        GoldenQuestion(id="q1", type="definition", question="A?", required_sources=["s1"]),
        GoldenQuestion(id="q2", type="definition", question="B?", required_sources=["s2"]),
        GoldenQuestion(id="q3", type="refusal", question="R?", should_refuse=True),  # skipped (no required_sources)
    ]
    retriever = FakeRetriever({"A?": ["s1", "sx"], "B?": ["sy", "sz"]})  # q1 hits, q2 misses
    report = evaluate_retrieval(qs, retriever, k=5, label="test", config={"foo": 1})
    assert isinstance(report, RetrievalReport)
    assert report.n_questions == 2  # only grounded counted
    assert report.metrics["hit_rate_at_5"] == 0.5
    assert report.metrics["recall_at_5"] == 0.5
    assert report.label == "test" and report.config == {"foo": 1}
    assert len(report.per_question) == 2
