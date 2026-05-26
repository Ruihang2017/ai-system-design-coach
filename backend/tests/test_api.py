from fastapi.testclient import TestClient

from app.main import app, get_orchestrator
from app.rag.models import AnswerResult, Chunk, RetrievedChunk, TokenUsage


class FakeOrchestrator:
    def answer(self, query: str) -> AnswerResult:
        rc = RetrievedChunk(chunk=Chunk(id="d::0", text="ctx", source_url="u", title="T", doc_id="d", chunk_index=0), score=0.9, n=1)
        return AnswerResult(request_id="r1", answer="grounded [1]", citations=[1], chunks=[rc],
                            refused=False, latency_ms={"total": 5.0}, usage=TokenUsage(total_tokens=10),
                            cost_usd=0.001, model="gpt-4o-mini")


def _client():
    app.dependency_overrides[get_orchestrator] = lambda: FakeOrchestrator()
    return TestClient(app)


def test_query_returns_answer_result():
    client = _client()
    resp = client.post("/query", json={"query": "what is rag?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "grounded [1]"
    assert body["citations"] == [1]
    assert body["chunks"][0]["chunk"]["title"] == "T"
    app.dependency_overrides.clear()


def test_query_validation_error_on_missing_field():
    client = _client()
    resp = client.post("/query", json={})
    assert resp.status_code == 422
    app.dependency_overrides.clear()


def test_query_upstream_error_returns_502_with_cors_headers():
    """An upstream/provider failure (e.g. OpenAI 429) must return a clean,
    CORS-headed error instead of a bare 500 that the browser misreports as CORS."""

    class BoomOrchestrator:
        def answer(self, query: str):
            raise RuntimeError("Error code: 429 - insufficient_quota")

    app.dependency_overrides[get_orchestrator] = lambda: BoomOrchestrator()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/query", json={"query": "hi"}, headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 502
    # The whole point: the CORS header is present on the error response.
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "insufficient_quota" in resp.json()["detail"]
    app.dependency_overrides.clear()


def test_health_ok(monkeypatch):
    class _FakeQdrant:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            return []

    monkeypatch.setattr("qdrant_client.QdrantClient", _FakeQdrant)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["qdrant"] is True


def test_health_reports_qdrant_down(monkeypatch):
    def _boom(*args, **kwargs):
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr("qdrant_client.QdrantClient", _boom)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["qdrant"] is False
