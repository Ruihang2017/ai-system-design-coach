import json
from pathlib import Path

from app.eval.request_logger import RequestLogger
from app.rag.models import RequestLog, TokenUsage


def _log():
    return RequestLog(
        request_id="r1", ts="2026-05-25T10:00:00Z", query="q", rewritten_query="q2",
        retrieved=[{"chunk_id": "d::0", "score": 0.7, "source_url": "u", "n": 1}],
        answer="a [1]", citations=[1], refused=False, latency_ms={"total": 12.0},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        cost_usd=0.0001, model="gpt-4o-mini", config_snapshot={"top_k": 5},
    )


def test_log_appends_one_jsonl_line(tmp_path: Path):
    logger = RequestLogger(tmp_path)
    logger.log(_log())
    logger.log(_log())
    files = list(tmp_path.glob("requests-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["request_id"] == "r1" and parsed["citations"] == [1]
