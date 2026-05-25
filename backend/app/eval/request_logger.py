"""Append-only JSONL request logger — the substrate for Phase 2+ evals."""

from datetime import datetime, timezone
from pathlib import Path

from app.rag.models import RequestLog


class RequestLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def log(self, record: RequestLog) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self._dir / f"requests-{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
