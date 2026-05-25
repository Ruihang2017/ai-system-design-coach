"""Run the 20 sample questions through the real pipeline and print cited answers.

Prerequisites:
  1. docker compose up -d   (Qdrant running)
  2. python scripts/ingest.py   (corpus ingested)
  3. OPENAI_API_KEY set in .env
Run from backend/:  python scripts/run_samples.py
"""

from pathlib import Path

import yaml

from app.logging_config import configure_logging
from app.rag.orchestrator import build_orchestrator


def main() -> None:
    configure_logging()
    qs = yaml.safe_load((Path(__file__).parent / "sample_questions.yaml").read_text(encoding="utf-8"))["questions"]
    orch = build_orchestrator()
    refused = 0
    for i, q in enumerate(qs, 1):
        r = orch.answer(q["text"])
        refused += int(r.refused)
        status = "REFUSED" if r.refused else f"cited {r.citations}"
        print(f"\n[{i:02d}] ({q['type']}) {q['text']}")
        print(f"     -> {status} | {r.latency_ms.get('total', 0)}ms | ${r.cost_usd:.4f}")
        print(f"     {r.answer[:300]}")
    print(f"\nSummary: {len(qs)} questions, {refused} refusals.")


if __name__ == "__main__":
    main()
