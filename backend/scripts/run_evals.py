"""Run the golden benchmark through the real pipeline and write a score report.

Prereqs: Qdrant running + corpus ingested + OPENAI_API_KEY in .env.
Run from backend/:  python scripts/run_evals.py
"""

from pathlib import Path

from app.config import settings
from app.evals.dataset import load_dataset
from app.evals.report import write_report
from app.evals.runner import evaluate
from app.logging_config import configure_logging
from app.rag.orchestrator import build_orchestrator


def main() -> None:
    configure_logging()
    dataset_path = Path(__file__).resolve().parents[1] / "app" / "evals" / "golden" / "dataset.yaml"
    questions = load_dataset(dataset_path)
    orchestrator = build_orchestrator(settings)
    report = evaluate(questions, orchestrator, settings)
    paths = write_report(report, settings.log_dir)
    print("\n=== Phase 2 Eval Report ===")
    print(f"run_id: {report.run_id}  questions: {report.n_questions}")
    for k, v in report.metrics.items():
        print(f"  {k}: {v}")
    print(f"report (json): {paths['json']}")
    print(f"report (html): {paths['html']}")


if __name__ == "__main__":
    main()
