"""Side-by-side comparison report for a retrieval sweep (HTML + Markdown + JSON)."""

import html
from pathlib import Path

from app.evals.models import SweepReport

_METRIC_COLS = ["recall_at_5", "precision_at_5", "hit_rate_at_5", "mrr"]


def _metric_val(metrics: dict, col: str) -> float:
    # Fall back to the k=10 variant's key if the k=5 key is absent.
    return metrics.get(col, metrics.get(col.replace("_5", "_10"), 0.0))


def _markdown(report: SweepReport) -> str:
    lines = [
        f"# Retrieval Sweep {report.run_id}", "",
        f"- ts: {report.ts}",
        f"- **winner: `{report.winner_label}`** {report.winner_config}",
        f"- stage winners: {report.stage_winners}", "",
        "| variant | " + " | ".join(_METRIC_COLS) + " | n |",
        "|" + "---|" * (len(_METRIC_COLS) + 2),
    ]
    for v in report.variants:
        mark = " **<-- winner**" if v.label == report.winner_label else ""
        vals = " | ".join(f"{_metric_val(v.metrics, m):.4f}" for m in _METRIC_COLS)
        lines.append(f"| `{v.label}`{mark} | {vals} | {v.n_questions} |")
    return "\n".join(lines) + "\n"


def _html(report: SweepReport) -> str:
    style = ("<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:6px 10px}tr.win{background:#e6ffe6;font-weight:600}"
             "code{background:#f4f4f4;padding:1px 4px}</style>")
    head = "".join(f"<th>{html.escape(m)}</th>" for m in _METRIC_COLS)
    rows = []
    for v in report.variants:
        cls = " class='win'" if v.label == report.winner_label else ""
        tds = "".join(f"<td>{_metric_val(v.metrics, m):.4f}</td>" for m in _METRIC_COLS)
        rows.append(f"<tr{cls}><td><code>{html.escape(v.label)}</code></td>{tds}<td>{v.n_questions}</td></tr>")
    return (f"<html><head>{style}</head><body>"
            f"<h1>Retrieval Sweep {html.escape(report.run_id)}</h1>"
            f"<p>Winner: <code>{html.escape(report.winner_label)}</code> &mdash; "
            f"{html.escape(str(report.winner_config))}</p>"
            f"<table><tr><th>variant</th>{head}<th>n</th></tr>{''.join(rows)}</table></body></html>")


def write_comparison_report(report: SweepReport, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"sweep-{report.run_id}"
    base.with_suffix(".json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    base.with_suffix(".html").write_text(_html(report), encoding="utf-8")
    return {
        "json": str(base.with_suffix(".json")),
        "md": str(base.with_suffix(".md")),
        "html": str(base.with_suffix(".html")),
    }
