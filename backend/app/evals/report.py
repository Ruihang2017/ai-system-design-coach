"""Write HTML and JSON eval reports from an EvalReport."""

from pathlib import Path

from app.evals.models import EvalReport, QuestionOutcome


def _render_html(report: EvalReport) -> str:
    def fmt_metric(k: str, v: float) -> str:
        if k.startswith("n_"):
            return str(int(v))
        return f"{v * 100:.1f}%"

    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{fmt_metric(k, v)}</td></tr>"
        for k, v in sorted(report.metrics.items())
    )

    def outcome_row(o: QuestionOutcome) -> str:
        row_class = "pass" if o.passed else "fail"
        truncated_q = o.question[:80] + "…" if len(o.question) > 80 else o.question
        citations_str = ", ".join(str(c) for c in o.citations) or "—"
        return (
            f'<tr class="{row_class}">'
            f"<td>{o.id}</td>"
            f"<td>{o.type}</td>"
            f"<td>{truncated_q}</td>"
            f"<td>{'yes' if o.should_refuse else 'no'}</td>"
            f"<td>{'yes' if o.refused else 'no'}</td>"
            f"<td>{'PASS' if o.passed else 'FAIL'}</td>"
            f"<td>{citations_str}</td>"
            f"<td>{'yes' if o.source_hit else 'no'}</td>"
            f"</tr>"
        )

    outcome_rows = "".join(outcome_row(o) for o in report.outcomes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Report {report.run_id}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  p.meta {{ color: #555; font-size: 0.9rem; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.7rem; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr.pass td {{ background: #e6f4ea; }}
  tr.fail td {{ background: #fce8e6; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Eval Report</h1>
<p class="meta"><strong>run_id:</strong> {report.run_id} &nbsp;|&nbsp; \
<strong>ts:</strong> {report.ts} &nbsp;|&nbsp; \
<strong>questions:</strong> {report.n_questions}</p>

<h2>Metrics</h2>
<table>
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>{metric_rows}</tbody>
</table>

<h2>Question Outcomes</h2>
<table>
  <thead>
    <tr>
      <th>ID</th><th>Type</th><th>Question</th>
      <th>Should Refuse</th><th>Refused</th><th>Passed</th>
      <th>Citations</th><th>Source Hit</th>
    </tr>
  </thead>
  <tbody>{outcome_rows}</tbody>
</table>
</body>
</html>"""


def write_report(report: EvalReport, out_dir: str | Path) -> dict[str, str]:
    """Write JSON and HTML reports to out_dir.

    Returns:
        dict with keys "json" and "html" pointing to the written file paths (str).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"eval-{report.run_id}"
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")

    return {"json": str(json_path), "html": str(html_path)}
