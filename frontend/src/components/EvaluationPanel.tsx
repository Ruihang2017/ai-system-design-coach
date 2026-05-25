import { useState } from "react";
import type { AnswerResult } from "../types";

export function EvaluationPanel({ result, highlight }: { result: AnswerResult; highlight: number | null }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="eval-panel">
      <button className="eval-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} Evaluation · {result.refused ? "refused" : `${result.chunks.length} sources`} ·{" "}
        {result.latency_ms.total ?? 0}ms · ${result.cost_usd.toFixed(4)}
      </button>
      {open && (
        <div className="eval-body">
          <div className="eval-meta">
            model={result.model} · tokens={result.usage.total_tokens} ·{" "}
            latency={JSON.stringify(result.latency_ms)}
          </div>
          {result.chunks.map((c) => (
            <div key={c.n} id={`chunk-${c.n}`} className={`chunk ${highlight === c.n ? "chunk-hl" : ""}`}>
              <div className="chunk-head">
                [{c.n}] <a href={c.chunk.source_url} target="_blank" rel="noreferrer">{c.chunk.title}</a>{" "}
                <span className="score">score {c.score.toFixed(3)}</span>
              </div>
              <div className="chunk-text">{c.chunk.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
