import { useState } from "react";
import type { AnswerResult, Highlight } from "../types";

export function EvaluationPanel({
  result,
  turnIdx,
  highlight,
}: {
  result: AnswerResult;
  turnIdx: number;
  highlight: Highlight;
}) {
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
          {result.chunks.map((c) => {
            const isHighlighted = highlight?.turn === turnIdx && highlight?.n === c.n;
            return (
              <div key={c.n} id={`chunk-${turnIdx}-${c.n}`} className={`chunk ${isHighlighted ? "chunk-hl" : ""}`}>
                <div className="chunk-head">
                  [{c.n}]{" "}
                  <a href={c.chunk.source_url} target="_blank" rel="noreferrer">
                    {c.chunk.title}
                  </a>{" "}
                  <span className="score">score {c.score.toFixed(3)}</span>
                </div>
                <div className="chunk-text">{c.chunk.text}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
