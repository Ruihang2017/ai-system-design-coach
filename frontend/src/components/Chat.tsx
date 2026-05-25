import { useState } from "react";
import { postQuery } from "../api/client";
import type { AnswerResult, Highlight } from "../types";
import { AnswerView } from "./AnswerView";
import { EvaluationPanel } from "./EvaluationPanel";

interface Turn { query: string; result?: AnswerResult; error?: string; }

export function Chat() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [highlight, setHighlight] = useState<Highlight>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q) return;
    setInput(""); setLoading(true);
    const idx = turns.length;
    setTurns((t) => [...t, { query: q }]);
    try {
      const result = await postQuery(q);
      setTurns((t) => t.map((turn, i) => (i === idx ? { ...turn, result } : turn)));
    } catch (err) {
      setTurns((t) => t.map((turn, i) => (i === idx ? { ...turn, error: String(err) } : turn)));
    } finally {
      setLoading(false);
    }
  }

  function onCiteClick(turnIdx: number, n: number) {
    setHighlight({ turn: turnIdx, n });
    document.getElementById(`chunk-${turnIdx}-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <div className="chat">
      {turns.map((turn, i) => (
        <div key={i} className="turn">
          <div className="q">{turn.query}</div>
          {turn.error && <div className="error">{turn.error}</div>}
          {turn.result && (
            <div className={`a ${turn.result.refused ? "refused" : ""}`}>
              <AnswerView answer={turn.result.answer} onCiteClick={(n) => onCiteClick(i, n)} />
              <EvaluationPanel result={turn.result} turnIdx={i} highlight={highlight} />
            </div>
          )}
        </div>
      ))}
      <form className="composer" onSubmit={submit}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask an AI engineering question…" />
        <button disabled={loading}>{loading ? "…" : "Ask"}</button>
      </form>
    </div>
  );
}
