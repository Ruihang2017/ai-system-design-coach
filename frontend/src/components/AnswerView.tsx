import { tokenizeAnswer } from "../lib/citations";

export function AnswerView({ answer, onCiteClick }: { answer: string; onCiteClick: (n: number) => void }) {
  return (
    <p className="answer">
      {tokenizeAnswer(answer).map((t, i) =>
        t.type === "text" ? (
          <span key={i}>{t.value}</span>
        ) : (
          <button key={i} className="cite" onClick={() => onCiteClick(t.n)}>
            [{t.n}]
          </button>
        ),
      )}
    </p>
  );
}
