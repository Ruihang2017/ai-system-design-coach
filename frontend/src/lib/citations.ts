export type AnswerToken =
  | { type: "text"; value: string }
  | { type: "cite"; value: string; n: number };

export function tokenizeAnswer(answer: string): AnswerToken[] {
  const tokens: AnswerToken[] = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(answer)) !== null) {
    if (m.index > last) tokens.push({ type: "text", value: answer.slice(last, m.index) });
    tokens.push({ type: "cite", value: m[0], n: Number(m[1]) });
    last = re.lastIndex;
  }
  if (last < answer.length) tokens.push({ type: "text", value: answer.slice(last) });
  return tokens;
}
