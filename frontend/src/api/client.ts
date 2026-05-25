import type { AnswerResult } from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function postQuery(query: string): Promise<AnswerResult> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  return (await res.json()) as AnswerResult;
}
