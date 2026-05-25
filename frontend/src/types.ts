export interface Chunk {
  id: string; text: string; source_url: string; title: string; doc_id: string; chunk_index: number;
}
export interface RetrievedChunk { chunk: Chunk; score: number; n: number; }
export interface TokenUsage { prompt_tokens: number; completion_tokens: number; total_tokens: number; }
export interface AnswerResult {
  request_id: string;
  answer: string;
  citations: number[];
  chunks: RetrievedChunk[];
  refused: boolean;
  latency_ms: Record<string, number>;
  usage: TokenUsage;
  cost_usd: number;
  model: string;
}
export type Highlight = { turn: number; n: number } | null;
