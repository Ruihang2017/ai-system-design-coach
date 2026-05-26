import { describe, it, expect, vi, afterEach } from "vitest";
import { postQuery } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockFetch(ok: boolean, status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({ ok, status, json: async () => body }) as unknown as Response),
  );
}

describe("postQuery", () => {
  it("returns the parsed AnswerResult on success", async () => {
    const payload = {
      request_id: "r",
      answer: "Use hybrid search [1].",
      citations: [1],
      chunks: [],
      refused: false,
      latency_ms: { total: 5 },
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      cost_usd: 0,
      model: "gpt-4o-mini",
    };
    mockFetch(true, 200, payload);
    const res = await postQuery("hi");
    expect(res.answer).toBe("Use hybrid search [1].");
    expect(res.citations).toEqual([1]);
  });

  it("surfaces the server error detail in the thrown message", async () => {
    mockFetch(false, 502, {
      detail: "Upstream pipeline error (RateLimitError): Error code: 429 - insufficient_quota",
    });
    await expect(postQuery("hi")).rejects.toThrow(/insufficient_quota/);
  });

  it("falls back to the status code when the error body has no detail", async () => {
    mockFetch(false, 500, {});
    await expect(postQuery("hi")).rejects.toThrow(/Query failed: 500/);
  });
});
