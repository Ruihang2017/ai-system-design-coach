import { describe, it, expect } from "vitest";
import { tokenizeAnswer } from "./citations";

describe("tokenizeAnswer", () => {
  it("splits text and citation markers", () => {
    const tokens = tokenizeAnswer("Use hybrid search [1] and rerankers [2].");
    expect(tokens.filter((t) => t.type === "cite").map((t) => t.n)).toEqual([1, 2]);
    expect(tokens[0]).toEqual({ type: "text", value: "Use hybrid search " });
  });

  it("handles answers without citations", () => {
    const tokens = tokenizeAnswer("No citations here.");
    expect(tokens).toEqual([{ type: "text", value: "No citations here." }]);
  });
});
