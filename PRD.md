# Product Requirements Document
# AI System Design Coach
### *Evaluated RAG Learning Platform for AI Engineers*

| | |
|---|---|
| **Status** | Draft v1.0 |
| **Owner** | Product Engineering |
| **Last Updated** | May 25, 2026 |
| **Stage** | 0→1 MVP |

---

## 1. TL;DR

**AI System Design Coach** is a RAG-powered learning platform that teaches AI engineering and production system design — and proves its own correctness through a first-class evaluation suite.

Unlike the dozens of "chat with docs" demos in the market, the differentiator is not the chat interface. It is the **evaluation infrastructure** wrapped around it: golden datasets, retrieval benchmarks, faithfulness scoring, citation validation, and regression testing.

> **Every answer is grounded. Every retrieval is measured. Every prompt change is benchmarked. Every hallucination has a failing test.**

This is what separates a toy chatbot from a production-grade AI system — and what makes the project demonstrably credible to engineers, hiring managers, and investors.

---

## 2. Problem

Learners and practicing engineers entering the AI/LLM space face three compounding problems:

**The fragmentation problem.** Knowledge is scattered across LangChain docs, RAGAS papers, vector DB blogs, observability tools, and engineering Twitter threads. There is no canonical "how to think about production RAG" source.

**The trust problem.** Most AI tutorials are demos. They look impressive in a notebook but collapse under real-world conditions — hallucinations, irrelevant retrieval, silent regressions. Learners cannot tell the difference between a system that *works* and one that *sounds smart*.

**The decision problem.** Even after reading the material, engineers struggle to translate concepts into architecture decisions: *RAG vs agentic RAG? Hybrid search vs reranker? Which evals matter for my use case?*

Existing tools (generic LLM chat, doc search, tutorials) solve none of these. They optimize for fluency, not faithfulness; for breadth, not decision-support; for demos, not durability.

---

## 3. Solution

A web application where engineers ask practical AI architecture questions and receive answers that are:

1. **Direct** — addresses the actual question, not generic content.
2. **Grounded** — every claim is backed by retrieved source material.
3. **Cited** — chunks shown inline; user can verify in one click.
4. **Honest** — refuses to answer when retrieved context is insufficient.
5. **Actionable** — includes implementation guidance, tradeoffs, and pitfalls.

Behind this UX sits the real product: an **evaluation harness** that continuously measures retrieval quality, answer faithfulness, citation accuracy, and refusal behavior across a golden benchmark — so every code change produces a before/after report.

---

## 4. Target Users

### Primary: AI Engineer in Training (0–2 years LLM experience)
Bootcamp graduates, ML engineers transitioning to LLM work, backend engineers picking up RAG. They can build a demo but want to learn what production looks like.

### Secondary: Practicing AI Engineer (2–5 years)
Engineers running RAG in production who need a reference for architecture decisions, eval design, and observability patterns.

### Tertiary: Hiring Manager / Tech Lead
Evaluates the *creator* of this system as a candidate. The eval rigor is the signal.

---

## 5. Goals & Non-Goals

### Goals
- Ship an evaluated RAG system, not a chatbot demo.
- Demonstrate measurable hallucination control through automated tests.
- Make every architectural choice (chunk size, top-k, reranker on/off) defensible with data.
- Produce a portfolio artifact that reads as a production AI engineering project.

### Non-Goals
- General-purpose chatbot or assistant.
- Multi-tenant SaaS with auth, billing, and orgs (defer).
- Mobile app.
- Fine-tuning custom models (use frontier APIs).
- Covering every AI topic — scope is **AI engineering / system design** only.

---

## 6. Success Metrics

### North Star
**Eval Pass Rate** — percentage of the 100+ golden benchmark questions that pass all evaluation criteria (faithfulness, relevance, correctness, citation, refusal).

> Target at MVP launch: **≥ 80%**

### Retrieval Quality (gating)
| Metric | Target |
|---|---|
| Recall@5 | ≥ 85% |
| MRR | ≥ 0.70 |
| Context Precision | ≥ 0.75 |

### Answer Quality (gating)
| Metric | Target |
|---|---|
| Faithfulness | ≥ 0.90 |
| Answer Relevance | ≥ 0.85 |
| Correctness vs golden | ≥ 0.80 |
| Citation Accuracy | ≥ 0.90 |
| Refusal Accuracy | ≥ 0.95 |

### Operational
- P95 latency per query: **< 4 seconds**
- Cost per query: **< $0.02**
- Regression detection: any change that drops a gating metric by >2% fails CI.

---

## 7. Functional Requirements

### 7.1 Query → Answer Pipeline
1. **Query rewriting** — expand abbreviations, resolve ambiguity, decompose multi-hop questions.
2. **Retrieval** — top-k chunks from vector DB; configurable hybrid search.
3. **Reranking (optional)** — cross-encoder reranker for top-N candidates.
4. **Generation** — LLM produces answer constrained to retrieved context.
5. **Citation validation** — every claim must map to a cited chunk; unsupported claims trigger refusal.
6. **Evaluation logging** — every request logged with retrieval set, answer, citations, latency, cost.

### 7.2 Answer Contract (enforced)
The assistant MUST:
- Only answer using retrieved context.
- Cite the chunk(s) supporting each non-trivial claim.
- Refuse with a specific message when context is insufficient:
  > *"I don't have enough information in the provided sources to answer that confidently."*

The assistant MUST NOT:
- Use parametric knowledge to fill gaps in retrieval.
- Cite chunks that do not support the claim.
- Answer adversarial prompts that instruct it to ignore documents.

### 7.3 Evaluation System
- **Golden dataset** of 80–120 questions across 7 types (definition, comparison, architecture, troubleshooting, refusal, multi-hop, adversarial).
- **Retrieval evals** computed independently from answer evals.
- **Answer evals** via RAGAS + custom assertions.
- **`pytest evals/`** as the single command to run the full suite.
- **HTML report** generated per run, archived in `reports/eval_runs/`.

### 7.4 Frontend
- Single-page chat interface with citations inline.
- Per-answer expandable "Evaluation" panel showing retrieved chunks, scores, latency, cost.
- "Compare runs" view for benchmarking configuration changes side-by-side.

### 7.5 Observability
- LangSmith trace for every request.
- Per-step latency and token cost.
- Eval score history dashboard.
- Regression alerts when gating metrics drop.

---

## 8. Technical Architecture

```
Frontend (React + TS)
        ↓
FastAPI Backend
        ↓
RAG Orchestrator
        ↓
┌─────────────────────────┐
│ Query Rewriter          │
│ Retriever (Qdrant)      │
│ Reranker (optional)     │
│ Generator (LLM)         │
│ Citation Validator      │
│ Evaluation Logger       │
└─────────────────────────┘
        ↓
LangSmith (traces) + Eval Store
```

### Stack
| Layer | Choice | Rationale |
|---|---|---|
| Frontend | React + TypeScript | Standard, fast iteration |
| Backend | FastAPI | Async, typed, ideal for LLM workloads |
| Vector DB | Qdrant | Production-grade, hybrid search built-in |
| Embeddings | OpenAI `text-embedding-3-large` | Quality baseline; swap-testable |
| Orchestration | LangChain | Mature, integrates with LangSmith |
| Tracing | LangSmith | Industry-relevant for portfolio |
| Eval | RAGAS + custom pytest | RAGAS for standard metrics, custom for refusal/citation |
| LLM | Claude Sonnet 4.6 / GPT-4 class | Configurable per run for A/B |

### Data Sources
50–150 documents covering: LangChain, LangGraph, LlamaIndex, Haystack, LangSmith, RAGAS, DeepEval, Qdrant, Weaviate, Pinecone, Chroma, FastAPI, Docker, Postgres, Redis, OpenTelemetry.

**Phase 1 corpus:** 14 curated URLs fetched from a `docs/sources.yaml` allowlist; 285 chunks ingested as of 2026-05-26. Ingestion is resilient — sources that return HTTP errors (e.g., bot-blocking doc sites) are logged and skipped without crashing the run; fetchable alternatives are substituted in the allowlist.

---

## 9. Roadmap

### Phase 1 — Basic RAG *(Week 1–2)* ✅ Complete — 2026-05-26
Doc ingestion → chunking → embedding → retrieval → generation with citations.
**Done when:** 20 sample questions return cited answers.
**Shipped:** pipeline + minimal UI shipped; ingestion/retrieval verified live (14 sources, 285 chunks); full cited-answer smoke pending user's `OPENAI_API_KEY` (plumbing verified via fake LLM + real retrieval).

### Phase 2 — Golden Eval Dataset *(Week 3)*
100 benchmark questions with expected answers, required sources, refusal cases.
**Done when:** `pytest evals/` produces a score report.

### Phase 3 — Retrieval Eval Dashboard *(Week 4)*
Recall@k, Precision@k, MRR, context precision. A/B chunk sizes, top-k values, hybrid vs vector, reranker on/off.
**Done when:** Chunking strategies are compared quantitatively, with a winner.

### Phase 4 — Answer Eval *(Week 5)*
Faithfulness, relevance, correctness, citation accuracy, refusal accuracy.
**Done when:** Every config change produces a before/after eval report.

### Phase 5 — Production-Style Observability *(Week 6)*
LangSmith integration, cost tracking, latency breakdown, eval history, regression alerts.
**Done when:** A reviewer can trace any answer end-to-end and see its eval scores.

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Golden dataset is biased or incomplete | High | High | 7 question types; include 15+ adversarial and refusal cases; review with peers |
| LLM-as-judge metrics drift between runs | Medium | Medium | Pin judge model version; use deterministic settings; average across 3 runs |
| Eval costs balloon | Medium | Medium | Cache embeddings; batch eval calls; use cheaper judge for routine runs |
| Project reads as "yet another RAG demo" | Medium | High | Lead the README with eval results, not features. Show before/after reports. |
| Scope creep into agents, tools, fine-tuning | High | High | Strict non-goals. Defer to v2. |
| Doc site bot-blocking (403 on automated fetch) | Medium | Low | Resilient ingestion: failing sources are logged and skipped; fetchable alternatives substituted in allowlist (observed in Phase 1: 4 URLs replaced). |

---

## 11. Open Questions

- ~~Do we ship with a single LLM or expose model choice in the UI?~~ **Resolved (Phase 1):** single configurable LLM via env (`LLM_PROVIDER` / `LLM_MODEL`); defaults to OpenAI `gpt-4o-mini`. Model choice is NOT exposed in the UI for MVP.
- ~~Embeddings model?~~ **Resolved (Phase 1):** local `BAAI/bge-small-en-v1.5` (fastembed, 384-d) is the default — zero API cost during development. OpenAI `text-embedding-3-large` retained as the Phase-3 A/B swap (see §8 Stack table — "swap-testable baseline").
- Should the eval dashboard be public (sharable URL) or local-only?
- Is there value in opening the golden dataset itself as a community contribution?
- For Phase 5, do we need a hosted demo or is a recorded walkthrough enough?

---

## 12. The One-Liner

> *Built an evaluated RAG system for AI engineering education with golden test sets, retrieval benchmarks, faithfulness scoring, citation validation, refusal tests, LangSmith tracing, and regression reports.*

This is the sentence that goes on the README, the resume, and the investor deck.

---

*End of PRD*