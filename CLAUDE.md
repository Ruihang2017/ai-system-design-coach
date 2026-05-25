# CLAUDE.md

This file gives Claude Code the working agreement for this project. Read it at the start of every session and follow it.

---

## Project

**AI System Design Coach** — an evaluated RAG learning platform for AI engineers.

Full product spec lives in `PRD.md`. The killer feature is **not** the chat UI — it is the evaluation infrastructure (golden datasets, retrieval benchmarks, faithfulness scoring, citation validation, regression tests).

When in doubt about scope or priorities, defer to `PRD.md`.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Frontend | React + TypeScript |
| Backend | FastAPI (Python 3.10+) |
| Vector DB | Qdrant |
| Orchestration | LangChain |
| Tracing | LangSmith |
| Eval | RAGAS + Pytest |
| LLM | Claude / GPT-4 class (configurable) |

---

## The Five Non-Negotiable Rules

These rules apply to **every** task. Do not skip them, even on small changes.

### Rule 1 — Always Test the Implementation

No implementation is "done" until tests pass. The contract:

- **Write tests first or alongside the code**, never as an afterthought.
- **Run the tests** before reporting completion. Show the output.
- For backend: `pytest` (unit + integration as relevant).
- For retrieval/answer changes: run the relevant eval slice in `backend/app/evals/`.
- For frontend: component tests where logic exists; manual verification documented in the handover otherwise.
- If a test fails, **fix it before declaring done**. Do not paper over with skips or excessive mocking.
- If you genuinely cannot test something (e.g., needs production keys), say so explicitly in the handover under "Untested".

> **Never report a task complete without showing test output.**

---

### Rule 2 — Write a Handover File After Every Implementation

At the end of every task, create a handover file at:

```
HANDOVERS/YYYY-MM-DD_HHMM_<short-task-slug>.md
```

Use this template exactly:

```markdown
# Handover: <Task Title>

**Date:** YYYY-MM-DD HH:MM
**Branch / Commit:** <branch>@<sha>
**Phase (from PRD):** <e.g. Phase 2 — Golden Eval Dataset>

## What Was Done
<2–6 sentences. Plain language. What the user can now do that they couldn't before.>

## Files Changed
- `path/to/file.py` — <one-line reason>
- `path/to/other.tsx` — <one-line reason>

## Tests
- Commands run: `pytest backend/tests/test_retriever.py -v`
- Result: <pass / fail counts, key numbers>
- New tests added: <list>

## Eval Impact (if applicable)
- Before: Recall@5 = 0.82, Faithfulness = 0.88
- After:  Recall@5 = 0.87, Faithfulness = 0.91
- Report: `reports/eval_runs/<run-id>.html`

## Known Issues / Untested
<Anything brittle, deferred, or unverified. Be honest.>

## How to Verify Locally
<Exact commands the user can run to reproduce the result.>
```

If the handover would be empty or trivial (e.g., a typo fix), still create one — keep it short. The history matters.

---

### Rule 3 — Always Recommend Next Steps (3 Options + a Pick)

After every completed task, end your response with a **Next Steps** section in this exact shape:

```
## Next Steps

**Option A — <Name>**
What it is: <one sentence>
Why: <one sentence on value>
Cost: <S/M/L effort>

**Option B — <Name>**
What it is: <one sentence>
Why: <one sentence on value>
Cost: <S/M/L effort>

**Option C — <Name>**
What it is: <one sentence>
Why: <one sentence on value>
Cost: <S/M/L effort>

**My recommendation: Option <X>**
Because <reasoning tied to the PRD phase, gating metrics, or de-risking>.
```

Guidance:
- The three options should be **genuinely different directions**, not three flavors of the same thing.
- Ground the recommendation in something concrete: the current PRD phase, a failing metric, an obvious risk, or user-facing impact.
- Don't hedge. Pick one and defend it. The user can override.

---

### Rule 4 — Keep PRD.md and README.md as Live Docs

These are **not** write-once artifacts. They evolve with the project.

**Update `PRD.md` when:**
- A feature is added, removed, or materially changed in scope.
- A success metric target is met, missed, or revised.
- An "Open Question" is resolved → move it to the relevant section.
- A risk materializes or is mitigated → update the risks table.
- A phase completes → mark it complete with the date.

**Update `README.md` when:**
- The user-facing setup, run, or build commands change.
- A new dependency is added.
- The eval workflow changes (this is a headline feature — keep it accurate).
- Project status changes (current phase, latest eval scores).

**README.md should always show:**
1. The one-line pitch.
2. Current phase and status.
3. Latest gating eval scores (Recall@5, Faithfulness, etc.) with the date.
4. Quickstart commands that actually work.
5. Link to the latest handover.

When updating either doc, include the update in your commit alongside the code change — not as a separate afterthought.

---

### Rule 5 — Commit, But Never Push

Git workflow:

1. After tests pass and docs are updated, stage all related changes.
2. Commit using **conventional commit** format:
   - `feat: add hybrid search to retriever`
   - `fix: citation validator no longer crashes on empty context`
   - `eval: add 20 refusal cases to golden dataset`
   - `docs: update PRD phase 2 to complete`
   - `chore: bump qdrant client to 1.x`
3. **Do not run `git push`.** The user merges manually.
4. If a change spans multiple logical units, make multiple commits — don't squash a feature, tests, and docs into one mega-commit.
5. If you create a new branch, name it `<type>/<short-slug>` (e.g., `feat/reranker`, `eval/golden-v2`).

If the user has uncommitted changes when you start, **ask before committing them**. Don't sweep someone else's work into your commit.

---

## Standard Workflow (Putting It All Together)

For any non-trivial task, follow this order:

1. **Understand** — re-read the relevant PRD section. Confirm scope with the user if ambiguous.
2. **Plan** — state the approach in 2–4 bullets before writing code.
3. **Implement** — write the code.
4. **Test** — run the tests; show output; fix until green.
5. **Eval (if RAG-pipeline-touching)** — run the relevant eval; record before/after.
6. **Update docs** — PRD and/or README as Rule 4 dictates.
7. **Commit** — conventional message; no push.
8. **Handover** — write the handover file per Rule 2.
9. **Recommend** — 3 next-step options + a pick per Rule 3.

---

## Repo Layout

```
ai-system-design-coach/
  backend/
    app/
      main.py
      rag/                 # ingest, chunking, retriever, reranker, generator, citation_checker
      evals/               # golden dataset, retrieval/answer eval runners, metrics
    tests/                 # pytest unit + integration tests
  frontend/
    src/
      pages/
      components/
  docs/
    raw/                   # source docs (gitignored if large)
    processed/             # chunked + embedded artifacts
  reports/
    eval_runs/             # HTML eval reports, archived
  HANDOVERS/               # one file per completed task (Rule 2)
  PRD.md                   # live (Rule 4)
  README.md                # live (Rule 4)
  CLAUDE.md                # this file
```

---

## Code Conventions

- **Python:** type hints required on public functions; `ruff` + `black`; no `print()` in app code — use the logger.
- **TypeScript:** strict mode on; no `any` without a comment justifying it.
- **Secrets:** never commit keys. Use `.env` (gitignored) and document required vars in `README.md`.
- **LLM prompts:** live in versioned files under `backend/app/rag/prompts/`, not inline. Changing a prompt is a code change and requires an eval run.
- **Error handling:** retrieval and LLM calls must have retry + timeout. Surface failures with structured logs to LangSmith.

---

## Anti-Patterns (Don't Do These)

- ❌ Marking a task done without running tests.
- ❌ "I'll write the handover next time" — write it now.
- ❌ Skipping the eval after touching the RAG pipeline.
- ❌ Updating PRD/README in a separate commit "later".
- ❌ Pushing to remote.
- ❌ Adding parametric knowledge fallbacks to mask retrieval failures. The system must refuse, not paper over.
- ❌ Committing prompt changes without an eval run showing the before/after.
- ❌ Vague next-step recommendations like "improve performance" or "add more features".

---

## When to Pause and Ask

Stop and ask the user before proceeding when:
- The task would require changing a **gating metric target** in the PRD.
- The change would add a new external service or paid API.
- A test reveals a deeper architectural issue, not just a bug.
- The user's request conflicts with something in `PRD.md`.

Otherwise, exercise judgment and proceed — explain your reasoning in the handover.

---

*This file is itself a live doc. If a rule here proves wrong in practice, propose a change to the user — don't silently ignore it.*