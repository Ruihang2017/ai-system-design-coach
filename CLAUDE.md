# CLAUDE.md

**Bias:** ship investment-grade POCs. Correctness > polish > volume. Smaller surface = sharper context next session.

## Doc layout

- `PRD.md` — source of truth for scope, schema, contracts. When PRD and code disagree, fix the disagreement before adding new work.
- `HANDOVERS` — one file per session, written at the end (Work cycle Step 9).
- `explorations/` — historical / scratch only. Never implement against it.

## Work cycle (every task)

1. **Understand** — re-read the relevant `docs/PRD.md` section. Confirm scope with the user if ambiguous. *If a Superpower skill (Brainstorm, Specify, etc.) activates, defer to it — see Superpower interop below.*
2. **Plan** — state the approach in 2–4 bullets. **Last HITL checkpoint.** After user signs off, run Steps 3–9 without pausing for confirmation.
3. **Implement** — write the code. Use the subagent strategy below.
4. **Test** — run the tests; show output; fix until green.
5. **Eval (if agent-layer-touching)** — run the relevant eval; record before/after.
6. **Clean up** — delete dead, legacy, or redundant code rather than leaving it. *(Narrow exception to no-HITL: if unsure whether something is still load-bearing, ask before deleting. Silent deletions break invariants.)*
7. **Update docs** — PRD and/or README per the doc layout above.
8. **Commit** — conventional message; no push.
9. **Handover** — write `docs/handovers/YYYY-MM-DD-<slug>.md` with: what shipped, what's left, gotchas, open questions.
10. **Recommend** — 3 next-step options with a pick.

## Subagent strategy (Step 3)

Default to fanning out. Sequential is the exception.

- **Decompose** the plan into independent units — one subagent per file, per layer, or per concern, whichever cleaves cleanly. Look for seams where work can run in parallel without shared mutable state.
- **Dispatch** subagents in parallel. Each gets: its scope, the relevant PRD section, the explicit "do not touch X" boundary, and the acceptance criteria for its slice.
- **Integrate** the results yourself. Resolve interface mismatches, ensure contracts line up, run the full test suite across the union — not just the per-subagent tests.
- **Don't fan out** for single-file edits, trivial changes, or work where subtasks share heavy state (most refactors). Fanout has overhead; spend it when it saves real wall-clock time.

## HITL boundary

- **Before Plan:** ask anything needed. Superpower owns this phase if it activates.
- **At Plan:** show the plan. User's last chance to redirect.
- **Steps 3–9:** automatic. No "shall I proceed?" between steps. Show outputs as you go; don't narrate transitions.
- **Pause mid-cycle only for:** a clean-up deletion you're unsure about (Step 6), a test failure you genuinely can't resolve, or discovering the plan was wrong and needs to change shape. In all three cases, surface the issue concisely and wait.

## Superpower interop

If a Superpower skill fires during Understand (Brainstorm, Specify, Plan-the-spec, etc.), let it run end-to-end. Its output spec becomes the input to Step 2 of the work cycle — do not re-do clarification or re-write the spec. Once the spec exists, jump straight to the Plan step and continue from there under the no-HITL rule.

If no Superpower skill fires, run Step 1 yourself: re-read the PRD, ask clarifying questions if needed, then proceed to Plan.

## Sibling boundaries

A folder's `CLAUDE.md` governs only code inside that folder. Don't reach across siblings — backend code doesn't import from frontend, and vice versa. Cross-folder changes are planned at root and split into per-folder subagents during Step 3; they're not handled inside a single sibling.

---

**Healthy when:** every task ends with a handover, every commit ties to a PRD section, and the next session can pick up cold from the latest handover file without asking what happened.