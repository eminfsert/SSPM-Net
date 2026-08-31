---
name: colab-checkpoint-commits
description: "Commit and push work to the feature branch at regular intervals — the Colab runtime loses everything unpushed"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-08-31T00:00:00.000Z
---

The user works from an ephemeral Google Colab VM and asked that work be committed at
regular intervals ("çalışmaları belli bir aralıkla kaybetmeyelim") — code, results
worth keeping, the session context (CLAUDE.md) and the memory mirror in
`.claude/memory/` all go to the `feature/*` branch, never main.

**Why:** when the Colab terminal/runtime dies, anything not pushed to GitHub is gone;
only the pushed branch is durable.

**How to apply:** after each meaningful step (a working experiment, a metrics table, a
new trainer knob) commit to `feature/complex-ri-merlin` and push. Keep `CLAUDE.md`
and `.claude/memory/*` in sync in the same commit so the next session can restore.
Sole author is the thesis author — no Claude trailers. See [[work-on-branches]] and
[[sspm-net-complex-ri-state]].
