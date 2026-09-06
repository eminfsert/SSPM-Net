---
name: short-experiments
description: "Keep steps simple and experiments short; ask before any run longer than ~10 minutes"
metadata:
  type: feedback
  modified: 2026-09-06T00:00:00.000Z
---

The user asked for simple, small steps and no long unattended test runs ("Please make it simple steps and try to not to test it like 30-40 min. Or ask me for that"), after earlier sessions chained 30–40 min experiment batteries.

**Why:** the Colab VM is ephemeral and the user wants to steer between steps; long batteries burned time on marginal knobs (Tracks C/D/E).

**How to apply:** one smoke run or 1–2 short real-patch runs per step, report, then continue. Ask the user before launching anything expected to take more than ~10 minutes (e.g. multi-row GT protocols). See [[sspm-net-complex-ri-state]] and [[colab-checkpoint-commits]].
