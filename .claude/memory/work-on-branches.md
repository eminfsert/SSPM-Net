---
name: work-on-branches
description: "Commit experimental work to feature branches, never directly to main"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 640618ad-8457-4d1a-9ad3-8ef5881117f1
  modified: 2026-08-29T23:14:31.659Z
---

The user asked for new development to go to a separate branch first, not directly to main ("direkt olarak master'ı atmayalım").

**Why:** main is the published demo accompanying the thesis; experimental extensions must not destabilize it.

**How to apply:** create/use a `feature/*` branch for changes in [[sspm-net-complex-ri-state]]-style work, keep main clean, and remind the user to push (gh auth is usually missing on the Colab VM). The user also asked that commits carry no Claude co-author trailer — the thesis author is the sole commit author.
