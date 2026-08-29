---
name: sspm-net-complex-ri-state
description: "State of the complex-data (RI/MERLIN) development work on SSPM-Net — branch, findings, recommended config, open items"
metadata: 
  node_type: memory
  type: project
  originSessionId: 640618ad-8457-4d1a-9ad3-8ef5881117f1
  modified: 2026-08-29T22:58:57.745Z
---

Work done 2026-08-29 on branch `feature/complex-ri-merlin` (commit 89a8f3e; main untouched, NOT pushed to GitHub yet — no gh auth on the machine).

**What was built:** `sspmnet/complex_data.py` (quad-pol TIFF loader from data/tiff, |Re|/|Im| pseudo-amplitude pair, L1 conditional-median calibration ×1.7456), trainer `ri_mode="targets"|"merlin"`, `merlin_loss="l1"|"nll"`, guided TV (`guide_edge_weights`) + Lee-style CV gate (`guide_cv_protect`), `scripts/compare_ri.py`.

**Recommended config (won 5/6 synthetic-GT metrics, validated on real data):** `TrainConfig(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0, guide_cv_protect=0.3)` with `load_quadpol_tiffs("data/tiff")`. True EPI 0.81→0.83, ENL-ROI +12–27% over baseline, ratio-ENL(HV) 0.75 vs baseline's 0.36.

**Key findings (for the thesis):**
- amp² is a sufficient statistic for reflectivity: RI decomposition adds supervision density (100% of pixels vs ~30% masked), not information; single-channel phase is uniform (no per-pixel info). Cross-channel phase (HH–VV CPD) DOES carry scattering-type signal (concentration 0.26 on structures vs 0.09 distributed) but the uint8 phase files are folded to [0,π] and inconsistent with re/im — too degraded to exploit; needs signed float SLC.
- Noisy-reference EPI/SSIM are misleading: vs the KNOWN clean, EPI rises 0.65→0.81+. Fair protocol = ratio-image ENL + synthetic-GT (scripts in scratchpad were used; results under results/ri_compare/, gitignored).
- `merlin_loss="nll"` collapses dark cross-pol channels to zero (black HV) — documented, default stays "l1".
- MERLIN output sits on the median convention (~13% darker channel means); corr/ENL/EPI are scale-invariant, MAD/RMSE are not — report scale-normalized MAD.

**Open items:** push branch (needs `gh auth login`); optional flat/rural patch test; try to obtain signed float SLC for the C3 covariance route (future work). CAUTION: this is a Colab VM — results/ri_compare/*.npy+png and this memory die with the runtime; only the pushed git branch is durable.
