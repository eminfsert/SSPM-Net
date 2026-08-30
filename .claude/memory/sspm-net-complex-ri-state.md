---
name: sspm-net-complex-ri-state
description: "State of the complex-data (RI/MERLIN) development work on SSPM-Net — branch, findings, recommended config, open items"
metadata: 
  node_type: memory
  type: project
  originSessionId: 640618ad-8457-4d1a-9ad3-8ef5881117f1
  modified: 2026-08-30T00:00:00.000Z
---

Work done 2026-08-29 on branch `feature/complex-ri-merlin`; main untouched. Branch IS pushed to GitHub (origin/feature/complex-ri-merlin, HEAD fe0a612 as of 2026-08-30). NOTE: the clone's fetch refspec may be main-only — run `git fetch origin 'refs/heads/*:refs/remotes/origin/*'` on a fresh VM to see the feature branch.

**Phase-feedback work (2026-08-30, same branch):** `sspmnet/phase_data.py` — the folded [0,π] uint8 phase files DO carry usable cross-channel evidence via the doubled angle 2φ: HV–VH phase agreement = per-pixel SNR map (reciprocity: shared speckle, thermal noise breaks phase agreement; mean coherence 0.655, ~0 on water/roads). Trainer knobs: `phase_smooth_boost` (TV/NLM ×(1+b·(1−snr))), `phase_fidelity` (cross-pol data term ×(1−f·(1−snr))), plus untested-by-default `phase_surface_boost`/`phase_protect`. Winner+PH(b=3,f=0.5): synthetic-GT ENL-ROI(HV) +122%, ENL-ROI(HH) +14% at equal PSNR, true EPI(HV) 0.768→0.782; b=5 → ENL(HV) +177% for −0.1dB; b=8 degrades. Real urban patch: ENL-ROI texture-saturated (small gains), ENLr(HV) drops (fidelity pushes noise floor down — intended), corr(HV,VH) up. Synthetic protocol now simulates reciprocity (shared xpol speckle + σ_n bisection-calibrated to real coherence 0.655). Reproduce: `python scripts/compare_ri.py --merlin --phase`.

**Residual-speckle refinement (2026-08-30, later session):** user found outputs still grainy. Added: (1) `whiteness_lambda` ratio-whiteness loss — penalize small-lag autocorrelation of noisy²/output² ratio; improves EVERY GT accuracy metric (+0.12/+0.21 dB PSNR, EPI up both channels); real data is oversampled (speckle lag-1 autocorr ~0.5) so use `whiteness_lags=(3,4,5)` there, (1,2,3) on white simulated speckle. (2) `polish` final guided-NLM pass on the output (guide=output, CV+det protected) — near-free ENL multiplier (GT ENL-ROI(HV) 140→193 at s=0.5 alone). Combined recommended: PH-b3 + whiteness 0.05 + polish 0.5 → GT ENL-ROI 141/163 (HH/HV) vs baseline ~100/62, EPI 0.827/0.791. (3) `nl_self_refresh` (self-referential NLM reference) HURT all metrics — negative result, default 0. Repro: `compare_ri.py --merlin --phase` (includes refinements).

**What was built:** `sspmnet/complex_data.py` (quad-pol TIFF loader from data/tiff, |Re|/|Im| pseudo-amplitude pair, L1 conditional-median calibration ×1.7456), trainer `ri_mode="targets"|"merlin"`, `merlin_loss="l1"|"nll"`, guided TV (`guide_edge_weights`) + Lee-style CV gate (`guide_cv_protect`), `scripts/compare_ri.py`.

**Recommended config (won 5/6 synthetic-GT metrics, validated on real data):** `TrainConfig(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0, guide_cv_protect=0.3)` with `load_quadpol_tiffs("data/tiff")`. True EPI 0.81→0.83, ENL-ROI +12–27% over baseline, ratio-ENL(HV) 0.75 vs baseline's 0.36.

**Key findings (for the thesis):**
- amp² is a sufficient statistic for reflectivity: RI decomposition adds supervision density (100% of pixels vs ~30% masked), not information; single-channel phase is uniform (no per-pixel info). Cross-channel phase (HH–VV CPD) DOES carry scattering-type signal (concentration 0.26 on structures vs 0.09 distributed) but the uint8 phase files are folded to [0,π] and inconsistent with re/im — too degraded to exploit; needs signed float SLC.
- Noisy-reference EPI/SSIM are misleading: vs the KNOWN clean, EPI rises 0.65→0.81+. Fair protocol = ratio-image ENL + synthetic-GT (scripts in scratchpad were used; results under results/ri_compare/, gitignored).
- `merlin_loss="nll"` collapses dark cross-pol channels to zero (black HV) — documented, default stays "l1".
- MERLIN output sits on the median convention (~13% darker channel means); corr/ENL/EPI are scale-invariant, MAD/RMSE are not — report scale-normalized MAD.

**Open items:** optional flat/rural patch test; try to obtain signed float SLC for the C3 covariance route (future work). CAUTION: this is a Colab VM — results/ri_compare/*.npy+png and this memory die with the runtime; only the pushed git branch is durable.
