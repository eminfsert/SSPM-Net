# Experiment scripts (session log, 2026-08-29)

Scripts that produced the complex-data (RI/MERLIN) comparison results, in
chronological order. They assume the repo at /content/SSPM-Net, an earlier
run's outputs under results/ri_compare/, and a CUDA GPU; run from the repo
root. See CLAUDE.md for the findings they support.

1. run_tuned.py        — +RI(targets) with stronger TV (vs saved baseline)
2. run_merlin.py       — first MERLIN input-separation run
3. run_merlin_v2.py    — MERLIN + L1 conditional-median calibration, tv_mult=5
4. run_synthetic_gt.py — ground-truth protocol: known clean + simulated
                         1-look complex speckle -> true PSNR/SSIM/EPI
5. run_synth_v3.py     — round 3 on synthetic GT: NLL loss (collapses on
                         cross-pol) and the winning L1 + CV-gate variant
6. run_real_v3.py      — winner (ri_mode="merlin", tv_mult=10,
                         guide_cv_protect=0.3) validated on the real patch,
                         incl. ratio-image ENL

## Phase-feedback session (2026-08-30)

7. run_phase_real.py   — baseline / RI winner / winner+PH(defaults) on the
                         real patch; regenerates denoised_baseline.npy
8. run_phase_synth.py  — GT protocol with reciprocity physics (shared
                         cross-pol speckle + calibrated thermal noise);
                         b/f ablation sweep -> b3-f0.5 balanced winner
9. run_phase_synth2.py — stronger boosts b5 / b8 (b8 starts degrading)
10. run_phase_real2.py — b3 / b5 validated on the real patch + figure

## Residual-speckle refinement session (2026-08-30, later)

11. run_polish_synth.py — GT test of self-ref NLM (A: hurts, off by
                          default), final NL polish (B: big ENL, tiny cost)
                          and ratio-whiteness loss (C: best accuracy gain);
                          C+B is the combined winner
12. run_polish_real.py  — real-patch validation of PH-b3 + C(lags 3-5) +
                          B(0.5), incl. zoom-crop figure
