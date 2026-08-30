# CLAUDE.md — project context for Claude Code sessions

## What this repo is

Demo/reference code for the M.Sc. thesis *"Application of Zero-Shot
Self-Supervised Speckle Denoising Techniques in Polarimetric SAR Imagery"*
(SSPM-Net). `main` is the published amplitude-only demo; experimental work
lives on feature branches.

## Working conventions

- Commit experimental work to `feature/*` branches, never directly to `main`.
- Do not add Claude co-author trailers to commit messages; the thesis author
  is the sole commit author.
- Report metrics honestly, including negative results; always produce the
  side-by-side comparison figure along with any metrics table.
- Environment is usually a Google Colab A100 VM (ephemeral!): anything not
  pushed to GitHub is lost on runtime reset. `results/` is gitignored.

## State of the complex-data (RI / MERLIN) work — branch `feature/complex-ri-merlin`

Extends the pipeline with the real/imaginary SLC components (`data/tiff/`,
uint8 per-component TIFFs: amp/real/imgy/pha × HH/HV/VH/VV).

**Recommended config** (validated winner, incl. phase feedback 2026-08-30):

```python
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase
amp, ri = load_quadpol_tiffs("data/tiff")           # L1-calibrated |Re|/|Im|
pha = load_quadpol_phase("data/tiff")               # folded [0,pi] phase
res = denoise(amp, TrainConfig(iters=700, ri_mode="merlin",
                               tv_mult=10.0, guide_cv_protect=0.3,
                               phase_smooth_boost=3.0, phase_fidelity=0.5),
              ri_pair=ri, pha=pha)
```

RI winner vs. the amplitude-only baseline: true EPI (vs clean) 0.81→0.83,
best SSIM, PSNR(HH) +0.24 dB, ENL-ROI +12–27%, ratio-ENL(HV) 0.75 vs 0.36.
Phase feedback (b=3, f=0.5) on top of the RI winner (synthetic-GT protocol
with reciprocity physics): ENL-ROI(HV) +122%, ENL-ROI(HH) +14% at equal
PSNR; true EPI(HV) 0.768→0.782; b=5 gives ENL-ROI(HV) +177% for ~0.1 dB
PSNR(HV); b=8 starts degrading accuracy. Reproduce with
`python scripts/compare_ri.py --merlin --phase`.

**Key findings (thesis-relevant):**

- amp² is a sufficient statistic for reflectivity: the RI decomposition adds
  supervision *density* (full-pixel Noise2Noise without masking), not
  information; GT-PSNR ceiling is unchanged.
- Single-channel phase is uniform as a VALUE, but cross-channel phase
  carries per-pixel evidence that survives the [0, π] uint8 fold via the
  doubled angle 2φ: the HV–VH phase agreement (reciprocity: shared speckle,
  independent thermal noise) is a per-pixel SNR / "is this value noise?"
  map (mean coherence 0.655 on the patch; ~0 on water/roads, high on
  vegetation — corr 0.64 with HV brightness). `sspmnet/phase_data.py`
  builds it; the trainer boosts TV/NLM where noise-dominated
  (`phase_smooth_boost`) and down-weights the cross-pol data term there
  (`phase_fidelity`). HH–VV coherence (surface scattering) and spatial
  phase coherence (deterministic targets) are also computed
  (`phase_surface_boost`, `phase_protect`) but off by default — not tested
  on synthetic (their physics isn't simulated) and no real-data win shown
  yet. Signed float SLC would still enable the full C3-covariance route
  (future work).
- Real-patch ENL-ROI is texture-saturated (~1.2–1.6 in an urban scene), so
  the large phase-feedback ENL gains show on the synthetic-GT protocol,
  not the urban patch (+0.4–4% there); real-data ratio-ENL(HV) drops
  0.74→0.63 as the fidelity term deliberately pushes noise-floor pixels
  down. corr(HV,VH) improves 0.9932→0.9934/0.9938. Report both honestly.
- The synthetic-GT protocol was extended with reciprocity physics: shared
  cross-pol speckle + i.i.d. thermal noise, σ_n calibrated (bisection) so
  the simulated HV–VH doubled-angle coherence matches the real patch's
  0.655 — the phase feedback is then validated through the same code path
  it uses on real data (`phase_feedback_maps(z=...)`).
- Noisy-reference EPI/SSIM are misleading (they reward similarity to speckle
  gradients); measured against a known clean, EPI *rises* 0.65→0.81+. Fair
  protocol = ratio-image ENL (ideal ≈ 1) + synthetic-GT PSNR/SSIM/EPI.
- `merlin_loss="nll"` (MERLIN's Gaussian NLL) collapses dark cross-pol
  channels to zero — keep the default `"l1"`.
- MERLIN outputs sit on the L1/median scale convention (~13% darker channel
  means); corr/ENL/EPI are scale-invariant, MAD/RMSE are not — report
  scale-normalized MAD.

**Open items:** flat/rural patch test (also the natural place where
`phase_surface_boost` could help); real-data value of `phase_protect` /
`phase_surface_boost`; obtain signed float SLC; decide on merging the
feature branch.

## Claude memory restore

Copies of the persistent memory files live in `.claude/memory/`. On a fresh
VM (project cloned to /content/SSPM-Net), restore with:

```bash
mkdir -p /root/.claude/projects/-content-SSPM-Net/memory
cp .claude/memory/* /root/.claude/projects/-content-SSPM-Net/memory/
```
