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

**Recommended config** (validated winner):

```python
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs
amp, ri = load_quadpol_tiffs("data/tiff")           # L1-calibrated |Re|/|Im|
res = denoise(amp, TrainConfig(iters=700, ri_mode="merlin",
                               tv_mult=10.0, guide_cv_protect=0.3),
              ri_pair=ri)
```

vs. the amplitude-only baseline: true EPI (vs clean) 0.81→0.83, best SSIM,
PSNR(HH) +0.24 dB, ENL-ROI +12–27%, ratio-ENL(HV) 0.75 vs 0.36.
Reproduce with `python scripts/compare_ri.py --merlin`.

**Key findings (thesis-relevant):**

- amp² is a sufficient statistic for reflectivity: the RI decomposition adds
  supervision *density* (full-pixel Noise2Noise without masking), not
  information; GT-PSNR ceiling is unchanged.
- Single-channel phase is uniform (no per-pixel info). Cross-channel HH–VV
  phase difference DOES carry scattering-type signal (concentration 0.26 on
  structures vs 0.09 distributed), but these uint8 phase files are folded to
  [0, π] and inconsistent with the re/im files — unusable here. Signed float
  SLC would enable the C3-covariance route (future work).
- Noisy-reference EPI/SSIM are misleading (they reward similarity to speckle
  gradients); measured against a known clean, EPI *rises* 0.65→0.81+. Fair
  protocol = ratio-image ENL (ideal ≈ 1) + synthetic-GT PSNR/SSIM/EPI.
- `merlin_loss="nll"` (MERLIN's Gaussian NLL) collapses dark cross-pol
  channels to zero — keep the default `"l1"`.
- MERLIN outputs sit on the L1/median scale convention (~13% darker channel
  means); corr/ENL/EPI are scale-invariant, MAD/RMSE are not — report
  scale-normalized MAD.

**Open items:** flat/rural patch test; obtain signed float SLC; decide on
merging the feature branch.

## Claude memory restore

Copies of the persistent memory files live in `.claude/memory/`. On a fresh
VM (project cloned to /content/SSPM-Net), restore with:

```bash
mkdir -p /root/.claude/projects/-content-SSPM-Net/memory
cp .claude/memory/* /root/.claude/projects/-content-SSPM-Net/memory/
```
