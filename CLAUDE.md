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
                               phase_smooth_boost=3.0, phase_fidelity=0.5,
                               whiteness_lambda=0.05, whiteness_lags=(3, 4, 5),
                               polish=0.5, edge_boost=1.0),
              ri_pair=ri, pha=pha)
```

(whiteness_lags=(1,2,3) on simulated white speckle; (3,4,5) on the real
patch — its speckle lag-1 autocorrelation is ~0.5 from oversampling.)

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

- Residual-speckle refinement (2026-08-30, later session): ratio-whiteness
  loss (`whiteness_lambda=0.05`) improves EVERY GT accuracy metric
  (PSNR +0.12/+0.21 dB, EPI(HH/HV) 0.823/0.785→0.829/0.793) — the best
  single addition since MERLIN; final non-local polish (`polish=0.5`,
  guided by the output itself, CV+det protected) is a near-free ENL
  multiplier (GT ENL-ROI(HV) 140→163 combined, →193 alone at s=0.5).
  Combined real-patch: ENL-ROI(HH) 1.19→1.26, ENL-ROI(HV) 1.55→1.59,
  corr up, visibly less grain in the zoom crops
  (results/ri_compare/compare_polish_zoom.png). Self-referential NLM
  reference (`nl_self_refresh`) HURT all metrics — negative result,
  default 0.

- Edge sharpening (2026-08-30, same session): user noted edges still soft
  ("a median filter also gives high ENL"). Gradient-matching edge LOSS
  (edge_sharp_lambda) FAILED both with span targets (collapses dark
  cross-pol: PSNR(HV) 8.7) and per-channel 2-look targets (re-injects
  speckle, ENL 163->23) — any noisy guide's gradients poison the match;
  documented negative, default 0. What WORKS: `edge_boost=1.0` —
  edge-masked unsharp of the final output (mask = span log-grad + 0.3x
  phase snr-coherence grad, rational squash, widened; per-channel dark
  gate `edge_boost_dark=0.2` mandatory, else real ENLr(HV) collapses
  0.65->0.18). GT: PSNR(HH) +0.18 dB, EPI 0.8273/0.7911->0.8315/0.7920
  (both up), ENL unchanged; real: noisy-EPI up, ENLr(HH) 0.843->0.866.

- Default-off phase knobs settled (2026-08-31): `phase_protect` and
  `phase_surface_boost` were evaluated on the real patch on top of the full
  stack, each against a MEAN-MATCHED control (the trainer multiplies the TV
  edge weights by the phase factor *without* renormalizing, so a knob that
  shifts the factor's mean is partly just a global `tv_mult` change; only
  the NLM term sees a mean-1 copy, i.e. the map's shape). Both stay 0.
  `phase_protect`: matched or beaten by the plain `tv_mult=7.9` control on
  every metric (EPI(HV) 0.6622 vs 0.6660, ENLr(HV) 0.4669 vs 0.5090) — its
  `det` map is near estimator noise (independent cross-channel agreement
  +0.06..+0.09 vs +0.02 for a random-phase null; the +0.71 HV-VH agreement
  is reciprocity, i.e. the same physical channel, not evidence).
  `phase_surface_boost`: gains are small (ENL-ROI +1.7%/+2.4% at s=1.0),
  reproduced by the `tv_mult=11.3` control, and — decisively — absent in
  the ROIs where the map is high (stratified ENLsurf 1.1376 vs control
  1.1393); plain `final` still wins on ratio-ENL. The `surface` map is a
  valid scene descriptor (ENL ~1.9x higher in its high-coherence ROIs) but
  not a useful regularizer knob on an urban patch. Scripts:
  `scripts/experiments/diag_phase_maps.py`, `diag_det_reliability.py`,
  `run_phase_extra_real.py`; tables in `docs/figures/`.

## Serious-gains work — branch `feature/scene-scale-arch` (2026-08-31)

Approved plan (see `.claude/memory/sspm-net-complex-ri-state.md` for full
state): Track A structural fixes — A1 `dropout_style` ("band" Dropout2d
historically zeroed the whole 1-ch LL band in ~30% of passes / "pixel"),
A2 `norm` ("batch"/"group"; BN ran on batch-1 stats and EMA never updated
buffers), A3 multi-level DWT (`wavelet_levels` now real: scale-recurrent
shared detail CNN, RF 68->136->256+, params unchanged) + optional
`low_freq_dilations`, A4 saturation mask (`load_quadpol_tiffs(...,
return_sat=True)` + `denoise(..., sat=)`; ~10%/channel of uint8 pixels are
clipped — `data/example_quadpol.npy` is the full-range source of the same
pixels). Track B scene-scale zero-shot: user will supply ~16 more 512x512
patches of the SAME scene; batch random-crop training over the patch pool
(still zero-shot: no external data). `TrainConfig.model_cfg` forwards
architecture overrides. Experiments: `scripts/experiments/
run_arch_ablation.py`, `run_a4_sat.py`. Success bar: GT PSNR >= +0.5 dB or
ENL-ROI >= +50% at equal EPI.

- A1/A2 RESULT (2026-08-31): **pixel+group is the new winning base** and
  the first change clearing the serious-gain bar. GT: PSNR +0.64/+1.51 dB
  (HH/HV), SSIM 0.796/0.801->0.834/0.838, EPI(HV) 0.791->0.828, ENL-ROI
  135/192->4630/561 (PSNR+SSIM+EPI rise together => genuine, not blur;
  visual check clean). Real: corr(HV,VH) 0.9924->0.9996, ENLr(HV)
  0.499->1.120 (ideal 1), EPI(HV) 0.656->0.714; urban ENL-ROI lower
  (1.20/1.50->1.01/1.28) — ROI metric disagrees with the ratio metric,
  report both. CAUTION: "pixel" dropout ALONE collapses cross-pol on GT
  (PSNR(HV) 17.8) — per-pixel dropout with batch-1 BatchNorm is unstable;
  only the combination works. Tables/figure in docs/figures/
  metrics_arch_ablation.txt. Subsequent ablations run on the pixel+group
  base (defaults in Config still historical until more validation).

**Open items:** flat/rural patch test (the natural place where
`phase_surface_boost` could still help — an urban patch has little surface
scattering); a dedicated `tv_mult` sweep (ratio-ENL and ENL-ROI disagree
about the operating point: `tv_mult=7.9` gives the best ENLr(HV) 0.509 vs
0.439 at 10, while ENL-ROI prefers 10); obtain signed float SLC; decide on
merging the feature branch.

## Claude memory restore

Copies of the persistent memory files live in `.claude/memory/`. On a fresh
VM (project cloned to /content/SSPM-Net), restore with:

```bash
# the project dir slug follows Claude Code's working directory:
# /content -> -content ; /content/SSPM-Net -> -content-SSPM-Net
mkdir -p /root/.claude/projects/-content/memory
cp .claude/memory/* /root/.claude/projects/-content/memory/
```

Also widen the clone's fetch refspec before looking for the branch (fresh
Colab clones are often main-only):

```bash
git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git fetch origin && git checkout feature/complex-ri-merlin
```

Checkpoint often: the Colab VM is ephemeral, so commit + push code, metric
tables, `CLAUDE.md` and `.claude/memory/` to the feature branch at regular
intervals.
