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

## !! PHASE DECODING FIX (2026-09-05) — supersedes every "phase is unusable" claim below

The bundled `data/tiff/*_pha.tiff` files carry the **full-range [0, 2*pi) SLC
phase**. They are NOT folded to [0, pi]. Sessions up to 2026-09-02 decoded them
as `pha/255*pi`, which correlates ~0.02 with the quadrant angle implied by
|Re|,|Im| — and that artefact produced the whole "the phase files are pixel-wise
inconsistent with the components, only the doubled angle 2*phi survives"
conclusion. **That conclusion was a decoding bug, not a property of the data.**

Correct decoding (convention: `|Re| ~ |sin phi|`, `|Im| ~ |cos phi|`; the global
90-degree rotation is identical in all four channels and cancels in every
coherence):

```python
phi = pha_uint8 / 255.0 * 2 * np.pi
S   = amp * np.exp(1j * phi)          # complex SLC, recoverable TODAY
```

Evidence — `scripts/experiments/diag_phase_decoding.py`, table
`docs/figures/diag_phase_decoding.txt`:

- `corr(log|Im/Re|, ratio implied by phi)` = **+0.994** in all four channels
  (the [0,pi] decoding gives 0.01–0.03).
- Phase **quadrants are uniformly occupied (~25% each)** -> the sign
  information is present, the phase is genuinely full-range.
- |Re| and |Im| reconstructed from `amp` + `phi` alone: **corr 0.996**.
- Physics of the recovered SLC: HV–VH single-angle complex coherence **0.772**
  (random-phase null 0.185; the old doubled-angle route gave 0.639 — doubling
  the angle also doubles the phase noise), HH–VV 0.348, spatial lag-1 0.698
  decaying to 0.321 at lag 2 (the real SAR oversampling signature).

**What this does and does NOT buy (measured, corrected 2026-09-05):** the
obvious payoff does NOT materialise, and an earlier claim in this file that it
did was wrong. HV and VH carry a constant **+140.25 deg** relative phase
offset; combining them without removing it is destructive interference (the
mean collapses to 0.44x), which inflates ENL as an artefact — that artefact is
what the first version of this note reported as "+14% ENL, -29% thermal". With
the offset properly removed, HV-VH global coherence is **0.83**: reciprocity
means the two channels are largely the SAME complex sample, so their speckle is
*shared* and coherent averaging cannot reduce it. Aligned coherent fusion gives
ENL **0.564** — worse than plain intensity averaging (**0.650**) and worse than
today's incoherent amplitude average (0.602). The only real effect is on the
independent thermal floor (-12% intensity in thermal-dominated areas). The
corrected single-angle SNR map is less noisy than the old doubled-angle one
(mean 0.772 vs 0.639) but a *worse* discriminator (bright/dark contrast 1.55 vs
1.75), and the signed HH-VV CPD is near its null on this urban patch
(concentration 0.084). **There is no quick win here.** The value of the fix is
that the C3/T3 covariance route and genuinely coherent methods are now
*possible* on the existing data, and that every prior phase conclusion rested
on a false premise.

**Consequence for the record below:** every phase knob in Tracks C and D
(`phase_smooth_boost`, `phase_fidelity`, `phase_protect`,
`phase_surface_boost`, `phase_helix_protect`, `fact_snr_gate`, and D2
`xpol_snr_input`) was built on the lossy doubled-angle maps. Their negative /
marginal results are explained by that and should not be read as "phase does
not help".

**Phase is NOT fed to the network today**: `Config.xpol_snr_input = False`
(`sspmnet/config.py:74`) -> `aux_in = None` (`sspmnet/trainer.py:418`) ->
`SSPMNet.forward` only concatenates the aux plane when the flag is on
(`sspmnet/model.py:193`). Phase enters solely as hand-set loss weights
(`phase_smooth_boost=1.5`, `phase_fidelity=0.5`), and only when `pha=` is
passed (`trainer.py:359`).

**Direction:** Track F (coherent SLC) is scoped down — the cheap coherent wins
are measured and negative; what remains is the C3/T3 covariance route, which is
real but substantial work. Track E stays the larger *measured* headroom and is
what was run next (see the Track E1 result below). Track E's "ask the user for
signed float SLC" item is dropped either way — the complex data is already here.

## !! INDEPENDENT RE-EVALUATION (2026-09-05) — read before trusting any GT number below

Every synthetic-GT number in this file up to 2026-09-05 was measured against a
"clean proxy" that was itself an output of this pipeline:

```python
clean = denoise(amp, TrainConfig(iters=700))["denoised"]   # circular
```

Measured consequence: the proxy carries only **46%** of the input's fine
texture (high-pass std / mean: input 0.607, proxy 0.277), so resembling it
rewards smoothing. `sspmnet/phantom.py` + `scripts/experiments/run_indep_eval.py`
re-score the headline claims against a reflectivity field built from scratch —
no network, no real pixels in the reference. Table
`docs/figures/metrics_indep_eval.txt`, figure `compare_indep_eval.png`.

| claim | claimed (circular GT) | independent GT |
|---|---|---|
| A1/A2 `pixel+group` | +0.64 / +1.51 dB | **+0.605 / +0.692 dB — HOLDS** (HV was ~2x inflated) |
| C1 `xpol_pair_input` | +0.13 dB (HV) | **+0.758 dB (HV), EPI(HV) +0.046 — BIGGER than claimed** |
| E1 `sat_censored` | +0.85 / +0.63 dB | **+0.051 / +0.050 dB, EPI(HV) −0.016 — DOES NOT REPLICATE** |
| baseline -> full stack | — | **+1.399 / +2.799 dB, EPI(HV) +0.189** |

**E1 is retracted as a win.** Its headline gain was a protocol artefact, and the
mechanism is now visible: the E1 output's texture (0.276) sits almost exactly on
the circular proxy's (0.277), i.e. it scored for resembling the reference rather
than for denoising. On the phantom, clipping costs 3.12 dB of PSNR(HV)
(41.529 unclipped -> 38.412 clipped) and E1 recovers 0.05 dB of it — 1.6%,
inside the 0.10 dB single-seed noise floor. The one-sided loss can lift a mildly
clipped smooth tail but cannot reconstruct spiky point targets from censored
information. What survives on the real patch are the scale-invariant,
non-circular metrics only: ENLr(HH) 0.879->0.922, flat-water grain 0.484->0.455,
scale-matched satRatio 0.797->0.847. Small, real, not a dB win. Defaults stay
off.

**C1 is upgraded, with a caveat.** `xpol_pair_input` buys far more accuracy than
the circular protocol could see (+0.758 dB HV, EPI +0.046) — but on the phantom
its flat-band grain DOUBLES (flatHP 2.33 -> 5.16), contradicting the Track C
revision's "no flat grain penalty" (real water CV 0.063 vs 0.068). The phantom's
flat band is exactly constant, so every fluctuation there is pure error; the
real "flat water" has its own texture and was masking the grain. **The real-data
grain measurements in this file are therefore optimistic.** C1 is an
accuracy-vs-flat-grain trade, not a free win — report it as such.

**A1/A2 holds** and is the one genuine structural win: PSNR +0.605/+0.692,
SSIM(HV) 0.922->0.958, flat grain 4.18->2.33 (nearly halved).

**Caveat on the phantom itself:** it is one independent reference, not the
truth. Its bright tail is 220 isolated point targets up to 15x q99 (the real
data's max/q99 is 16x, so the range is right, but the density and spikiness are
a design choice) and that character is what drives the E1 verdict. Read it
together with the real patch's scale-invariant metrics, never alone.

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

- A4 RESULT (2026-08-31): NEGATIVE — sat-masking stays off. Matched uint8
  clipping in the GT protocol costs a real -1.7/-1.5 dB (the 8-bit data
  limits the pipeline), but EXCLUDING saturated pixels from the data term
  makes it worse (-1.6 dB more; bright top-1% RMSE 56.5->74.6): a clipped
  target is a biased-but-informative lower bound, no target lets the
  regularizers flatten bright structures (real ENL-ROI jump 1.20->1.80 is
  exactly that flattening, EPI(HV) 0.656->0.583). Full-range .npy
  amplitude breaks the q99/clip(0,5) normalization outright (documented).
  Future fix idea: censored one-sided loss (target >= clip) or log-domain
  training. Table: docs/figures/metrics_a4_sat.txt.

- A3 RESULT (2026-08-31): MARGINAL — `wavelet_levels` stays 1. On the new
  pixel+group base, RF expansion no longer pays: GT accuracy slips
  monotonically with depth (PSNR(HH) 24.48->24.23->24.02, SSIM/EPI too;
  the GT ENL(HV) rise 561->2478 is no longer backed by accuracy, unlike
  the A1/A2 jump). Real patch: small consistent gains (ENL-ROI(HV)
  1.28->1.39, EPI(HH) 0.677->0.689 at lv3) — below the bar. Dilations add
  nothing. Re-check lv2 once during scene-scale training (256 px crops cut
  context). Table: docs/figures/metrics_a3_rf.txt.

- tv_mult sweep SETTLED (2026-09-01, pixel+group base): the old-base
  disagreement is gone — the operating point is a flat plateau (0.16 dB
  total PSNR(HH) spread over tv 5..15, PSNR(HV) flat). GT accuracy and
  real EPI/ENLr(HH) mildly prefer tv5; ENL-ROI and ENLr(HV) mildly prefer
  high tv. `tv_mult=10` stays the default (mid-plateau); tv5 is the
  max-accuracy point. Table: docs/figures/metrics_tvmult_sweep.txt.

- Track C REVISED (2026-09-01, after the user's visual check): the
  original "winner" claim was overstated — pair+group+debias leaves
  MORE flat-area grain than base (water high-pass std +48%, visible
  CV ~2x; noisy-EPI and ENLr rewarded the grain). What survives:
  `xpol_pair_input=True` alone (all GT metrics up, NO flat grain
  penalty: water CV 0.063 vs base 0.068) — the real Track C win.
  `polgroup_guides` and `thermal_debias` stay off by default (see
  docs/figures/metrics_hv_track.txt REVISION for the full story and
  the untested polgroup fix idea). Lesson: add flat-region high-pass
  std to every ablation table.
  Original result for reference: the HV bar was declared MET.
  Winner: `xpol_pair_input=True` (model_cfg) + `polgroup_guides=True` +
  `thermal_debias=0.5`. C1 (xpol branch sees both reciprocal planes,
  +0.12% params) gives a real HV gain (GT PSNR(HV) +0.13 dB, EPI(HV)
  0.834->0.841, real ENLr(HV) 1.12->1.05); C2 (regularizer guidance split
  by pol group) pays on the real patch (EPI(HV) 0.714->0.757, EPI(HH)
  0.677->0.701); C3 (thermal-floor debias, sigma_th auto-estimated from
  the HV-VH difference) fixes the dark areas: dark-bin relative RMSE
  -15% and real ENLr(HV) lands exactly on the ideal 1.00 at t=0.5
  (t=1.0 helps GT more but overshoots the real ratio metric). Cumulative
  GT PSNR(HV) +0.27..+0.34 dB. Negatives: merlin_recip_weight 0.25/0.75
  flat, phase_fidelity 0.75 slightly worse. Tables/figures:
  docs/figures/metrics_hv_track.txt, compare_hv_winner.png.

- Track D RESULT (2026-09-02): NEGATIVE against the bar — plateau
  confirmed. Knobs (all default off, kept for the record):
  `xpol_fused_target` (+`xpol_fused_loss` "l1"/"l2"), `xpol_target_debias`,
  `model_cfg={"xpol_snr_input": True}` (`SSPMNet.forward(x, aux=)`),
  `phase_helix_protect` (new `helix` map), `fact_snr_gate`
  (`compute_soft_histogram(weight=)`). Single-seed noise floor measured:
  PSNR(HV) 0.10 dB (base@s42 vs @s43). Only consistent gain: target-domain
  thermal debias (D1f+tdb k=1: +0.11 dB, replicates at seed 43, dark bias
  -17%; k=2: +0.20 dB, bias -40%) — but real water mean drops and grain
  contrast rises (CV 0.097->0.118 at k=1), the post-hoc-debias trade-off
  in milder form; use only for radiometric deliverables. L1 fusion alone
  is negative (correlation-dependent median shift; "l2" fixes the bias,
  gain inside noise). D2 -0.31 dB (over-smooths low-snr areas; cleanest
  real flats but EPI down), D3 neutral, D4 negative, combos worse. Full
  table + verdict: docs/figures/metrics_hv_phase.txt, compare_hv_phase.png,
  phase_helix_map.png. Plan: docs/plans/track-d-hv-phase-plan.md.

- NEXT (resume point): **Track E — full-range data + log-domain
  training**, plan in `docs/plans/track-e-fullrange-logdomain-plan.md`.
  Order: E1 censored one-sided loss on saturated targets (existing data,
  small change) -> E2 `domain="log"` pipeline (full-range
  `data/example_quadpol.npy`, log-Rayleigh histogram) -> E3/Track B when
  the user supplies full-range Re/Im and the ~16 scene patches. Bar:
  recover >= +1.0 dB of the measured -1.7 dB clipping cost on GT at equal
  EPI, no flat grain increase.

- Track E1 RESULT (2026-09-05): **RETRACTED — see the independent
  re-evaluation section at the top of this file. The dB gain below was measured
  against the circular proxy and does NOT replicate (+0.05 dB, EPI(HV) −0.016
  on an independent ground truth). Kept for the record; defaults stay off.**
  Original write-up: POSITIVE, but short of the +1.0 dB bar.
  The censored one-sided data term (`TrainConfig.sat_censored`) recovers a
  large part of the 8-bit clipping cost that A4 measured. Protocol identical
  to A4's matched-uint8-clipping GT leg, re-run on the current pixel+group +
  xpol_pair_input base; A4's negative control reproduces exactly
  (`clip+sat` -1.80/-1.48 dB), so the protocol is sound.

  Clipping cost this session: HH -1.652 dB, HV -0.771 dB. Recovered:

  | variant | HH | HV |
  |---|---|---|
  | `clip+sat` (A4, drop saturated) | -1.804 dB | -1.475 dB |
  | `clip+cens` (E1) | **+0.824 dB** | **+0.594 dB** |
  | `clip+cens+tv.5` | **+0.853 dB (52%)** | **+0.625 dB (81%)** |
  | `clip+cens+tv1` | +0.770 dB | +0.612 dB |

  What makes this credible rather than a metric artefact: EVERY accuracy
  metric moves together. GT EPI(HH) 0.774->0.817, EPI(HV) 0.804->0.830,
  SSIM(HV) 0.813->0.833, bright-top-1% RMSE 49.9->41.0, brightR (mean
  output/clean over the top-1% clean pixels, the flattening detector)
  0.777->0.821. And the mandatory flat-water grain column from the Track C
  lesson goes DOWN, not up: waterHP 0.4213->0.3688 (below the unclipped
  ceiling's 0.3718). GT ENL(HV) falls 1486->1380 — honest: the high ENL of
  `clip`/`clip+sat` came from flattening bright structure (brightR 0.677).

  Real patch agrees: EPI(HH) 0.667->0.717, EPI(HV) 0.721->0.754, ENLr(HH)
  0.879->0.922 (toward the ideal 1), satRatio (median output/input where HV
  itself is clipped, LS scale-matched) 0.797->0.847, waterHP 0.484->0.455,
  waterCV 0.097->0.090.

  CAUTION on `satRatio`: it is scale-SENSITIVE and must be measured after
  scale-matching. The raw ratios (base 0.686 -> E1 0.763) partly just report
  which output sits lower overall; the amplitude-only baseline scores 0.553
  raw but 0.813 matched, i.e. it is NOT worse than the current stack on the
  bright tail once the scales are aligned. Both scripts now match first.

  **Verdict against the plan's bar (recover >= +1.0 dB in BOTH channels,
  equal EPI, no grain increase):** EPI and grain criteria are MET and
  exceeded; the dB criterion is NOT met (52% / 81% of the cost). The
  real-patch `satRatio >= 0.9` sub-criterion is also not met (0.847
  scale-matched, up from 0.797). Still the first genuine multi-metric win since A1/A2, and ~8x the
  measured 0.10 dB single-seed noise floor.

  **Recommended:** `sat_censored=True, sat_tv_relax=0.5` with
  `sat=` from `load_quadpol_tiffs(..., return_sat=True)`. `sat_tv_relax`
  adds nothing in PSNR (+0.03 dB, inside noise) but gives the best EPI and
  the lowest flat-water grain; 1.0 overshoots (RMSEbright 41.0->43.1).
  Defaults in `TrainConfig` left off pending the E2 comparison.
  Table `docs/figures/metrics_track_e.txt`, figure
  `docs/figures/compare_track_e.png`, script
  `scripts/experiments/run_track_e.py`.

  **Next: E2 log-domain.** The remaining ~half of the clipping cost is the
  information the 8-bit file simply does not contain; E2 attacks it from the
  other side with the full-range `data/example_quadpol.npy` and a log-domain
  pipeline (`domain="log"`), where the heavy bright tail is representable at
  all. E1 and E2 are complementary, not alternatives.

**Open items:** flat/rural patch test (the natural place where
`phase_surface_boost` could still help — an urban patch has little surface
scattering); obtain signed float SLC; decide on merging the feature
branch; Track B awaits the user's ~16 scene patches in `data/scene/`.

## Track W — phase differences answered; spectral whitening / sub-look N2N (2026-09-06)

Question: 4 phases -> 3 phase DIFFERENCES; what do they mean, do they carry
speckle information? Measured (plan + numbers in
`docs/plans/track-w-spectral-whitening-plan.md`):

- **Inter-channel differences carry NO usable speckle information.** Speckle
  (the random scatterer arrangement) is common to all channels and CANCELS in
  a phase difference; what remains is thermal noise (HV-VH, coherence 0.77)
  and the scattering mechanism (HH-VV CPD bimodal at +-180 on bright
  pixels; HH-HV helix rising with brightness). Fisher analysis: the complex
  HV/VH pair gives only 11% more information about the clean intensity than
  the amplitude pair, which the pipeline already uses. HH-VV / HH-HV maps are
  genuine SCENE descriptors not in the amplitudes (R^2 0.10-0.14) -> W3
  (NLM similarity feature), not done.
- **The speckle information is in the SPATIAL phase (complex spectrum).** The
  SLC fills ~50% of Nyquist (2x oversampling): lag-1 normalised-speckle
  correlation 0.60; 27% (tiff) / 55% (unclipped) of a pixel's log-speckle is
  linearly predictable from its 8 neighbours — a direct violation of the
  blind-spot independence assumption. Spectrum centroid +0.022 cyc/px makes
  corr(Re, Im_neighbour) = 0.09 (MERLIN leak); centring fixes it.
  uint8 clipping puts 28% of the power out of band (unclipped npy SLC: 2.5%).
- **W1 spectral whitening (flatten + 2x decimate) = NEGATIVE**: the residual
  lag-1 0.21 is absorbed wholesale into the output (mottling, blur, haloes).
- **Unclipped centred SLC input alone (`load_quadpol_slc(...,
  amp_npy="data/example_quadpol.npy")` + `denoise(slc=)`) beats the clipped
  TIFF base on every real-patch column** (EPI 0.42/0.44 -> 0.74/0.72,
  ENLr(HV) 0.17 -> 0.96, waterCV 0.097 -> 0.088). New base for complex work.
- **W2 sub-look Noise2Noise (`TrainConfig.sublook_n2n=1.0`, every other
  MERLIN step uses |Re A|/|Im B| of the two disjoint half-bands) = POSITIVE
  vs its control at equal EPI**: ENL-ROI(HV) 0.99 -> 2.51, waterHP -32%,
  waterCV -19%, more of the correlated speckle left in the ratio image.
  ENLr(HV) 0.96 -> 0.84 moves the wrong way.
  **Independent GT (shaped speckle, `sspmnet/phantom.py::make_phantom_slc`
  with the real transfer function, HV-VH coherence matched to 0.78,
  simulated lag-1 0.54/0.39 vs real 0.62/0.48): W2 does NOT clear the bar** —
  dPSNR -0.05/-0.08 dB (inside the 0.10 dB floor), EPI(HV) -0.011, SSIM
  -0.002; ENL(HV) 0.58 -> 0.97 and flatHP -14%. A smoothing trade, not an
  accuracy gain; the real-patch effect is the same trade amplified by scene
  texture. The ratio-whiteness columns barely move, i.e. the extra
  independent supervision does not stop the low-frequency part of the
  correlated speckle from being absorbed (that absorption reproduces on the
  GT: the control's ratio keeps 86% of the input lag-1, 52% of neighbour-R2).
  `sublook_n2n` stays 0 by default.
  Tables/figures: `docs/figures/metrics_track_w.txt`, `compare_track_w.png`,
  `metrics_track_w_gt.txt`, `compare_track_w_gt.png`; scripts
  `scripts/experiments/run_track_w.py` (`--control`), `run_track_w_gt.py`.
- **Protocol upgrade:** every future GT number should use the shaped-speckle
  phantom (`make_phantom_slc(clean, sigma_n, transfer=Hf)`), not the white
  one — white speckle cannot show the correlated-speckle absorption.

**Where phase differences stand (user decision 2026-09-06: noted, parked).**
Per-pixel, inter-channel phase cannot buy more than ~0.3 dB on this data —
a Fisher-information ceiling, not an experimental miss: complex/amplitude
information ratio 2.11/1.98 for HV-VH (coherence 0.83) and 1.07/1.00 for
HH-VV (coherence 0.35), i.e. even a perfect C3/T3 Wishart data term tops out
at ~0.3 dB on the amplitude channels. Two uses remain open, untried:
(1) **W3** — HH-VV CPD / |gamma| maps as a non-local GROUPING feature
(scene information the amplitudes lack, R^2 0.10; not bounded by the
ceiling because it changes which pixels are averaged, not per-pixel
information; expectation low; ~20 min code + one 8 min run);
(2) **polarimetric output** — despeckle the covariance matrix itself
(coherence, CPD, H/A/alpha), the only denoising problem in which the phase
differences are the target; the shaped-speckle phantom can score it with a
controllable HH-VV coherence. Larger scope; a thesis-scope decision.
Remaining measured levers for amplitude quality are NOT in the phase:
clipping (solved by the unclipped SLC input), low-frequency absorption of
correlated speckle (open; likely a low-frequency regularisation question),
and Track B's ~16 scene patches (more data).

**RESUME POINT:** Track W parked. Next session: decide W3 vs polarimetric
output vs Track B (if the scene patches arrive). Use `denoise(slc=...)` with
the unclipped SLC as the base for any complex-data work, and the
shaped-speckle GT (`run_track_w_gt.py` pattern) for any GT claim.
- Code: `sspmnet/spectral.py`, `load_quadpol_slc`, `denoise(slc=)`,
  `phase_feedback_maps` single-angle on both paths (`load_quadpol_phase`
  now returns the full 2*pi range).

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
