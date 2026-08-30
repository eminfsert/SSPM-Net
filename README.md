# SSPM-Net

**Synchronized Spatio-Polarimetric Masking (SSPM)** — zero-shot
quad-polarimetric SAR despeckling.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eminfsert/SSPM-Net/blob/main/SSPM_Net_Demo.ipynb)

This repository is the example demo and reference code accompanying the paper
*“Application of Zero-Shot Self-Supervised Speckle Denoising Techniques in
Polarimetric Synthetic Aperture Radar Imagery.”*

SSPM-Net removes speckle noise from quad-pol Synthetic Aperture Radar (SAR)
images **without any clean reference and without pre-training**: the network is
trained from scratch on a *single* image (zero-shot, per-image optimization),
and **SSPM is what provides the self-supervision** — per polarization group:

- **Co-pol channels (HH, VV):** blind-spot self-supervision — random pixels are
  masked and the network must predict each hidden value from its surrounding
  neighbors, so it learns image structure instead of memorizing the noise.
- **Cross-pol channels (HV, VH):** a *synchronized* mask drops the **same**
  pixels in both channels. By monostatic reciprocity, HV and VH share the
  **same clean signal but carry independent noise**, so the network is trained
  to predict the masked HV from the (noisy) VH and vice-versa. It can only fit
  the shared clean signal, not the independent noise — radar physics itself
  provides the supervision.

This repository is a clean, minimal implementation with a one-click Colab demo.

> **TR —** SSPM-Net, dört-polarizasyonlu SAR görüntülerindeki benek gürültüsünü
> **temiz referans ve ön-eğitim olmadan**, her görüntü için sıfırdan eğitilerek
> giderir. Temel katkı, **Eşzamanlı Uzamsal-Polarimetrik Maskeleme (SSPM)**:
> eş-polarizasyon kanallarında kör-nokta öz-denetimi, çapraz-polarizasyon
> kanallarında ise karşılıklılık (HV = VH) ilkesiyle her kanalı diğerinden
> tahmin etme.

---

## Quick start — run in Google Colab

**[▶ Open `SSPM_Net_Demo.ipynb` in Colab](https://colab.research.google.com/github/eminfsert/SSPM-Net/blob/main/SSPM_Net_Demo.ipynb)** (or click the badge above).

Then just:

1. **Runtime → Change runtime type → GPU** (an A100/T4 is ideal).
2. **Runtime → Run all.**

The notebook clones this repo, loads the bundled quad-pol patch, trains SSPM-Net
from scratch on it, and shows the **noisy vs. denoised** result inline together
with speckle / reciprocity metrics. No setup, no Google Drive, no manual data
download.

> **TR —** Üstteki rozete tıkla → Colab açılır → GPU seç → *Run all*. Notebook
> repoyu klonlar, örnek yamayı yükler, modeli sıfırdan eğitir ve gürültülü vs.
> temizlenmiş sonucu metriklerle birlikte gösterir.

---

## Method at a glance

```
Input (HH, HV, VH, VV), amplitude, normalized to [0,1]
   │
   ├── SSPM masking  ──  co-pol: independent blind-spot
   │                     cross-pol: synchronized (HV ↔ VH predict each other)
   │
   ├── Asymmetric dual branch
   │     • Co-pol branch   (HH, VV share weights)
   │     • Cross-pol branch (HV, VH share weights)
   │     each branch:  Haar DWT → Swin Transformer (LL, global context)
   │                              + CNN (LH/HL/HH detail bands)
   │                   → inverse DWT → feature map
   │
   ├── Cross-Polarization Attention   (4 channels attend to each other)
   ├── Per-channel refinement × 4
   └── Output (HH, HV, VH, VV)

Training: edge-aware TV + HV≈VH consistency + speckle factorization with
Rayleigh histogram matching + non-local self-similarity, with EMA and a fixed
iteration budget; the final output is the single EMA checkpoint at the last
iteration, refined with D4 × MC-dropout test-time augmentation.
```

## Repository layout

```
SSPM-Net/
├── SSPM_Net_Demo.ipynb     # ← the demo: open in Colab, Run all (GPU)
├── requirements.txt
├── data/
│   ├── example_quadpol.npy # bundled real quad-pol patch (4, 512, 512) [HH,HV,VH,VV]
│   └── tiff/               # same patch as per-component TIFFs (amp/real/imgy/pha × 4 pol)
├── scripts/
│   └── compare_ri.py       # amplitude-only vs. complex-aux head-to-head
└── sspmnet/
    ├── model.py            # SSPMNet (the network)
    ├── complex_data.py     # optional SLC (real/imgy) auxiliaries: RI-N2N + guide
    ├── masking.py          # SSPM: QuadPolSpatialMasker, BernoulliMasker
    ├── trainer.py          # zero-shot training loop  ->  denoise(...)
    ├── losses.py           # self-supervised losses + speckle/histogram terms
    ├── metrics.py          # ENL, ENL-ROI, EPI, SSIM, HV/VH reciprocity
    ├── config.py           # model architecture config
    ├── freq_decomposition.py   # Haar DWT / inverse DWT
    ├── high_freq_branch.py     # Swin Transformer (processes the LL sub-band)
    ├── low_freq_branch.py      # CNN (processes the detail sub-bands)
    ├── reconstruction.py
    └── cross_attention.py
```

> The sub-module class names `HighFreqBranch` (Swin, on the LL sub-band) and
> `LowFreqBranch` (CNN, on the detail sub-bands) are kept for historical
> stability; their actual roles are documented above and in the source.

## Run it locally (optional)

The Colab notebook is the easiest path, but you can also run from Python:

```bash
git clone https://github.com/eminfsert/SSPM-Net.git
cd SSPM-Net
pip install -r requirements.txt
```

Python ≥ 3.8 and PyTorch ≥ 1.12. A CUDA GPU is strongly recommended.

```python
import numpy as np
from sspmnet import denoise, TrainConfig

amp = np.load("data/example_quadpol.npy")   # (4, H, W), order [HH, HV, VH, VV]
result = denoise(amp, TrainConfig(iters=700))
denoised = result["denoised"]                # (4, H, W), same scale as the input
print("Stopped at step:", result["stop_step"])
```

Input is amplitude (any positive scale; it is normalized internally by the
per-channel 99th percentile). For best results use square patches (e.g.
256–512 px) whose side is a multiple of the wavelet/window size.

### Optional: complex (real/imaginary) auxiliaries — smoother flats, sharper edges

If the single-look complex components are available (as in `data/tiff/`,
which holds per-channel `amp`, `real`, `imgy`, `pha` TIFFs), the pipeline can
exploit two additional physics facts:

- **RI Noise2Noise targets (MERLIN-style).** For fully-developed speckle,
  Re ⟂ Im given the reflectivity, so |Re| and |Im| are two *independent*
  noisy amplitude observations of the same clean signal. Supervising the
  masked pixels with both targets halves the effective target-noise power.
- **Multi-look guided regularization.** The ~8-look span built from
  |Re|²/|Im|² of all 4 channels gives a far cleaner (log-domain, ratio-based)
  edge map than the 1-look amplitude; the edge-aware TV and non-local
  losses are steered by it, so flat areas are smoothed harder while true
  edges are protected better.

The per-channel phase is uniformly distributed for distributed targets and
is therefore not used directly as a value; what it DOES carry is
cross-channel evidence, exploited by the **phase feedback** below.

```python
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs

amp, ri_pair = load_quadpol_tiffs("data/tiff")   # calibrated |Re|/|Im| pair
result = denoise(amp, TrainConfig(iters=700), ri_pair=ri_pair)
```

### Optional: phase feedback — a per-pixel "is this value noise?" map

Monostatic reciprocity gives S<sub>HV</sub> = S<sub>VH</sub>: the two
cross-pol channels share the *same* complex speckle and differ only by
additive thermal/system noise. Their local phase agreement (circular
concentration of the doubled angle, robust to the [0, π] fold of the
bundled uint8 phase files) therefore estimates the per-pixel
signal-to-noise ratio: ≈1 where the observed value is signal (speckle
included), ≈floor where it is instrument noise (dark cross-pol: roads,
water, shadow). `sspmnet.phase_data` turns the `*_pha.tiff` files into
this map (plus an HH–VV surface-scattering and a spatial phase-coherence
map), and the trainer uses it two ways:

- **Regularization boost** (`phase_smooth_boost`): TV and the non-local
  loss are multiplied by `1 + b·(1 − snr)` — noise-dominated pixels are
  smoothed harder, while the multi-look edge weights keep true edges
  protected.
- **Data-term down-weight** (`phase_fidelity`): the cross-pol MERLIN /
  N2N loss is scaled by `1 − f·(1 − snr)` — where the observation is
  thermal noise, its positive-median target is trusted less and the
  regularizers take over.

```python
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase

amp, ri_pair = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")
cfg = TrainConfig(iters=700, ri_mode="merlin", tv_mult=10.0,
                  guide_cv_protect=0.3,
                  phase_smooth_boost=3.0, phase_fidelity=0.5)
result = denoise(amp, cfg, ri_pair=ri_pair, pha=pha)
```

On the ground-truth protocol (known clean + simulated 1-look complex
speckle with shared cross-pol speckle and thermal noise calibrated to the
real patch's HV–VH coherence), this configuration raises ENL-ROI(HV) by
+122% and ENL-ROI(HH) by +14% over the complex-aux winner at equal PSNR,
with true EPI(HV) improving 0.768 → 0.782 (`b=5` trades a further ENL
gain, +177%, for ~0.1 dB of PSNR(HV)). Reproduce with
`python scripts/compare_ri.py --merlin --phase`.

### Optional: residual-speckle refinement — whiteness loss + non-local polish

Two further mechanisms remove the residual grain that survives the
regularizers, without blurring (both validated on the ground-truth
protocol; a third — refreshing the non-local reference from the model's
own EMA output, `nl_self_refresh` — was tried and HURT every metric, so
it defaults to off):

- **Ratio-whiteness loss** (`whiteness_lambda`, e.g. `0.05`): for a
  perfect result the intensity ratio noisy²/output² is pure speckle —
  spatially white. Residual speckle left in the output *structures* the
  ratio; penalizing its small-lag autocorrelation actively pushes that
  residue out. On oversampled real data keep the lags above the speckle
  correlation length (`whiteness_lags=(3, 4, 5)`; lag-1 autocorrelation
  of the bundled patch's speckle is ~0.5 and is NOT residue).
- **Final non-local polish** (`polish`, e.g. `0.5`): one guided-NLM pass
  on the finished output using the output itself as the similarity guide,
  blended per pixel and shut off on heterogeneous / deterministic pixels
  (CV gate + the phase `det` map) — grain has many similar neighbors in
  the cleaned image and averages out; edges and point targets don't.

On the ground-truth protocol, PH + whiteness + polish(0.5) improves
every accuracy metric over the phase-feedback config (PSNR +0.12/+0.21 dB,
EPI(HH/HV) 0.823/0.785 → 0.827/0.791) while raising ENL-ROI to 141/163
(HH/HV; the amplitude-only baseline sits at ~100/62). The
`--merlin --phase` reproduction command includes both refinements.

Two RI modes are available via `TrainConfig.ri_mode`:

- `"targets"` (default): the amplitude pipeline is kept and the masked
  losses gain the two independent RI targets.
- `"merlin"`: full MERLIN-style **input separation** — each step the network
  is fed ONE component's pseudo-amplitude and supervised by the OTHER, so
  the input carries none of the target's noise and *every* pixel supervises
  (no masking). Cross-pol channels are additionally supervised by the
  reciprocal channel's opposite component (`merlin_recip_weight`).
  Inference averages the predictions from both components.

`python scripts/compare_ri.py` runs the amplitude-only baseline and the
complex-aux variant head-to-head on the bundled TIFF patch and prints the
fair metrics table. The relevant knobs live in `TrainConfig`:
`ri_mode`, `ri_weight` (share of the masked loss on RI targets, default
0.6), `guide_tv` / `guide_alpha`, `guide_nlm`, and `merlin_recip_weight`.

Two further refinements for the MERLIN mode:

- `merlin_loss="nll"` uses MERLIN's original Gaussian negative
  log-likelihood instead of L1 — unbiased in expectation (the predicted
  intensity converges to the true reflectivity), removing the median-
  convention scale offset of the L1 target.
- `guide_cv_protect` (e.g. `0.3`) adds a Lee-style heterogeneity gate to
  the guided TV: the local coefficient of variation of the multi-look span
  separates homogeneous areas (smoothed at full strength) from texture /
  point targets (regularization shut off), targeting strong speckle
  suppression and edge preservation at the same time.

The model also prints a metrics table comparing the noisy input and the
SSPM-Net output, e.g.:

```
  Method         corr(HV,VH)      MAD     RMSE  ENL-ROI(HV)   EPI(HV)  SSIM(HV)
  Noisy               0.xxxx   x.xxxx   x.xxxx         x.xx     1.000     1.000
  SSPM-Net            0.xxxx   x.xxxx   x.xxxx         x.xx     x.xxx     x.xxx
```

Higher **ENL-ROI** means stronger speckle suppression; higher **corr(HV,VH)**
(and lower MAD/RMSE) means better polarimetric reciprocity — the main evidence
for the method.

## The bundled example

`data/example_quadpol.npy` is a real 512×512 quad-pol patch from the
**AIR-PolSAR (Gaofen-3)** scene used in the thesis. Note that even in the noisy
input the HV and VH channels are nearly identical in the mean — that is the
reciprocity that SSPM exploits.

## Citation

This code accompanies the M.Sc. thesis *“Application of Zero-Shot
Self-Supervised Speckle Denoising Techniques in Polarimetric Synthetic
Aperture Radar Imagery”*.

```bibtex
@mastersthesis{sert2026sspmnet,
  title  = {Application of Zero-Shot Self-Supervised Speckle Denoising
            Techniques in Polarimetric Synthetic Aperture Radar Imagery},
  author = {Sert, Muhammed Emin},
  school = {Hacettepe University, Dept. of Electrical and Electronics Engineering},
  year   = {2026}
}
```

**Author:** Muhammed Emin Sert · **Advisor:** Prof. Dr. Uğur Baysal ·
Hacettepe University.

## License

Released under the [MIT License](LICENSE).
