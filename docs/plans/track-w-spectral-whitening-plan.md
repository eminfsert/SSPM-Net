# Track W — phase-based speckle whitening (spectral pre-processing) + sub-look N2N

Date: 2026-09-06. Branch `feature/scene-scale-arch`. Status: CLOSED — W1 negative, W2 a smoothing trade on the shaped-speckle GT (see CLAUDE.md, Track W). Durable outcomes: the unclipped centred SLC input, the shaped-speckle GT protocol, the phase-difference analysis below.

## Question that started it

"We have four phases, so three independent phase DIFFERENCES. What do they mean,
do they carry speckle information, and can they remove speckle or point at the
clean signal?"  Answered with read-only measurements on the real patch
(`phi = pha/255*2pi`, unclipped SLC = npy amplitude + pha phase).

## Findings

### A. Inter-channel phase differences (HV-VH, HH-VV, HH-HV)

| pair | local 7x7 \|gamma\| (null 0.16) | meaning | speckle information? |
|---|---|---|---|
| HV-VH | 0.77 (dark 0.55, bright 0.84) | reciprocity; deviation = thermal noise; constant +140 deg offset | per-pixel phase agreement tracks intensity (corr +0.49) but this is thermal-SNR information; Fisher analysis: the complex pair gives only 11% more information about the clean intensity than the amplitude pair (2.11x vs 1.98x over one channel), and the amplitude correlation is already exploited (N2N HV<->VH, `xpol_pair_input`). Redundant. |
| HH-VV (CPD) | 0.35 (bright 0.39) | scattering mechanism: bimodal on bright pixels, 53% within +-30 deg of 180 (double bounce); flat on dark | per-pixel Fisher gain 1.05-1.17x only. But it is SCENE information not present in the amplitudes (R^2 from 4 local intensities + CV = 0.10-0.14; spatial autocorrelation beyond the window 0.11-0.22 vs null 0.02-0.05): the off-diagonal of the clean covariance. Groups similar scatterers; does not remove speckle. |
| HH-HV (helix) | 0.32, rises 0.27 -> 0.41 with brightness | reflection-symmetry breaking (oriented man-made targets) | same class as HH-VV: scene descriptor. |

Physics: speckle is the random scatterer arrangement inside the cell, common to
all four channels. Subtracting two channels' phases CANCELS that common random
phase; what remains is the polarimetric response of the scatterer (scene) plus
thermal noise. Inter-channel phase differences therefore describe what is
*beyond* speckle, which is why coherent HV+VH fusion was measured negative.

### B. Spatial phase (complex spectrum): the real speckle information

- Spectrum fills only ~42-60% of Nyquist (Hamming-like, ~2x oversampling).
  Lag-1 complex coherence 0.68/0.57 (cols/rows) -> lag-1 correlation of the
  normalised speckle intensity **0.60/0.46**.
- **Blind-spot self-supervision assumes independent neighbour noise.**
  Measured: 27-29% (clipped uint8) / 54-56% (full range) of a pixel's
  log-speckle is linearly predictable from its 8 neighbours. The network can
  reproduce that as "signal" -> residual grain, `whiteness_lags=(3,4,5)`
  patches, ENLr not reaching 1.
- Spectrum centroid +0.022 cyc/px along rows -> corr(Re, Im_neighbour)
  +0.09/-0.10, a mild violation of MERLIN's Re/Im independence. Centring the
  spectrum brings it to ~0.
- Fix (only possible with the phase): centre + flatten in-band + 2x
  decimation (band-limited, lossless on the 256 grid) -> lag-1 speckle
  correlation 0.60 -> 0.21, neighbour-R^2 0.27 -> 0.16 (full range
  0.56 -> 0.19-0.24). Scene preserved (local-mean corr 0.98).
- Sub-looks (two half-bands) have independent speckle (synthetic control
  corr 0.02) and shared reflectivity: a genuinely independent Noise2Noise
  pair, unlike HV/VH whose speckle is ~70% shared. Real-data sub-look
  correlation 0.3-0.5 is texture / point targets / clipping, not speckle.
- uint8 clipping puts 28% of the power out of band; the unclipped SLC
  (npy amplitude + pha) 2.5%. Spectral work uses `load_quadpol_slc(...,
  amp_npy="data/example_quadpol.npy")`.
- Protocol gap: `run_indep_eval.py` simulates WHITE speckle, so no GT number
  so far could see the correlated-speckle problem.

## Method

- W0 (done): `sspmnet/spectral.py` (`centre_spectrum`, `estimate_transfer`,
  `whiten`, `unwhiten_amp`, `sublooks`, `speckle_whiteness`, `reim_leak`);
  `load_quadpol_slc`; `denoise(slc=)`; `phase_feedback_maps` single-angle on
  both paths.
- W1 (running): real patch, `base` vs `flat+decim`, full stack, 700 iters.
  `scripts/experiments/run_track_w.py`.
- W2 (next, ask first): sub-look N2N data term.
- W3 (later): CPD / coherence maps as NLM similarity features.
- Then: phantom with SHAPED speckle (real transfer function) for a fair GT
  protocol.

Bar: independent shaped-speckle GT PSNR >= +0.5 dB in at least one channel
at equal EPI, flatHP not worse; real patch: ratio-image residual whiteness
towards the input's own value, ENLr towards 1, waterHP not up.
