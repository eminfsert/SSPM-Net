"""Track E1: censored (one-sided) loss on uint8-saturated targets.

Measures how much of the 8-bit clipping cost E1 recovers, on the CURRENT base
(pixel+group + xpol_pair_input).  A4 established the two reference points:
matched uint8 clipping costs -1.7/-1.5 dB against an unclipped substrate, and
masking the saturated pixels OUT of the data term loses a further -1.6 dB.

Synthetic leg (the ceiling measurement).  The clean proxy and the speckle +
thermal simulation are identical to run_hv_phase.py / run_tvmult_sweep.py; the
simulated components are then quantized to uint8 with per-channel clip
fractions matched to the real TIFFs, exactly as in run_a4_sat.py.  Rows:

    noclip      unclipped substrate            -- the ceiling
    clip        clipped, no countermeasure     -- today's state
    clip+sat    clipped, saturated pixels DROPPED (A4's negative control)
    clip+cens   clipped, saturated targets CENSORED (E1)
    clip+cens+tv{r}   E1 + TV relaxed over the saturated tail

Real leg: base vs cens vs cens+tv_relax on the real patch.

Bar (docs/plans/track-e-fullrange-logdomain-plan.md): recover >= +1.0 dB of the
clipping cost on GT in BOTH channels, at equal EPI, with no flat-water grain
increase; on the real patch the bright top-1% must not flatten (ratio >= 0.9).

All runs are .npy-cached under results/ri_compare so an interrupted Colab
session resumes.  Usage:  python scripts/experiments/run_track_e.py
"""
import os
import sys
import glob

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps, _local_coherence, _local_mean
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric)

OUT = "results/ri_compare"
os.makedirs(OUT, exist_ok=True)

amp, ri, sat_real = load_quadpol_tiffs("data/tiff", return_sat=True)
pha = load_quadpol_phase("data/tiff")
maps_real = phase_feedback_maps(pha=pha, win=7)


def cached(fname, fn):
    f = os.path.join(OUT, fname)
    if os.path.exists(f):
        print(f"[cached] {fname}", flush=True)
        return np.load(f)
    out = fn()
    np.save(f, out)
    return out


# ── clean proxy + simulated speckle (identical to run_hv_phase.py) ──
clean = cached("denoised_baseline.npy",
               lambda: denoise(amp, TrainConfig(iters=700))["denoised"]
               ).astype(np.float64)
xpol = 0.5 * (clean[1] + clean[2])
clean[1] = xpol
clean[2] = xpol

u = np.exp(2j * pha.astype(np.float64))
target_coh = float(_local_coherence(u[1] * np.conj(u[2]), 7).mean())
rng = np.random.default_rng(7)


def cn(shape, sigma):
    return rng.normal(0, sigma / np.sqrt(2), shape) + \
        1j * rng.normal(0, sigma / np.sqrt(2), shape)


def simulate(sigma_n):
    g = np.stack([cn(clean[0].shape, 1.0) for _ in range(4)])
    g[2] = g[1]
    return clean * g + np.stack([cn(clean[0].shape, sigma_n) for _ in range(4)])


def sim_coh(sigma_n):
    z = simulate(sigma_n)
    uu = np.exp(2j * np.angle(z))
    return float(_local_coherence(uu[1] * np.conj(uu[2]), 7).mean())


rms_x = float(np.sqrt((clean[1] ** 2).mean()))
lo, hi = 0.05 * rms_x, 3.0 * rms_x
for _ in range(12):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if sim_coh(mid) > target_coh else (lo, mid)
sigma_n = 0.5 * (lo + hi)
print(f"sigma_n={sigma_n:.3f}  sim coherence={sim_coh(sigma_n):.3f} "
      f"(target {target_coh:.3f})", flush=True)

rng = np.random.default_rng(7)
z = simulate(sigma_n)
amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(z.real).astype(np.float32),
                      np.abs(z.imag).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)

# ── uint8 clipping matched to the REAL per-channel saturation fractions ──
def u8_clip(x, frac_sat):
    thr = np.quantile(x, 1.0 - frac_sat)
    return np.clip(np.round(x * (255.0 / max(thr, 1e-9))), 0, 255).astype(np.float32)


def real_sat_frac(comp):
    from sspmnet.complex_data import _read_tiff
    return [float((_read_tiff(f) >= 255).mean())
            for f in sorted(glob.glob(f"data/tiff/*_{comp}.tiff"))]


frac_amp, frac_re, frac_im = (real_sat_frac(c) for c in ("amp", "real", "imgy"))
print("real sat fractions amp/re/im:",
      [f"{v:.3f}" for v in frac_amp], [f"{v:.3f}" for v in frac_re],
      [f"{v:.3f}" for v in frac_im], flush=True)

amp_c = np.stack([u8_clip(amp_sim[c], frac_amp[c]) for c in range(4)])
re_c = np.stack([u8_clip(np.abs(z.real[c]), frac_re[c]) for c in range(4)]).astype(np.float32)
im_c = np.stack([u8_clip(np.abs(z.imag[c]), frac_im[c]) for c in range(4)]).astype(np.float32)
sat_c = (amp_c >= 255.0) | (re_c >= 255.0) | (im_c >= 255.0)
ri_c = calibrate_ri(amp_c, re_c, im_c, mode="l1")
print(f"synthetic sat fraction/ch: {(sat_c.mean(axis=(1, 2)) * 100).round(2)}",
      flush=True)

# ── flat-water mask (grain column; identical to run_hv_phase.py) ──
span8 = np.sqrt((ri.astype(np.float64) ** 2).mean(axis=(0, 1)))
mu_s = _local_mean(span8, 21)
cv_s = np.sqrt(np.maximum(_local_mean(span8 ** 2, 21) - mu_s ** 2, 0)) / (mu_s + 1e-6)
x2 = 0.5 * (ri[:, 1] + ri[:, 2]).mean(axis=0).astype(np.float64)
x2s = _local_mean(x2, 21)
m0 = (x2s < np.percentile(x2s, 20)) & (cv_s < 0.40)
water = _local_mean(m0.astype(np.float64), 5) > 0.999
print(f"water mask: {int(water.sum())} px", flush=True)


def grain(d, c=1):
    hp = d[c] - _local_mean(d[c].astype(np.float64), 5)
    return (float(hp[water].std()),
            float(d[c][water].std() / max(d[c][water].mean(), 1e-9)))


# ── variants ──
PG = {"dropout_style": "pixel", "norm": "group"}
BASE_M = {**PG, "xpol_pair_input": True}
STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, polish=0.5, edge_boost=1.0)

# (name, cache tag, TrainConfig extras, use clipped substrate, sat argument)
SYNTH = [
    ("noclip",        "trke_synth_noclip",   {},                              False, None),
    ("clip",          "trke_synth_clip",     {},                              True,  None),
    ("clip+sat",      "trke_synth_clip_sat", {},                              True,  "sat"),
    ("clip+cens",     "trke_synth_cens",     {"sat_censored": True},          True,  "sat"),
    ("clip+cens+tv.5", "trke_synth_cens_tv05",
     {"sat_censored": True, "sat_tv_relax": 0.5},                             True,  "sat"),
    ("clip+cens+tv1", "trke_synth_cens_tv10",
     {"sat_censored": True, "sat_tv_relax": 1.0},                             True,  "sat"),
]
REAL = [
    ("base",          "trke_real_base",      {},                              None),
    ("cens",          "trke_real_cens",      {"sat_censored": True},          "sat"),
    ("cens+tv.5",     "trke_real_cens_tv05",
     {"sat_censored": True, "sat_tv_relax": 0.5},                             "sat"),
]

runs_s = {}
for name, tag, extra, clipped, satarg in SYNTH:
    print(f"\n=== synth {name} ===", flush=True)
    a_in = amp_c if clipped else amp_sim
    ri_in = ri_c if clipped else ri_sim
    s_in = sat_c if satarg else None
    runs_s[name] = cached(tag + ".npy", lambda: denoise(
        a_in, TrainConfig(**{**STACK, **extra}, whiteness_lags=(1, 2, 3),
                          model_cfg=BASE_M),
        ri_pair=ri_in, pha=maps_sim, sat=s_in)["denoised"]).astype(np.float64)

runs_r = {}
for name, tag, extra, satarg in REAL:
    print(f"\n=== real {name} ===", flush=True)
    s_in = sat_real if satarg else None
    runs_r[name] = cached(tag + ".npy", lambda: denoise(
        amp, TrainConfig(**{**STACK, **extra}, whiteness_lags=(3, 4, 5),
                         model_cfg=BASE_M),
        ri_pair=ri, pha=maps_real, sat=s_in)["denoised"]).astype(np.float64)


# ── tables ──
def scale_match(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        out[c] = x[c] * float((x[c] * ref[c]).sum() / max((x[c] ** 2).sum(), 1e-9))
    return out


def psnr(x, ref):
    return 10 * np.log10(ref.max() ** 2 / max(((x - ref) ** 2).mean(), 1e-12))


thr99 = np.quantile(clean, 0.99)
bright = clean >= thr99

lines = []
rois_s, rs_s = find_top_k_rois(amp_sim[1].astype(np.float64))
W = 11
cols = ["PSNR(HH)", "PSNR(HV)", "SSIM(HV)", "EPI(HH)", "EPI(HV)", "ENL(HV)",
        "RMSEbrgt", "brightR", "waterHP", "waterCV"]
title = ("Track E1 synthetic-GT (vs known clean; scale-matched); base = "
         "pixel+group + xpol_pair_input.  RMSEbrgt = RMSE over the top-1% "
         "clean pixels; brightR = mean output/clean ratio there (flattening "
         f"detector, ideal 1); waterHP = flat-water high-pass std of HV "
         f"(clean {grain(clean)[0]:.4f})")
hdr = "  {:<16}".format("Method") + "".join(f"{c:>{W}}" for c in cols)
print("\n" + title)
print(hdr)
lines += [title, hdr]
for name, d in runs_s.items():
    ds = scale_match(d, clean)
    g = grain(ds)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1]),
            enl_roi_multi(d[1], rois_s, rs_s),
            float(np.sqrt(((ds - clean) ** 2)[bright].mean())),
            float((ds[bright] / np.maximum(clean[bright], 1e-9)).mean()),
            g[0], g[1]]
    line = "  {:<16}".format(name) + "".join(f"{v:>{W}.4f}" for v in vals)
    print(line, flush=True)
    lines.append(line)

# recovered fraction of the clipping cost
if "noclip" in runs_s and "clip" in runs_s:
    def p(nm, c):
        return psnr(scale_match(runs_s[nm], clean)[c], clean[c])
    for c, cn_ in ((0, "HH"), (1, "HV")):
        cost = p("noclip", c) - p("clip", c)
        s = f"  clipping cost({cn_}) = {cost:+.3f} dB;  recovered: " + ", ".join(
            f"{nm} {p(nm, c) - p('clip', c):+.3f} dB"
            for nm in runs_s if nm not in ("noclip", "clip"))
        print(s)
        lines.append(s)

rois_r, rs_r = find_top_k_rois(amp[1])


def ratio_enl(d, c):
    eps = 1e-3
    rI = (amp[c].astype(np.float64) ** 2 + eps) / (d[c] ** 2 + eps)
    v = rI[(d[c] > 2) & (amp[c] > 0)]
    return (v.mean() / v.std()) ** 2


# Per-CHANNEL saturation mask: the union mask includes pixels that are
# saturated in some other channel while HV itself sits near zero, and the
# output/input ratio there is meaningless (it blew up to 5.7e8).
sat_hv_r = sat_real[1]
cols_r = ["EPI(HH)", "EPI(HV)", "ENLr(HH)", "ENLr(HV)", "ENL-ROI(HV)",
          "satRatio", "waterHP", "waterCV"]
title_r = ("Track E1 real patch.  satRatio = MEDIAN output/input over the "
           "pixels where HV ITSELF is clipped (>=1 means the bright tail is "
           "not flattened; the per-channel mask and the median keep it robust "
           "- the union mask includes pixels near zero in HV and blows up)")
hdr_r = "  {:<16}".format("Method") + "".join(f"{c:>{W}}" for c in cols_r)
print("\n" + title_r)
print(hdr_r)
lines += ["", title_r, hdr_r]
for name, d in runs_r.items():
    g = grain(d)
    vals = [epi_metric(amp[0].astype(np.float64), d[0]),
            epi_metric(amp[1].astype(np.float64), d[1]),
            ratio_enl(d, 0), ratio_enl(d, 1),
            enl_roi_multi(d[1], rois_r, rs_r),
            float(np.median(d[1][sat_hv_r] / np.maximum(amp[1][sat_hv_r], 1.0))),
            g[0], g[1]]
    line = "  {:<16}".format(name) + "".join(f"{v:>{W}.4f}" for v in vals)
    print(line, flush=True)
    lines.append(line)

with open(f"{OUT}/metrics_track_e.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}/metrics_track_e.txt")
