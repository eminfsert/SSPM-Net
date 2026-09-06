"""Track W, step 2: spectral whitening on the REAL patch — base vs flat+decim.

base       today's full stack on the uint8 TIFF amplitude + calibrated |Re|/|Im|
           pair + phase maps (512x512)
flat+decim the same stack on the spectrally whitened, unclipped SLC
           (npy amplitude + pha phase; centred, in-band flattened, 2x
           decimated -> 256x256), output resampled back to 512 for the
           same-grid metrics.  whiteness_lags=(1,2,3) here (the residual
           speckle correlation is short-range after whitening) instead of
           the (3,4,5) the raw oversampled data needs.

Metrics (real patch, no clean reference): EPI vs noisy, ratio-ENL (ideal 1),
ENL-ROI(HV), flat-water high-pass std / CV (LS scale-matched onto the noisy
amplitude first — both are scale-sensitive), and the residual-speckle
whiteness on each run's NATIVE grid: rLag1 = lag-1 correlation of the
normalised ratio image noisy^2/out^2 next to the lag-1 correlation of the
noisy speckle itself on that grid (the ratio should carry ALL the speckle,
so the ideal is the input's own value; a lower value means the network
absorbed correlated speckle into the output).

Usage:  python scripts/experiments/run_track_w.py   (runs are .npy-cached)
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")
from sspmnet import (denoise, TrainConfig, load_quadpol_tiffs,
                     load_quadpol_phase, load_quadpol_slc, spectral)
from sspmnet.phase_data import phase_feedback_maps, _local_mean
from sspmnet.metrics import find_top_k_rois, enl_roi_multi, epi_metric

OUT = "results/track_w"
CACHE = "results/ri_compare"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

amp, ri, sat_real = load_quadpol_tiffs("data/tiff", return_sat=True)
pha = load_quadpol_phase("data/tiff")
maps_real = phase_feedback_maps(pha=pha, win=7)
slc = load_quadpol_slc("data/tiff", amp_npy="data/example_quadpol.npy")


def cached(fname, fn):
    f = os.path.join(CACHE, fname)
    if os.path.exists(f):
        print(f"[cached] {fname}", flush=True)
        return np.load(f)
    out = fn()
    np.save(f, out)
    return out


# ── flat-water mask (identical to run_track_e.py) ──
span8 = np.sqrt((ri.astype(np.float64) ** 2).mean(axis=(0, 1)))
mu_s = _local_mean(span8, 21)
cv_s = np.sqrt(np.maximum(_local_mean(span8 ** 2, 21) - mu_s ** 2, 0)) / (mu_s + 1e-6)
x2 = 0.5 * (ri[:, 1] + ri[:, 2]).mean(axis=0).astype(np.float64)
x2s = _local_mean(x2, 21)
m0 = (x2s < np.percentile(x2s, 20)) & (cv_s < 0.40)
water = _local_mean(m0.astype(np.float64), 5) > 0.999
print(f"water mask: {int(water.sum())} px", flush=True)

PG = {"dropout_style": "pixel", "norm": "group", "xpol_pair_input": True}
STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, polish=0.5, edge_boost=1.0, model_cfg=PG)

# ── runs ──
runs = {}      # name -> (out_native, noisy_native_intensity, out_512)
print("\n=== base ===", flush=True)
d = cached("trkw_real_base.npy", lambda: denoise(
    amp, TrainConfig(**STACK, whiteness_lags=(3, 4, 5)),
    ri_pair=ri, pha=maps_real)["denoised"]).astype(np.float64)
runs["base"] = (d, amp.astype(np.float64) ** 2, d)

print("\n=== flat+decim ===", flush=True)
zw, info = spectral.whiten(slc, flatten=True, decim=2)
dw = cached("trkw_real_flatdecim_v2.npy", lambda: denoise(
    None, TrainConfig(**STACK, whiteness_lags=(1, 2, 3)),
    slc=zw)["denoised"]).astype(np.float64)
runs["flat+decim"] = (dw, np.abs(zw).astype(np.float64) ** 2,
                      spectral.unwhiten_amp(dw, info))

# ── W2: sub-look Noise2Noise on the 512 grid (centred, unclipped SLC; no
#    flattening, no decimation).  --control adds the same SLC without the
#    sub-look term, to separate "unclipped centred SLC" from "sub-look N2N".
slc_c, _ = spectral.centre_spectrum(slc)
if "--control" in sys.argv:
    print("\n=== slc512 (control) ===", flush=True)
    dc = cached("trkw_real_slc512.npy", lambda: denoise(
        None, TrainConfig(**STACK, whiteness_lags=(3, 4, 5)),
        slc=slc_c)["denoised"]).astype(np.float64)
    runs["slc512"] = (dc, np.abs(slc_c).astype(np.float64) ** 2, dc)
print("\n=== slc512+sublook ===", flush=True)
ds_ = cached("trkw_real_slc512_sub.npy", lambda: denoise(
    None, TrainConfig(**STACK, whiteness_lags=(3, 4, 5), sublook_n2n=1.0),
    slc=slc_c)["denoised"]).astype(np.float64)
runs["slc512+sub"] = (ds_, np.abs(slc_c).astype(np.float64) ** 2, ds_)


# ── metrics ──
def scale_match(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        out[c] = x[c] * float((x[c] * ref[c]).sum() / max((x[c] ** 2).sum(), 1e-9))
    return out


def ratio_enl(d512, c):
    eps = 1e-3
    rI = (amp[c].astype(np.float64) ** 2 + eps) / (d512[c] ** 2 + eps)
    v = rI[(d512[c] > 2) & (amp[c] > 0)]
    return (v.mean() / v.std()) ** 2


def grain(d512, c=1):
    hp = d512[c] - _local_mean(d512[c], 5)
    return (float(hp[water].std()),
            float(d512[c][water].std() / max(d512[c][water].mean(), 1e-9)))


def ratio_whiteness(out_native, noisy_I, c=1):
    r = (noisy_I[c] + 1e-3) / (out_native[c] ** 2 + 1e-3)
    w = spectral.speckle_whiteness(r)
    w0 = spectral.speckle_whiteness(noisy_I[c])
    return 0.5 * (w["lag1_x"] + w["lag1_y"]), 0.5 * (w0["lag1_x"] + w0["lag1_y"]), w["neigh_r2"], w0["neigh_r2"]


rois_r, rs_r = find_top_k_rois(amp[1].astype(np.float64))
cols = ["EPI(HH)", "EPI(HV)", "ENLr(HH)", "ENLr(HV)", "ENL-ROI(HV)",
        "waterHP", "waterCV", "rLag1", "inLag1", "rNeighR2", "inNeighR2"]
W = 11
title = ("Track W real patch (512 grid; flat+decim output resampled from 256). "
         "rLag1/rNeighR2 = residual-speckle whiteness of the ratio image on the run's "
         "NATIVE grid, inLag1/inNeighR2 = the same for the noisy input on that grid "
         "(ideal: ratio == input, i.e. all speckle left in the ratio)")
hdr = "  {:<12}".format("Method") + "".join(f"{c:>{W}}" for c in cols)
lines = [title, hdr]
print("\n" + title)
print(hdr)
for name, (dn, nI, d512) in runs.items():
    ds = scale_match(d512, amp.astype(np.float64))
    g = grain(ds)
    rw = ratio_whiteness(dn, nI)
    vals = [epi_metric(amp[0].astype(np.float64), d512[0]),
            epi_metric(amp[1].astype(np.float64), d512[1]),
            ratio_enl(d512, 0), ratio_enl(d512, 1),
            enl_roi_multi(d512[1], rois_r, rs_r),
            g[0], g[1], rw[0], rw[1], rw[2], rw[3]]
    line = "  {:<12}".format(name) + "".join(f"{v:>{W}.4f}" for v in vals)
    print(line, flush=True)
    lines.append(line)
# ── native-grid comparison (256): base block-averaged 2x2 vs flat+decim ──
w256 = water.reshape(256, 2, 256, 2).mean((1, 3)) > 0.999
n256 = np.abs(slc).reshape(4, 256, 2, 256, 2).mean((2, 4))
b256 = runs["base"][0].reshape(4, 256, 2, 256, 2).mean((2, 4))
rois2, rs2 = find_top_k_rois(np.abs(zw[1]).astype(np.float64))
title2 = ("Native 256 grid: base = 2x2 block mean of the 512 output vs its block-mean "
          "noisy; flat+decim vs the whitened noisy |z_w|. ENLr/EPI against each row's "
          "own noisy; waterCV over the 256 water mask; ENL-ROI on ROIs picked on |z_w|")
cols2 = ["EPI(HH)", "EPI(HV)", "ENLr(HH)", "ENLr(HV)", "ENL-ROI(HV)", "waterCV", "out/in"]
hdr2 = "  {:<12}".format("Method") + "".join(f"{c:>{W}}" for c in cols2)
print("\n" + title2)
print(hdr2)
lines += ["", title2, hdr2]
for name, o, n in (("base", b256, n256), ("flat+decim", dw, np.abs(zw).astype(np.float64))):
    def enlr(c):
        r = ((n[c] ** 2 + 1e-3) / (o[c] ** 2 + 1e-3))[o[c] > 2]
        return (r.mean() / r.std()) ** 2
    vals = [epi_metric(n[0], o[0]), epi_metric(n[1], o[1]), enlr(0), enlr(1),
            enl_roi_multi(o[1], rois2, rs2),
            float(o[1][w256].std() / max(o[1][w256].mean(), 1e-9)),
            float(o[1].mean() / n[1].mean())]
    line = "  {:<12}".format(name) + "".join(f"{v:>{W}.4f}" for v in vals)
    print(line, flush=True)
    lines.append(line)

with open(f"{OUT}/metrics_track_w.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {OUT}/metrics_track_w.txt")

# ── figure: HV full / zoom / flat-water crop, Noisy vs base vs flat+decim ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

zy, zx, zs = 180, 260, 160
wy, wx, ws = 180, 408, 32
ims = [("Noisy", amp.astype(np.float64))] + [
    (n, scale_match(r[2], amp.astype(np.float64))) for n, r in runs.items()]
fig, axes = plt.subplots(4, len(ims), figsize=(3.8 * len(ims), 14.5))
nat = [n256, b256, dw] + [runs[n][2].reshape(4, 256, 2, 256, 2).mean((2, 4))
                          for n in runs if n not in ("base", "flat+decim")]
vmax = np.quantile(amp[1], 0.99)
wmax = 3.0 * np.median(ims[1][1][1][wy:wy + ws, wx:wx + ws])
for col, (nm, im_) in enumerate(ims):
    v = np.clip(im_[1] / vmax, 0, 1)
    axes[0, col].imshow(v, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"{nm} — HV", fontsize=9)
    axes[1, col].imshow(v[zy:zy + zs, zx:zx + zs], cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[1, col].set_title("zoom — HV", fontsize=8)
    wc = im_[1][wy:wy + ws, wx:wx + ws]
    axes[2, col].imshow(wc, cmap="gray", vmin=0, vmax=wmax, interpolation="nearest")
    axes[2, col].set_title(f"flat water  std={wc.std():.2f} "
                           f"CV={wc.std() / max(wc.mean(), 1e-9):.3f}", fontsize=8)
for col, nm_ in enumerate(nat):
    v = np.clip(nm_[1][zy // 2: zy // 2 + zs // 2, zx // 2: zx // 2 + zs // 2] / vmax, 0, 1)
    axes[3, col].imshow(v, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[3, col].set_title("native 256 zoom (base = 2x2 block mean)", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/compare_track_w.png", dpi=130)
print(f"wrote {OUT}/compare_track_w.png")
