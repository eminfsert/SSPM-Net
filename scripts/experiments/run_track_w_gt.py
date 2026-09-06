"""Track W, independent GT with SHAPED (spatially correlated) speckle.

The phantom protocol so far (run_indep_eval.py) simulated WHITE speckle, so
it could not see the correlated-speckle problem that Track W is about.  Here
the phantom SLC is built with the REAL transfer function estimated from the
bundled patch (sspmnet.spectral.estimate_transfer on the centred, unclipped
SLC), applied to both the speckle and the thermal noise, and the thermal
level is bisected so the simulated HV-VH coherence matches the real 0.77.
No clipping (this models the unclipped npy + phase path).

Rows (both through ``denoise(slc=...)``, pixel+group + xpol_pair_input, full
stack, 700 iters):
    slc       control — centred SLC input, MERLIN Re/Im pairs only
    slc+sub   + TrainConfig.sublook_n2n = 1.0 (W2)

Metrics vs the KNOWN phantom (LS scale-matched): PSNR / SSIM / EPI, ENL on
ROIs, EPI on the urban blocks, bright-tail RMSE, brightR, flatHP; plus the
ratio-image whiteness columns (rLag1 vs inLag1, rNeighR2 vs inNeighR2).

Usage: python scripts/experiments/run_track_w_gt.py      (.npy-cached)
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_slc, spectral
from sspmnet.phantom import make_phantom, make_phantom_slc, transfer_from_profiles
from sspmnet.phase_data import _local_mean
from sspmnet.metrics import find_top_k_rois, enl_roi_multi, epi_metric, ssim_metric

OUT = "results/track_w_gt"
CACHE = "results/ri_compare"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def cached(fname, fn):
    f = os.path.join(CACHE, fname)
    if os.path.exists(f):
        print(f"[cached] {fname}", flush=True)
        return np.load(f)
    out = fn()
    np.save(f, out)
    return out


# ── real patch: transfer function + HV-VH coherence target ──
slc_real = load_quadpol_slc("data/tiff", amp_npy="data/example_quadpol.npy")
slc_c, _ = spectral.centre_spectrum(slc_real)
py, px = spectral.estimate_transfer(slc_c)
Hf = transfer_from_profiles(py, px)


def _coh(x, y, w=7):
    from scipy.ndimage import uniform_filter
    n = (uniform_filter(np.real(x * np.conj(y)), w)
         + 1j * uniform_filter(np.imag(x * np.conj(y)), w))
    d = np.sqrt(uniform_filter(np.abs(x) ** 2, w) * uniform_filter(np.abs(y) ** 2, w))
    return float((np.abs(n) / np.maximum(d, 1e-9)).mean())


off = np.angle((slc_c[1] * np.conj(slc_c[2])).sum())
target_coh = _coh(slc_c[1], slc_c[2] * np.exp(1j * off))
real_white = spectral.speckle_whiteness(np.abs(slc_c[1]) ** 2)
print(f"real: HV-VH coherence {target_coh:.4f}; speckle lag1 "
      f"{real_white['lag1_x']:.3f}/{real_white['lag1_y']:.3f} neighR2 {real_white['neigh_r2']:.3f}",
      flush=True)

# ── phantom on the real amplitude scale, shaped speckle ──
q99 = float(np.quantile(np.abs(slc_real[0]), 0.99))
clean, masks = make_phantom(seed=0, q99=q99, return_masks=True)
clean = clean.astype(np.float64)
water, urban = masks["flat"], masks["urban"]

rms_x = float(np.sqrt((clean[1] ** 2).mean()))
lo, hi = 0.02 * rms_x, 3.0 * rms_x
for _ in range(14):
    mid = 0.5 * (lo + hi)
    zt = make_phantom_slc(clean, mid, transfer=Hf)
    lo, hi = (mid, hi) if _coh(zt[1], zt[2]) > target_coh else (lo, mid)
sigma_n = 0.5 * (lo + hi)
z = make_phantom_slc(clean, sigma_n, transfer=Hf).astype(np.complex64)
sim_white = spectral.speckle_whiteness(np.abs(z[1]) ** 2)
print(f"sim : sigma_n={sigma_n:.3f} HV-VH coherence {_coh(z[1], z[2]):.4f}; speckle lag1 "
      f"{sim_white['lag1_x']:.3f}/{sim_white['lag1_y']:.3f} neighR2 {sim_white['neigh_r2']:.3f}",
      flush=True)
noisy = np.abs(z).astype(np.float64)

PG = {"dropout_style": "pixel", "norm": "group", "xpol_pair_input": True}
STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, whiteness_lags=(3, 4, 5), polish=0.5,
             edge_boost=1.0, model_cfg=PG)

runs = {}
print("\n=== slc (control) ===", flush=True)
runs["slc"] = cached("trkwgt_slc.npy", lambda: denoise(
    None, TrainConfig(**STACK), slc=z)["denoised"]).astype(np.float64)
print("\n=== slc+sub ===", flush=True)
runs["slc+sub"] = cached("trkwgt_slc_sub.npy", lambda: denoise(
    None, TrainConfig(**STACK, sublook_n2n=1.0), slc=z)["denoised"]).astype(np.float64)


# ── metrics vs the KNOWN phantom ──
def scale_match(x, ref):
    o = x.copy()
    for c in range(x.shape[0]):
        o[c] = x[c] * float((x[c] * ref[c]).sum() / max((x[c] ** 2).sum(), 1e-9))
    return o


def psnr(x, ref):
    return 10 * np.log10(ref.max() ** 2 / max(((x - ref) ** 2).mean(), 1e-12))


def ratio_whiteness(d, c=1):
    r = (noisy[c] ** 2 + 1e-3) / (d[c] ** 2 + 1e-3)
    w = spectral.speckle_whiteness(r)
    w0 = spectral.speckle_whiteness(noisy[c] ** 2)
    return (0.5 * (w["lag1_x"] + w["lag1_y"]), 0.5 * (w0["lag1_x"] + w0["lag1_y"]),
            w["neigh_r2"], w0["neigh_r2"])


thr99 = np.quantile(clean, 0.99)
bright = clean >= thr99
rois, rs = find_top_k_rois(clean[1])
W = 11
cols = ["PSNR(HH)", "PSNR(HV)", "SSIM(HH)", "SSIM(HV)", "EPI(HH)", "EPI(HV)",
        "ENL(HV)", "EPIurb(HV)", "RMSEbrgt", "brightR", "flatHP",
        "rLag1", "inLag1", "rNeighR2", "inNeighR2"]
title = ("Track W independent GT, SHAPED speckle (real transfer function, HV-VH coherence "
         f"matched to {target_coh:.3f}; simulated speckle lag1 {sim_white['lag1_x']:.2f}/"
         f"{sim_white['lag1_y']:.2f} vs real {real_white['lag1_x']:.2f}/{real_white['lag1_y']:.2f}). "
         "All outputs LS scale-matched. flatHP = high-pass std of HV over the flat band "
         f"(clean {float((clean[1] - _local_mean(clean[1], 5))[water].std()):.4f}); "
         "rLag1/rNeighR2 = ratio-image whiteness vs the input's (ideal: equal)")
hdr = "  {:<10}".format("Method") + "".join(f"{c:>{W}}" for c in cols)
lines = [title, hdr]
print("\n" + title)
print(hdr)
for name, d in runs.items():
    ds = scale_match(d, clean)
    hp = float((ds[1] - _local_mean(ds[1], 5))[water].std())
    rw = ratio_whiteness(d)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[0], ds[0]), ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1]),
            enl_roi_multi(d[1], rois, rs),
            epi_metric(clean[1] * urban, ds[1] * urban),
            float(np.sqrt(((ds - clean) ** 2)[bright].mean())),
            float((ds[bright] / np.maximum(clean[bright], 1e-9)).mean()), hp,
            rw[0], rw[1], rw[2], rw[3]]
    line = "  {:<10}".format(name) + "".join(f"{v:>{W}.4f}" for v in vals)
    print(line, flush=True)
    lines.append(line)
da, db = scale_match(runs["slc"], clean), scale_match(runs["slc+sub"], clean)
s = (f"  W2 sub-look N2N vs control: dPSNR(HH) {psnr(db[0], clean[0]) - psnr(da[0], clean[0]):+.3f}  "
     f"dPSNR(HV) {psnr(db[1], clean[1]) - psnr(da[1], clean[1]):+.3f}  "
     f"dEPI(HV) {epi_metric(clean[1], db[1]) - epi_metric(clean[1], da[1]):+.4f}")
print("\n" + s)
lines += ["", s]
with open(f"{OUT}/metrics_track_w_gt.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {OUT}/metrics_track_w_gt.txt")

# ── figure: clean / noisy / slc / slc+sub — HV full, zoom on urban, flat crop ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ims = [("Clean", clean), ("Noisy (shaped)", noisy)] + [
    (n, scale_match(d, clean)) for n, d in runs.items()]
uy, ux = np.argwhere(urban).min(axis=0)
zs = 128
wy, wx = np.argwhere(water)[len(np.argwhere(water)) // 2]
ws = 32
wy, wx = int(np.clip(wy - ws // 2, 0, 512 - ws)), int(np.clip(wx - ws // 2, 0, 512 - ws))
fig, axes = plt.subplots(3, len(ims), figsize=(3.6 * len(ims), 11))
vmax = np.quantile(clean[1], 0.99)
wmax = 3.0 * float(clean[1][water].mean())
for col, (nm, im_) in enumerate(ims):
    v = np.clip(im_[1] / vmax, 0, 1)
    axes[0, col].imshow(v, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"{nm} — HV", fontsize=9)
    axes[1, col].imshow(v[uy:uy + zs, ux:ux + zs], cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[1, col].set_title("urban zoom — HV", fontsize=8)
    wc = im_[1][wy:wy + ws, wx:wx + ws]
    axes[2, col].imshow(wc, cmap="gray", vmin=0, vmax=wmax, interpolation="nearest")
    axes[2, col].set_title(f"flat band  std={wc.std():.2f} CV={wc.std() / max(wc.mean(), 1e-9):.3f}",
                           fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/compare_track_w_gt.png", dpi=130)
print(f"wrote {OUT}/compare_track_w_gt.png")
