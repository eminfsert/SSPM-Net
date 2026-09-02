"""Track D — feeding the cross-pol channel with the phase evidence.

Base = pixel+group + xpol_pair_input (the revised Track C winner).
Variants (full recommended stack, tv_mult=10, each knob on top of the base):
  pg          : pixel+group only (Track C reference, cached tvsweep tv10)
  base        : + xpol_pair_input                       (Track C revised winner)
  D1f         : xpol_fused_target  (mean of the HV/VH opposite-component
                targets instead of a loss mixture)
  D1f+tdb.5   : + xpol_target_debias 0.5   (target-domain thermal debias)
  D1f+tdb1    : + xpol_target_debias 1.0
  D2          : model_cfg xpol_snr_input (phase snr map as network input)
  D3          : phase_helix_protect 0.5 (xpol structure protection)
  D4          : fact_snr_gate 1.0 (Rayleigh terms gated on the floor)
  combo       : winners combined (decided after the single-knob rows; the
                script takes the set from COMBO below)

Protocols identical to run_hv_track.py (same clean proxy cache, same
seed/noise).  NEW mandatory column (Track C lesson): flat-water high-pass
std / CV — the grain amplitude in a homogeneous dark region, which the
noisy-reference EPI + ratio-ENL pair is blind to.  All runs .npy-cached.
"""
import os, sys
import numpy as np

sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps, _local_coherence, _local_mean
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
os.makedirs(OUT, exist_ok=True)

amp, ri = load_quadpol_tiffs("data/tiff")
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


# ── clean proxy + simulated speckle: EXACTLY as run_tvmult_sweep.py ──
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
    n = np.stack([cn(clean[0].shape, sigma_n) for _ in range(4)])
    return clean * g + n


def sim_coh(sigma_n):
    z = simulate(sigma_n)
    uu = np.exp(2j * np.angle(z))
    return float(_local_coherence(uu[1] * np.conj(uu[2]), 7).mean())


rms_x = float(np.sqrt((clean[1] ** 2).mean()))
lo, hi = 0.05 * rms_x, 3.0 * rms_x
for _ in range(12):
    mid = 0.5 * (lo + hi)
    if sim_coh(mid) > target_coh:
        lo = mid
    else:
        hi = mid
sigma_n = 0.5 * (lo + hi)
print(f"sigma_n={sigma_n:.3f}  sim coherence={sim_coh(sigma_n):.3f} "
      f"(target {target_coh:.3f})", flush=True)

rng = np.random.default_rng(7)
z = simulate(sigma_n)
amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(z.real).astype(np.float32),
                      np.abs(z.imag).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)

# ── flat-water mask (grain column) ──
# Homogeneous dark region of the REAL patch: dark on the smoothed 2-look
# cross-pol mean AND low local CV of the 8-look span, eroded 5x5 so no
# edge pixel survives. The clean proxy is a denoise of the same patch, so
# the same mask is valid for the synthetic protocol.
# (the 8-look span's CV floor is ~0.30-0.33 on this oversampled data, so
# the homogeneity threshold is 0.40; 21x21 statistics; ~11.6k px, matching
# the Track C revision mask)
span8 = np.sqrt((ri.astype(np.float64) ** 2).mean(axis=(0, 1)))
mu_s = _local_mean(span8, 21)
cv_s = np.sqrt(np.maximum(_local_mean(span8 ** 2, 21) - mu_s ** 2, 0)) / (mu_s + 1e-6)
x2 = 0.5 * (ri[:, 1] + ri[:, 2]).mean(axis=0).astype(np.float64)
x2s = _local_mean(x2, 21)
m0 = (x2s < np.percentile(x2s, 20)) & (cv_s < 0.40)
water = _local_mean(m0.astype(np.float64), 5) > 0.999
print(f"water mask: {int(water.sum())} px "
      f"(HV noisy mean {amp[1][water].mean():.2f}, "
      f"co-pol mean {amp[0][water].mean():.2f})", flush=True)
np.save(f"{OUT}/water_mask.npy", water)


def grain(d, c=1):
    """flat-water high-pass std (grain amplitude) and CV of channel c."""
    hp = d[c] - _local_mean(d[c].astype(np.float64), 5)
    return float(hp[water].std()), float(d[c][water].std() / max(d[c][water].mean(), 1e-9))


# ── variants ──
PG = {"dropout_style": "pixel", "norm": "group"}
BASE_M = {**PG, "xpol_pair_input": True}
STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, polish=0.5, edge_boost=1.0)
D1 = {"xpol_fused_target": True}
VARIANTS = [
    # (name, cache_tag, TrainConfig extras, model extras)
    ("pg",         "tvsweep_{p}_tv10", {}, PG),
    ("base",       "hv_{p}_pairin", {}, BASE_M),
    ("D1f",        "hvp_{p}_d1f", D1, BASE_M),
    ("D1f+tdb.5",  "hvp_{p}_d1f_tdb05", {**D1, "xpol_target_debias": 0.5}, BASE_M),
    ("D1f+tdb1",   "hvp_{p}_d1f_tdb10", {**D1, "xpol_target_debias": 1.0}, BASE_M),
    ("D2",         "hvp_{p}_d2", {}, {**BASE_M, "xpol_snr_input": True}),
    ("D3",         "hvp_{p}_d3", {"phase_helix_protect": 0.5}, BASE_M),
    ("D4",         "hvp_{p}_d4", {"fact_snr_gate": 1.0}, BASE_M),
]
# combination rows (edit after reading the single-knob rows; kept explicit
# so the cache tags stay stable)
COMBO = [
    ("D1f+D2",     "hvp_{p}_d1f_d2", D1, {**BASE_M, "xpol_snr_input": True}),
    ("D1f+D3+D4",  "hvp_{p}_d1f_d3_d4",
     {**D1, "phase_helix_protect": 0.5, "fact_snr_gate": 1.0}, BASE_M),
    ("D1f+tdb.5+D2+D3+D4", "hvp_{p}_dall",
     {**D1, "xpol_target_debias": 0.5, "phase_helix_protect": 0.5,
      "fact_snr_gate": 1.0}, {**BASE_M, "xpol_snr_input": True}),
]
if "--combo" in sys.argv:
    VARIANTS += COMBO

runs_s, runs_r = {}, {}
for name, tag, extra, mcfg in VARIANTS:
    print(f"\n=== synth {name} ===", flush=True)
    runs_s[name] = cached(tag.format(p="synth") + ".npy", lambda: denoise(
        amp_sim, TrainConfig(**{**STACK, **extra}, whiteness_lags=(1, 2, 3),
                             model_cfg=mcfg),
        ri_pair=ri_sim, pha=maps_sim)["denoised"]).astype(np.float64)
    print(f"\n=== real {name} ===", flush=True)
    runs_r[name] = cached(tag.format(p="real") + ".npy", lambda: denoise(
        amp, TrainConfig(**{**STACK, **extra}, whiteness_lags=(3, 4, 5),
                         model_cfg=mcfg),
        ri_pair=ri, pha=maps_real)["denoised"]).astype(np.float64)


# ── tables ──
def scale_match(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        s = float((x[c] * ref[c]).sum() / max((x[c] ** 2).sum(), 1e-9))
        out[c] = x[c] * s
    return out


def psnr(x, ref):
    return 10 * np.log10(ref.max() ** 2 / max(((x - ref) ** 2).mean(), 1e-12))


def dark_rel_rmse(d, ref):
    """relative RMSE of HV in the darkest 20% (thermal-dominated) pixels."""
    c, o = ref[1], d[1]
    s = float((o * c).sum() / max((o ** 2).sum(), 1e-9))
    o = o * s
    m = c < np.percentile(c, 20)
    return float(np.sqrt(((o - c) ** 2)[m].mean()) / max(c[m].mean(), 1e-9))


def dark_bias(d, ref):
    """mean signed error of HV (scale-matched) in the darkest 20%."""
    c, o = ref[1], d[1]
    s = float((o * c).sum() / max((o ** 2).sum(), 1e-9))
    m = c < np.percentile(c, 20)
    return float((o * s - c)[m].mean())


lines = []
rois_s, rs_s = find_top_k_rois(amp_sim[1].astype(np.float64))
W = 11
cols = ["PSNR(HH)", "PSNR(HV)", "SSIM(HV)", "EPI(HH)", "EPI(HV)", "ENL(HV)",
        "darkRelHV", "darkBias", "waterHP", "waterCV"]
hdr = "  {:<20}".format("Method") + "".join(f"{c:>{W}}" for c in cols)
title = ("Synthetic-GT (vs known clean; scale-matched); base = pixel+group"
         " + xpol_pair_input; waterHP = flat-water high-pass std of HV"
         f" (clean HV: {grain(clean)[0]:.4f}, CV {grain(clean)[1]:.4f})")
print("\n" + title); print(hdr)
lines += [title, hdr]


def synth_row(name, d):
    ds = scale_match(d, clean)
    g = grain(ds)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1]),
            enl_roi_multi(d[1], rois_s, rs_s), dark_rel_rmse(d, clean),
            dark_bias(d, clean), g[0], g[1]]
    line = "  {:<20}".format(name) + "".join(f"{v:>{W}.4f}" for v in vals)
    print(line, flush=True); lines.append(line)


for name, d in runs_s.items():
    synth_row(name, d)

rois_r, rs_r = find_top_k_rois(amp[1])


def ratio_enl(d, c):
    eps = 1e-3
    rI = (amp[c].astype(np.float64) ** 2 + eps) / (d[c] ** 2 + eps)
    v = rI[(d[c] > 2) & (amp[c] > 0)]
    return (v.mean() / v.std()) ** 2


cols = ["corr(HV,VH)", "ENL-ROI(HV)", "EPI(HH)", "EPI(HV)", "ENLr(HH)",
        "ENLr(HV)", "HVdarkMean", "waterHP", "waterCV"]
W2 = 12
hdr = "  {:<20}".format("Method") + "".join(f"{c:>{W2}}" for c in cols)
title = ("Real patch (noisy-reference; ratio-ENL ideal ~= 1; HVdarkMean = "
         "mean HV output on the water mask; waterHP/CV = grain)")
print("\n" + title); print(hdr); lines += ["", title, hdr]


def real_row(name, d):
    rec = reciprocity_metrics(d[1], d[2])
    g = grain(d)
    vals = [rec["corr"], enl_roi_multi(d[1], rois_r, rs_r),
            epi_metric(amp[0], d[0]), epi_metric(amp[1], d[1]),
            ratio_enl(d, 0), ratio_enl(d, 1), float(d[1][water].mean()),
            g[0], g[1]]
    line = "  {:<20}".format(name) + "".join(f"{v:>{W2}.4f}" for v in vals)
    print(line, flush=True); lines.append(line)


for name, d in runs_r.items():
    real_row(name, d)

with open(f"{OUT}/metrics_hv_phase.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

# ── figures: HV full + zoom + water crop, real patch ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
show = [n for n in runs_r if n != "pg"]
zy, zx, zs = 180, 260, 160
wy, wx, ws = 180, 408, 32            # the archived flat-water crop
fig, axes = plt.subplots(3, len(show) + 1, figsize=(3.6 * (len(show) + 1), 11))
ims = [("Noisy", amp.astype(np.float64))] + [(n, runs_r[n]) for n in show]
vmax = np.quantile(amp[1], 0.99)
wmax = 3.0 * np.median(runs_r["base"][1][wy:wy + ws, wx:wx + ws])
for col, (nm, im_) in enumerate(ims):
    v = np.clip(im_[1] / vmax, 0, 1)
    axes[0, col].imshow(v, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"{nm} — HV", fontsize=9)
    axes[1, col].imshow(v[zy:zy + zs, zx:zx + zs], cmap="gray",
                        vmin=0, vmax=1, interpolation="nearest")
    axes[1, col].set_title("zoom — HV", fontsize=8)
    wc = im_[1][wy:wy + ws, wx:wx + ws]
    axes[2, col].imshow(wc, cmap="gray", vmin=0, vmax=wmax, interpolation="nearest")
    axes[2, col].set_title(f"water crop std={wc.std():.2f} "
                           f"CV={wc.std() / max(wc.mean(), 1e-9):.3f}", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/compare_hv_phase.png", dpi=130)

# helix map figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
axes[0].imshow(np.clip(amp[1] / vmax, 0, 1), cmap="gray"); axes[0].set_title("noisy HV")
axes[1].imshow(maps_real["helix"], cmap="magma", vmin=0, vmax=1); axes[1].set_title("helix map (real)")
axes[2].imshow(maps_real["snr"], cmap="magma", vmin=0, vmax=1); axes[2].set_title("snr map (real)")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/phase_helix_map.png", dpi=120)
print("\nSaved", f"{OUT}/compare_hv_phase.png", f"{OUT}/phase_helix_map.png")
