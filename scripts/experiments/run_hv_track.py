"""Track C — cross-pol (HV) deep-work ablation on the pixel+group base.

Variants (full recommended stack, tv_mult=10):
  base        : cached from the tv_mult sweep (tvsweep_*_tv10.npy)
  pairin      : C1  model_cfg xpol_pair_input=True (xpol branch sees BOTH
                reciprocal planes; +0.12% params)
  polgroup    : C2  polgroup_guides=True (TV/NLM/polish guides split by
                polarization group + per-channel whiteness)
  pair+group  : C1+C2
  wr25 / wr75 : C4  merlin_recip_weight sweep (never swept; default 0.5)
  fid75       : C4  phase_fidelity 0.5 -> 0.75
  + thermal_debias (C3) applied POST-HOC to the cached outputs (it is a
    pure output-side transform), t in {0.5, 1.0}, on base and the winner.

Protocols identical to run_tvmult_sweep.py (same clean proxy cache, same
seed/noise) + an HV-focused table: dark-bin relative RMSE on synthetic GT.
All runs .npy-cached.
"""
import os, sys
import numpy as np

sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase
from sspmnet.complex_data import calibrate_ri, estimate_thermal_sigma
from sspmnet.phase_data import phase_feedback_maps, _local_coherence
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

# ── variants ──
PG = {"dropout_style": "pixel", "norm": "group"}
STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, polish=0.5, edge_boost=1.0)
VARIANTS = [
    # (name, cache_tag or None->reuse tvsweep, TrainConfig extras, model extras)
    ("base",       "tvsweep_{p}_tv10", {}, {}),
    ("pairin",     "hv_{p}_pairin", {}, {"xpol_pair_input": True}),
    ("polgroup",   "hv_{p}_polgroup", {"polgroup_guides": True}, {}),
    ("pair+group", "hv_{p}_pairgroup",
     {"polgroup_guides": True}, {"xpol_pair_input": True}),
    ("wr25",       "hv_{p}_wr25", {"merlin_recip_weight": 0.25}, {}),
    ("wr75",       "hv_{p}_wr75", {"merlin_recip_weight": 0.75}, {}),
    ("fid75",      "hv_{p}_fid75", {"phase_fidelity": 0.75}, {}),
]

runs_s, runs_r = {}, {}
for name, tag, extra, mextra in VARIANTS:
    print(f"\n=== synth {name} ===", flush=True)
    runs_s[name] = cached(tag.format(p="synth") + ".npy", lambda: denoise(
        amp_sim, TrainConfig(**STACK, **extra, whiteness_lags=(1, 2, 3),
                             model_cfg={**PG, **mextra}),
        ri_pair=ri_sim, pha=maps_sim)["denoised"]).astype(np.float64)
    print(f"\n=== real {name} ===", flush=True)
    runs_r[name] = cached(tag.format(p="real") + ".npy", lambda: denoise(
        amp, TrainConfig(**STACK, **extra, whiteness_lags=(3, 4, 5),
                         model_cfg={**PG, **mextra}),
        ri_pair=ri, pha=maps_real)["denoised"]).astype(np.float64)


def debias(d, amp_src, snr, t):
    s_th = estimate_thermal_sigma(amp_src, snr)
    out = d.copy()
    for c in (1, 2):
        out[c] = np.sqrt(np.maximum(out[c] ** 2 - t * s_th ** 2, 0.0))
    return out


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


lines = []
rois_s, rs_s = find_top_k_rois(amp_sim[1].astype(np.float64))
hdr = ("  {:<12}".format("Method") + "".join(
    f"{c:>11}" for c in ["PSNR(HH)", "PSNR(HV)", "SSIM(HV)", "EPI(HH)",
                         "EPI(HV)", "ENL(HV)", "darkRelHV"]))
print("\nSynthetic-GT (vs known clean; scale-matched); pixel+group base")
print(hdr)
lines += ["Synthetic-GT (vs known clean; scale-matched); pixel+group base", hdr]


def synth_row(name, d):
    ds = scale_match(d, clean)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1]),
            enl_roi_multi(d[1], rois_s, rs_s), dark_rel_rmse(d, clean)]
    line = "  {:<12}".format(name) + "".join(f"{v:>11.4f}" for v in vals)
    print(line, flush=True); lines.append(line)


for name, d in runs_s.items():
    synth_row(name, d)
best = "pair+group"        # provisional; debias rows show both anyway
for t in (0.5, 1.0):
    synth_row(f"base+db{t:g}",
              debias(runs_s["base"], amp_sim, maps_sim["snr"], t))
    synth_row(f"{best}+db{t:g}",
              debias(runs_s[best], amp_sim, maps_sim["snr"], t))

rois_r, rs_r = find_top_k_rois(amp[1])


def ratio_enl(d, c):
    eps = 1e-3
    rI = (amp[c].astype(np.float64) ** 2 + eps) / (d[c] ** 2 + eps)
    v = rI[(d[c] > 2) & (amp[c] > 0)]
    return (v.mean() / v.std()) ** 2


hdr = ("  {:<12}".format("Method") + "".join(
    f"{c:>12}" for c in ["corr(HV,VH)", "ENL-ROI(HV)", "EPI(HH)",
                         "EPI(HV)", "ENLr(HH)", "ENLr(HV)"]))
print("\nReal patch (noisy-reference; ratio-ENL ideal ~= 1)")
print(hdr); lines += ["", "Real patch (noisy-reference; ratio-ENL ideal ~= 1)", hdr]


def real_row(name, d):
    rec = reciprocity_metrics(d[1], d[2])
    vals = [rec["corr"], enl_roi_multi(d[1], rois_r, rs_r),
            epi_metric(amp[0], d[0]), epi_metric(amp[1], d[1]),
            ratio_enl(d, 0), ratio_enl(d, 1)]
    line = "  {:<12}".format(name) + "".join(f"{v:>12.4f}" for v in vals)
    print(line, flush=True); lines.append(line)


for name, d in runs_r.items():
    real_row(name, d)
for t in (0.5, 1.0):
    real_row(f"base+db{t:g}", debias(runs_r["base"], amp, maps_real["snr"], t))
    real_row(f"{best}+db{t:g}", debias(runs_r[best], amp, maps_real["snr"], t))

with open(f"{OUT}/metrics_hv_track.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
show = ["base", "pairin", "polgroup", "pair+group"]
zy, zx, zs = 180, 260, 160
fig, axes = plt.subplots(2, len(show) + 1, figsize=(4.2 * (len(show) + 1), 9.0))
ims = [("Noisy", amp)] + [(n, runs_r[n]) for n in show]
for col, (nm, im_) in enumerate(ims):
    v = np.clip(im_[1] / np.quantile(amp[1], 0.99), 0, 1)
    axes[0, col].imshow(v, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"{nm} — HV", fontsize=10)
    axes[1, col].imshow(v[zy:zy + zs, zx:zx + zs], cmap="gray",
                        vmin=0, vmax=1, interpolation="nearest")
    axes[1, col].set_title(f"zoom — HV", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/compare_hv_track.png", dpi=130)
print("\nSaved", f"{OUT}/compare_hv_track.png")
