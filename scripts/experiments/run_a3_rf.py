"""A3: receptive-field expansion on top of the A1/A2 winner base
(dropout_style="pixel", norm="group").  Variants: wavelet_levels 2 / 3
(scale-recurrent shared detail CNN; RF 68 -> 136 -> 256+ px, params
unchanged) and levels=2 + dilated detail CNN [1,2,4,8,1].

Same two protocols as run_arch_ablation.py; the "pixel+group" (levels=1)
runs are reused from its cache as the reference row.
"""
import os, sys
import numpy as np

sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps, _local_coherence
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
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


# ── synthetic substrate (identical to run_arch_ablation) ──
clean = np.load(f"{OUT}/denoised_baseline.npy").astype(np.float64)
xpol = 0.5 * (clean[1] + clean[2]); clean[1] = xpol; clean[2] = xpol
u = np.exp(2j * pha.astype(np.float64))
target_coh = float(_local_coherence(u[1] * np.conj(u[2]), 7).mean())
rng = np.random.default_rng(7)
def cn(shape, sigma):
    return rng.normal(0, sigma/np.sqrt(2), shape) + 1j*rng.normal(0, sigma/np.sqrt(2), shape)
def simulate(sigma_n):
    g = np.stack([cn(clean[0].shape, 1.0) for _ in range(4)]); g[2] = g[1]
    return clean * g + np.stack([cn(clean[0].shape, sigma_n) for _ in range(4)])
def sim_coh(sigma_n):
    z = simulate(sigma_n); uu = np.exp(2j*np.angle(z))
    return float(_local_coherence(uu[1]*np.conj(uu[2]), 7).mean())
rms_x = float(np.sqrt((clean[1]**2).mean()))
lo, hi = 0.05*rms_x, 3.0*rms_x
for _ in range(12):
    mid = 0.5*(lo+hi)
    lo, hi = (mid, hi) if sim_coh(mid) > target_coh else (lo, mid)
sigma_n = 0.5*(lo+hi)
rng = np.random.default_rng(7)
z = simulate(sigma_n)
amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(z.real).astype(np.float32),
                      np.abs(z.imag).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)

BASE = {"dropout_style": "pixel", "norm": "group"}
STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, polish=0.5, edge_boost=1.0)
VARIANTS = [
    ("pg-lv1",     dict(BASE)),                       # cached from arch run
    ("pg-lv2",     dict(BASE, wavelet_levels=2)),
    ("pg-lv3",     dict(BASE, wavelet_levels=3)),
    ("pg-lv2-dil", dict(BASE, wavelet_levels=2,
                        low_freq_dilations=[1, 2, 4, 8, 1])),
]

runs_s, runs_r = {}, {}
for name, mc in VARIANTS:
    tag = ("arch_synth_pixel_group.npy" if name == "pg-lv1"
           else f"a3_synth_{name.replace('-', '_')}.npy")
    print(f"\n=== synth {name} ===", flush=True)
    runs_s[name] = cached(tag, lambda: denoise(
        amp_sim, TrainConfig(**STACK, whiteness_lags=(1, 2, 3), model_cfg=mc),
        ri_pair=ri_sim, pha=maps_sim)["denoised"]).astype(np.float64)
    tag = ("arch_real_pixel_group.npy" if name == "pg-lv1"
           else f"a3_real_{name.replace('-', '_')}.npy")
    print(f"\n=== real {name} ===", flush=True)
    runs_r[name] = cached(tag, lambda: denoise(
        amp, TrainConfig(**STACK, whiteness_lags=(3, 4, 5), model_cfg=mc),
        ri_pair=ri, pha=maps_real)["denoised"]).astype(np.float64)

# ── tables (same formats as run_arch_ablation) ──
def scale_match(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        s = float((x[c]*ref[c]).sum() / max((x[c]**2).sum(), 1e-9))
        out[c] = x[c]*s
    return out
def psnr(x, ref):
    return 10*np.log10(ref.max()**2 / max(((x-ref)**2).mean(), 1e-12))

lines = []
rois_s, rs_s = find_top_k_rois(amp_sim[1].astype(np.float64))
hdr = ("  {:<12}".format("Method") + "".join(f"{c:>11}" for c in
       ["PSNR(HH)", "PSNR(HV)", "SSIM(HH)", "SSIM(HV)", "EPI(HH)",
        "EPI(HV)", "ENL(HH)", "ENL(HV)"]))
print("\nSynthetic-GT (vs known clean; scale-matched)")
print(hdr); lines += ["Synthetic-GT (vs known clean; scale-matched)", hdr]
for name, d in runs_s.items():
    ds = scale_match(d, clean)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[0], ds[0]), ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1]),
            enl_roi_multi(d[0], rois_s, rs_s), enl_roi_multi(d[1], rois_s, rs_s)]
    line = "  {:<12}".format(name) + "".join(f"{v:>11.4f}" for v in vals)
    print(line); lines.append(line)

rois_r, rs_r = find_top_k_rois(amp[1])
def ratio_enl(d, c):
    eps = 1e-3
    rI = (amp[c].astype(np.float64)**2 + eps) / (d[c]**2 + eps)
    v = rI[(d[c] > 2) & (amp[c] > 0)]
    return (v.mean()/v.std())**2
hdr = ("  {:<12}".format("Method") + "".join(f"{c:>12}" for c in
       ["corr(HV,VH)", "ENL-ROI(HH)", "ENL-ROI(HV)", "EPI(HH)", "EPI(HV)",
        "ENLr(HH)", "ENLr(HV)"]))
print("\nReal patch (noisy-reference; ratio-ENL ideal ~= 1)")
print(hdr); lines += ["", "Real patch (noisy-reference; ratio-ENL ideal ~= 1)", hdr]
for name, d in runs_r.items():
    rec = reciprocity_metrics(d[1], d[2])
    vals = [rec["corr"], enl_roi_multi(d[0], rois_r, rs_r),
            enl_roi_multi(d[1], rois_r, rs_r),
            epi_metric(amp[0], d[0]), epi_metric(amp[1], d[1]),
            ratio_enl(d, 0), ratio_enl(d, 1)]
    line = "  {:<12}".format(name) + "".join(f"{v:>12.4f}" for v in vals)
    print(line); lines.append(line)

with open(f"{OUT}/metrics_a3_rf.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = ["Noisy"] + [n for n, _ in VARIANTS]
imgs = [amp] + [runs_r[n] for n, _ in VARIANTS]
fig, axes = plt.subplots(2, len(names), figsize=(4.2*len(names), 9.0))
for col, (nm, im_) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im_[ch] / np.quantile(amp[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}", fontsize=10)
        axes[r_i, col].axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/compare_a3_rf.png", dpi=130)
print("\nSaved", f"{OUT}/compare_a3_rf.png")
