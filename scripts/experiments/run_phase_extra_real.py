"""Real-data value of the two phase knobs that are OFF by default:
``phase_surface_boost`` (extra smoothing where the HH-VV co-pol coherence
says "surface / distributed scatterer") and ``phase_protect`` (less
smoothing where the single-channel spatial phase coherence says
"deterministic target").

Both are evaluated ON TOP of the full recommended stack (MERLIN + PH(b3,f0.5)
+ whiteness + polish + edge_boost), on the real urban patch — the synthetic
protocol cannot judge them because it simulates neither surface scattering
nor deterministic targets.

Besides the standard real-data table, ENL is reported STRATIFIED by the
surface map: homogeneous ROIs are split into the 10 with the highest and the
10 with the lowest mean 'surface' coherence, which isolates where the knob
actually acts. Runs are cached as .npy so an interrupted session resumes.
"""
import os, sys
import numpy as np

sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase
from sspmnet.phase_data import phase_feedback_maps
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
os.makedirs(OUT, exist_ok=True)

amp, ri = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")
maps = phase_feedback_maps(pha=pha, win=7)

FULL = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
            guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
            whiteness_lambda=0.05, whiteness_lags=(3, 4, 5),
            polish=0.5, edge_boost=1.0)

VARIANTS = [
    ("final",      {}),
    ("surf0.5",    dict(phase_surface_boost=0.5)),
    ("surf1.0",    dict(phase_surface_boost=1.0)),
    ("prot0.3",    dict(phase_protect=0.3)),
    ("prot0.6",    dict(phase_protect=0.6)),
]

runs = {}
for name, extra in VARIANTS:
    f = f"{OUT}/denoised_{name.replace('.', 'p')}.npy"
    if os.path.exists(f):
        print(f"=== {name} (cached) ===", flush=True)
        runs[name] = np.load(f)
        continue
    print(f"\n=== {name} {extra} ===", flush=True)
    res = denoise(amp, TrainConfig(**FULL, **extra), ri_pair=ri, pha=pha)
    runs[name] = res["denoised"]
    np.save(f, res["denoised"])

# ── ROIs: picked once on the noisy HV channel, then stratified by 'surface'
rois, rs = find_top_k_rois(amp[1], top_k=40)
surf = maps["surface"]
by_surf = sorted(rois, key=lambda ij: surf[ij[0]:ij[0]+rs, ij[1]:ij[1]+rs].mean())
lo_rois, hi_rois = by_surf[:10], by_surf[-10:]
rois10 = rois[:10]
print(f"\nsurface-stratified ROIs: low mean "
      f"{np.mean([surf[i:i+rs, j:j+rs].mean() for i, j in lo_rois]):.3f}, "
      f"high mean "
      f"{np.mean([surf[i:i+rs, j:j+rs].mean() for i, j in hi_rois]):.3f}")


def ratio_enl(d, c):
    eps = 1e-3
    rI = (amp[c].astype(np.float64) ** 2 + eps) / (d[c].astype(np.float64) ** 2 + eps)
    v = rI[(d[c] > 2) & (amp[c] > 0)]
    return (v.mean() / v.std()) ** 2


cols = ["corr(HV,VH)", "ENL-ROI(HH)", "ENL-ROI(HV)", "ENLsurf(HV)",
        "ENLflat(HV)", "EPI(HH)", "EPI(HV)", "SSIM(HV)", "ENLr(HH)", "ENLr(HV)"]
hdr = "  {:<10}".format("Method") + "".join(f"{c:>13}" for c in cols)
print("\nReal-patch metrics (noisy-reference EPI/SSIM; ratio-ENL ideal ~= 1)")
print(hdr)
lines = [hdr]
for name, d in runs.items():
    rec = reciprocity_metrics(d[1], d[2])
    vals = [rec["corr"], enl_roi_multi(d[0], rois10, rs),
            enl_roi_multi(d[1], rois10, rs),
            enl_roi_multi(d[1], hi_rois, rs), enl_roi_multi(d[1], lo_rois, rs),
            epi_metric(amp[0], d[0]), epi_metric(amp[1], d[1]),
            ssim_metric(amp[1], d[1]), ratio_enl(d, 0), ratio_enl(d, 1)]
    line = "  {:<10}".format(name) + "".join(f"{v:>13.4f}" for v in vals)
    print(line)
    lines.append(line)
with open(f"{OUT}/metrics_real_phase_extra.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = ["Noisy"] + list(runs.keys())
imgs = [amp] + list(runs.values())
fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 9.0))
for col, (nm, im_) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im_[ch] / np.quantile(amp[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}", fontsize=10)
        axes[r_i, col].axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/compare_phase_extra.png", dpi=130)
print("\nSaved", f"{OUT}/compare_phase_extra.png")
