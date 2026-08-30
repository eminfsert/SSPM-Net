"""Phase feedback on REAL data: baseline vs winner (MERLIN L1 tv10 + CV gate)
vs winner + phase feedback (HV-VH reciprocity coherence maps from the pha
TIFFs). Regenerates results/ri_compare/denoised_baseline.npy (clean proxy
for the synthetic protocol) after the VM reset."""
import os
import sys

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")

from sspmnet import (denoise, TrainConfig, load_quadpol_tiffs,
                     load_quadpol_phase)
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
os.makedirs(OUT, exist_ok=True)
amp, ri = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")

W = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
         guide_cv_protect=0.3)
variants = {
    "baseline":  (TrainConfig(iters=700), None, None),
    "winner":    (TrainConfig(**W), ri, None),
    "winner+PH": (TrainConfig(**W), ri, pha),
}

runs = {}
for name, (cfg, ri_a, pha_a) in variants.items():
    print(f"\n=== {name} ===", flush=True)
    res = denoise(amp, cfg, ri_pair=ri_a, pha=pha_a)
    runs[name] = res["denoised"]
    np.save(os.path.join(OUT, f"denoised_{name.replace('+', '_')}.npy"),
            res["denoised"])

rois, rs = find_top_k_rois(amp[1])


def ratio_enl(d, c):
    eps = 1e-3
    m = (d[c] > 2) & (amp[c] > 0)
    rI = (amp[c].astype(np.float64) ** 2 + eps) / (d[c].astype(np.float64) ** 2 + eps)
    v = rI[m]
    return (v.mean() / v.std()) ** 2


print("\nReal-data metrics (noisy-reference; ratio-ENL ideal ~= 1)")
cols = ["corr(HV,VH)", "ENL-ROI(HV)", "ENL-ROI(HH)", "EPI(HV)", "SSIM(HV)",
        "ENLr(HH)", "ENLr(HV)"]
hdr = "  {:<12}".format("Method") + "".join(f"{c:>13}" for c in cols)
print(hdr)
lines = [hdr]
for name, d in runs.items():
    rec = reciprocity_metrics(d[1], d[2])
    vals = [rec["corr"], enl_roi_multi(d[1], rois, rs),
            enl_roi_multi(d[0], rois, rs), epi_metric(amp[1], d[1]),
            ssim_metric(amp[1], d[1]), ratio_enl(d, 0), ratio_enl(d, 1)]
    line = "  {:<12}".format(name) + "".join(f"{v:>13.4f}" for v in vals)
    print(line)
    lines.append(line)
with open(os.path.join(OUT, "metrics_real_phase.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["Noisy"] + list(runs.keys())
imgs = [amp] + list(runs.values())
fig, axes = plt.subplots(2, len(names), figsize=(4.7 * len(names), 9.5))
for col, (nm, im_) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im_[ch] / np.quantile(amp[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}")
        axes[r_i, col].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_real_phase.png"), dpi=130)
print("Saved", os.path.join(OUT, "compare_real_phase.png"))
