"""Winner validation on REAL data: v3-L1+gate (MERLIN L1 + CV gate,
tv_mult=10) vs saved baseline / MERLIN-v2 — metrics + ratio-ENL + figure."""
import os
import sys

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")

from scipy.ndimage import gaussian_filter
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
amp, ri = load_quadpol_tiffs("data/tiff")

cfg = TrainConfig(iters=700, ri_mode="merlin", merlin_loss="l1",
                  tv_mult=10.0, guide_cv_protect=0.3)
res = denoise(amp, cfg, ri_pair=ri)
np.save(os.path.join(OUT, "denoised_v3_L1_gate.npy"), res["denoised"])

runs = {
    "baseline": np.load(os.path.join(OUT, "denoised_baseline.npy")),
    "MERLIN-v2": np.load(os.path.join(OUT, "denoised_MERLIN_v2.npy")),
    "v3-L1+gate": res["denoised"],
}

rois, rs = find_top_k_rois(amp[1])


def ratio_enl(d, c):
    eps = 1e-3
    m = (d[c] > 2) & (amp[c] > 0)
    rI = (amp[c].astype(np.float64) ** 2 + eps) / (d[c].astype(np.float64) ** 2 + eps)
    v = rI[m]
    return (v.mean() / v.std()) ** 2


print("Real-data metrics")
hdr = "  {:<12}".format("Method") + "".join(
    f"{c:>13}" for c in ["corr(HV,VH)", "ENL-ROI(HV)", "ENL-ROI(HH)",
                         "EPI(HV)", "SSIM(HV)", "ENLr(HH)", "ENLr(HV)"])
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
with open(os.path.join(OUT, "metrics_real_v3.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["Noisy", "baseline", "MERLIN-v2", "v3-L1+gate"]
imgs = [amp] + list(runs.values())
fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
for col, (nm, im_) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im_[ch] / np.quantile(amp[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}")
        axes[r_i, col].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_real_v3.png"), dpi=130)
print("Saved", os.path.join(OUT, "compare_real_v3.png"))
