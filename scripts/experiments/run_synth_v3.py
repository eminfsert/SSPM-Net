"""Synthetic-GT evaluation of round-3 variants:
  v3-NLL+gate : MERLIN NLL loss + Lee-style CV gate (tv_mult=5)
  v3-L1+gate  : MERLIN L1 loss + CV gate, stronger TV (tv_mult=10)
against the saved synthetic baseline / MERLIN-v2 results."""
import os
import sys

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")

from sspmnet import denoise, TrainConfig
from sspmnet.complex_data import calibrate_ri
from sspmnet.metrics import epi_metric, ssim_metric

OUT = "results/ri_compare"
rng = np.random.default_rng(7)

clean = np.load(os.path.join(OUT, "synth_clean.npy")).astype(np.float64)
g_r = rng.normal(0.0, np.sqrt(0.5), clean.shape)
g_i = rng.normal(0.0, np.sqrt(0.5), clean.shape)
re, im = clean * g_r, clean * g_i
amp_sim = np.sqrt(re ** 2 + im ** 2).astype(np.float32)
saved = np.load(os.path.join(OUT, "synth_noisy.npy"))
assert np.allclose(amp_sim, saved, atol=1e-4), "synthetic regen mismatch!"
ri_sim = calibrate_ri(amp_sim, np.abs(re).astype(np.float32),
                      np.abs(im).astype(np.float32), mode="l1")

variants = {
    "v3-NLL+gate": TrainConfig(iters=700, ri_mode="merlin", merlin_loss="nll",
                               tv_mult=5.0, guide_cv_protect=0.3),
    "v3-L1+gate": TrainConfig(iters=700, ri_mode="merlin", merlin_loss="l1",
                              tv_mult=10.0, guide_cv_protect=0.3),
}

results = {
    "Noisy": amp_sim.astype(np.float64),
    "baseline": np.load(os.path.join(OUT, "synth_baseline.npy")).astype(np.float64),
    "MERLIN-v2": np.load(os.path.join(OUT, "synth_MERLIN_v2.npy")).astype(np.float64),
}
for name, cfg in variants.items():
    print(f"\n=== synthetic {name} ===")
    res = denoise(amp_sim, cfg, ri_pair=ri_sim)
    results[name] = res["denoised"].astype(np.float64)
    np.save(os.path.join(OUT, f"synth_{name.replace('-', '_').replace('+', '_')}.npy"),
            res["denoised"])


def scale_match(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        s = float((x[c] * ref[c]).sum() / max((x[c] ** 2).sum(), 1e-9))
        out[c] = x[c] * s
    return out


def psnr(x, ref):
    mse = ((x - ref) ** 2).mean()
    return 10 * np.log10(ref.max() ** 2 / max(mse, 1e-12))


print("\nGround-truth metrics (vs. the KNOWN clean; scale-matched)")
hdr = "  {:<13}".format("Method") + "".join(
    f"{c:>12}" for c in ["PSNR(HH)", "PSNR(HV)", "SSIM*(HH)", "SSIM*(HV)",
                         "EPI*(HH)", "EPI*(HV)"])
print(hdr)
lines = [hdr]
for name, d in results.items():
    ds = scale_match(d, clean)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[0], ds[0]), ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1])]
    line = "  {:<13}".format(name) + "".join(f"{v:>12.4f}" for v in vals)
    print(line)
    lines.append(line)
with open(os.path.join(OUT, "metrics_synth_v3.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["Clean (GT)", "baseline", "MERLIN-v2", "v3-NLL+gate", "v3-L1+gate"]
imgs = [clean] + [results[n] for n in names[1:]]
fig, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 9.3))
for col, (nm, im_) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im_[ch] / np.quantile(clean[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}")
        axes[r_i, col].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_synth_v3.png"), dpi=130)
print("Saved", os.path.join(OUT, "compare_synth_v3.png"))
