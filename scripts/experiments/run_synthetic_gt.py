"""Ground-truth validation: known clean image + simulated 1-look complex
speckle -> denoise -> TRUE PSNR / SSIM / EPI measured against the clean.

Clean proxy: the baseline denoised result (realistic PolSAR structure).
Speckle: z = clean * (g_r + j g_i), g ~ N(0, 1/2) i.i.d. -> Rayleigh amp,
which matches the pipeline's own speckle model, and gives genuine Re/Im
for the MERLIN path.
"""
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

clean = np.load(os.path.join(OUT, "denoised_baseline.npy")).astype(np.float64)
# enforce HV == VH in the clean so reciprocity physics holds exactly
xpol = 0.5 * (clean[1] + clean[2])
clean[1] = xpol
clean[2] = xpol

g_r = rng.normal(0.0, np.sqrt(0.5), clean.shape)
g_i = rng.normal(0.0, np.sqrt(0.5), clean.shape)
# reciprocity: HV and VH share the clean signal but carry independent noise
re, im = clean * g_r, clean * g_i
amp_sim = np.sqrt(re ** 2 + im ** 2).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(re).astype(np.float32),
                      np.abs(im).astype(np.float32), mode="l1")

variants = {
    "baseline": dict(cfg=TrainConfig(iters=700), ri=None),
    "MERLIN-v2": dict(cfg=TrainConfig(iters=700, ri_mode="merlin",
                                      tv_mult=5.0), ri=ri_sim),
}

results = {"Noisy": amp_sim.astype(np.float64)}
for name, v in variants.items():
    print(f"\n=== synthetic {name} ===")
    res = denoise(amp_sim, v["cfg"], ri_pair=v["ri"])
    results[name] = res["denoised"].astype(np.float64)
    np.save(os.path.join(OUT, f"synth_{name.replace('-', '_')}.npy"),
            res["denoised"])
np.save(os.path.join(OUT, "synth_clean.npy"), clean.astype(np.float32))
np.save(os.path.join(OUT, "synth_noisy.npy"), amp_sim)


def scale_match(x, ref):
    """Per-channel least-squares scale (removes the global scale nuisance
    identically for every method)."""
    out = x.copy()
    for c in range(x.shape[0]):
        s = float((x[c] * ref[c]).sum() / max((x[c] ** 2).sum(), 1e-9))
        out[c] = x[c] * s
    return out


def psnr(x, ref):
    mse = ((x - ref) ** 2).mean()
    return 10 * np.log10(ref.max() ** 2 / max(mse, 1e-12))


print("\nGround-truth metrics (vs. the KNOWN clean; scale-matched)")
hdr = "  {:<10}".format("Method") + "".join(
    f"{c:>13}" for c in ["PSNR(HH)", "PSNR(HV)", "SSIM*(HH)", "SSIM*(HV)",
                         "EPI*(HH)", "EPI*(HV)"])
print(hdr)
lines = [hdr]
for name, d in results.items():
    ds = scale_match(d, clean)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[0], ds[0]), ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1])]
    line = "  {:<10}".format(name) + "".join(f"{v:>13.4f}" for v in vals)
    print(line)
    lines.append(line)
with open(os.path.join(OUT, "metrics_synthetic.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["Clean (GT)", "Noisy", "baseline", "MERLIN-v2"]
imgs = [clean, results["Noisy"], results["baseline"], results["MERLIN-v2"]]
fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
for col, (nm, im_) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im_[ch] / np.quantile(clean[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}")
        axes[r_i, col].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_synthetic.png"), dpi=130)
print("Saved", os.path.join(OUT, "compare_synthetic.png"))
