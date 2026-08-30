"""Ground-truth validation of the phase feedback: known clean + simulated
1-look complex speckle WITH reciprocity physics -> TRUE PSNR / SSIM / EPI.

Simulation model (matches what the real patch exhibits):
    co-pol   : z_c  = clean_c * g_c + n_c          g ~ CN(0,1) i.i.d.
    cross-pol: z_hv = clean_x * g_x + n_hv         SHARED speckle g_x
               z_vh = clean_x * g_x + n_vh         (monostatic reciprocity)
    n ~ CN(0, sigma_n^2) i.i.d. thermal/system noise.

sigma_n is calibrated so the simulated HV-VH doubled-angle coherence matches
the value measured on the real pha TIFFs (~0.655) — the same statistic the
phase feedback consumes, computed through the same code path.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")

from sspmnet import denoise, TrainConfig, load_quadpol_phase
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps, _local_coherence
from sspmnet.metrics import epi_metric, ssim_metric

OUT = "results/ri_compare"
rng = np.random.default_rng(7)

clean = np.load(os.path.join(OUT, "denoised_baseline.npy")).astype(np.float64)
xpol = 0.5 * (clean[1] + clean[2])
clean[1] = xpol
clean[2] = xpol

# ── target statistic from the REAL phase files ──
pha_real = load_quadpol_phase("data/tiff")
u = np.exp(2j * pha_real.astype(np.float64))
target_coh = float(_local_coherence(u[1] * np.conj(u[2]), 7).mean())
print(f"real HV-VH doubled-angle coherence (win=7): {target_coh:.3f}")


def cn(shape, sigma):
    return rng.normal(0, sigma / np.sqrt(2), shape) + \
        1j * rng.normal(0, sigma / np.sqrt(2), shape)


def simulate(sigma_n):
    g = np.stack([cn(clean[0].shape, 1.0) for _ in range(4)])
    g[2] = g[1]                                   # reciprocity: shared speckle
    n = np.stack([cn(clean[0].shape, sigma_n) for _ in range(4)])
    return clean * g + n


def sim_coh(sigma_n):
    z = simulate(sigma_n)
    uu = np.exp(2j * np.angle(z))
    return float(_local_coherence(uu[1] * np.conj(uu[2]), 7).mean())


# ── calibrate sigma_n (bisection on the scale of the cross-pol RMS) ──
rms_x = float(np.sqrt((clean[1] ** 2).mean()))
lo, hi = 0.05 * rms_x, 3.0 * rms_x
for _ in range(12):
    mid = 0.5 * (lo + hi)
    c = sim_coh(mid)
    if c > target_coh:
        lo = mid
    else:
        hi = mid
sigma_n = 0.5 * (lo + hi)
print(f"calibrated sigma_n = {sigma_n:.3f} ({sigma_n/rms_x:.2f} x xpol RMS), "
      f"sim coherence = {sim_coh(sigma_n):.3f}")

rng = np.random.default_rng(7)                     # fixed noise for all runs
z = simulate(sigma_n)
re, im = z.real, z.imag
amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(re).astype(np.float32),
                      np.abs(im).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)                # same code path as real

W = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
         guide_cv_protect=0.3)
variants = {
    "winner":        (TrainConfig(**W), None),
    "PH-b1.5-f0.5":  (TrainConfig(**W, phase_smooth_boost=1.5, phase_fidelity=0.5), maps_sim),
    "PH-b3-f0.5":    (TrainConfig(**W, phase_smooth_boost=3.0, phase_fidelity=0.5), maps_sim),
    "PH-b1.5-f0.8":  (TrainConfig(**W, phase_smooth_boost=1.5, phase_fidelity=0.8), maps_sim),
    "PH-b1.5-f0":    (TrainConfig(**W, phase_smooth_boost=1.5, phase_fidelity=0.0), maps_sim),
    "PH-b0-f0.5":    (TrainConfig(**W, phase_smooth_boost=0.0, phase_fidelity=0.5), maps_sim),
}

results = {"Noisy": amp_sim.astype(np.float64)}
for name, (cfg, pha_a) in variants.items():
    print(f"\n=== synthetic {name} ===", flush=True)
    res = denoise(amp_sim, cfg, ri_pair=ri_sim, pha=pha_a)
    results[name] = res["denoised"].astype(np.float64)
np.save(os.path.join(OUT, "synth_phase_clean.npy"), clean.astype(np.float32))
np.save(os.path.join(OUT, "synth_phase_noisy.npy"), amp_sim)
for name, d in results.items():
    if name != "Noisy":
        np.save(os.path.join(OUT, f"synth_{name.replace('.','p').replace('-','_')}.npy"), d)


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
cols = ["PSNR(HH)", "PSNR(HV)", "SSIM*(HH)", "SSIM*(HV)", "EPI*(HH)", "EPI*(HV)"]
hdr = "  {:<14}".format("Method") + "".join(f"{c:>12}" for c in cols)
print(hdr)
lines = [hdr]
for name, d in results.items():
    ds = scale_match(d, clean)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[0], ds[0]), ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1])]
    line = "  {:<14}".format(name) + "".join(f"{v:>12.4f}" for v in vals)
    print(line)
    lines.append(line)
with open(os.path.join(OUT, "metrics_synth_phase.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("done")
