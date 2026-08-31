"""A4: does excluding uint8-saturated pixels from the MERLIN data term help,
and does feeding the full-dynamic-range .npy amplitude (instead of the
clipped uint8 amp TIFF) help?

Synthetic leg: the GT protocol is extended with the REAL degradation — the
simulated components are quantized to uint8 with per-channel clip fractions
matched to the real TIFFs, so for the first time the synthetic test includes
the saturation the real pipeline actually suffers.  Variants:

    noclip-final : unclipped substrate (ceiling; cached from arch ablation)
    clip-final   : clipped+quantized substrate, no countermeasures
    clip-sat     : + saturation mask on the MERLIN data term
    clip-satfull : + full-range float amplitude (regularizers/hist/fact see
                   the true bright tail; the data term stays RI)

Real leg: final (cached) vs sat vs npy-amp+sat.
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
amp, ri, sat_real = load_quadpol_tiffs("data/tiff", return_sat=True)
amp_npy = np.load("data/example_quadpol.npy").astype(np.float32)
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


# ── synthetic substrate (identical construction to run_arch_ablation) ──
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
amp_s = np.abs(z).astype(np.float32)

# ── uint8 clipping matched to the REAL per-channel saturation fractions ──
def u8_clip(x, frac_sat):
    """Scale so `frac_sat` of pixels land at >=255, quantize to uint8."""
    thr = np.quantile(x, 1.0 - frac_sat)
    return np.clip(np.round(x * (255.0 / max(thr, 1e-9))), 0, 255).astype(np.float32)

import glob
def real_sat_frac(comp):
    import tifffile
    fs = sorted(glob.glob(f"data/tiff/*_{comp}.tiff"))
    return [float((tifffile.imread(f) == 255).mean()) for f in fs]  # hh,hv,vh,vv order by name

frac_amp = real_sat_frac("amp"); frac_re = real_sat_frac("real"); frac_im = real_sat_frac("imgy")
print("real sat fractions amp/re/im:",
      [f"{v:.3f}" for v in frac_amp], [f"{v:.3f}" for v in frac_re],
      [f"{v:.3f}" for v in frac_im], flush=True)
# NOTE: glob order is hh, hv, vh, vv — matches channel order.

amp_c = np.stack([u8_clip(amp_s[c], frac_amp[c]) for c in range(4)])
re_c = np.stack([u8_clip(np.abs(z.real[c]), frac_re[c]) for c in range(4)]).astype(np.float32)
im_c = np.stack([u8_clip(np.abs(z.imag[c]), frac_im[c]) for c in range(4)]).astype(np.float32)
sat_c = (amp_c >= 255.0) | (re_c >= 255.0) | (im_c >= 255.0)
ri_c = calibrate_ri(amp_c, re_c, im_c, mode="l1")
maps_s = phase_feedback_maps(z=z)
# full-range float amp on the clipped substrate, rescaled to amp_c's units
s_full = float(np.median(amp_c[amp_c < 250] / np.maximum(amp_s[amp_c < 250], 1e-9)))
amp_full = amp_s * s_full

STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, polish=0.5, edge_boost=1.0)

runs_s = {}
runs_s["noclip-final"] = cached("arch_synth_final.npy", lambda: denoise(
    amp_s, TrainConfig(**STACK, whiteness_lags=(1, 2, 3)),
    ri_pair=calibrate_ri(amp_s, np.abs(z.real).astype(np.float32),
                         np.abs(z.imag).astype(np.float32), mode="l1"),
    pha=maps_s)["denoised"]).astype(np.float64)
for name, kw in [("clip-final",   dict()),
                 ("clip-sat",     dict(sat=sat_c)),
                 ("clip-satfull", dict(sat=sat_c, full=True))]:
    a_in = amp_full if kw.pop("full", False) else amp_c
    sat_in = kw.get("sat")
    print(f"\n=== synth {name} ===", flush=True)
    runs_s[name] = cached(f"a4_synth_{name.replace('-','_')}.npy",
        lambda: denoise(a_in, TrainConfig(**STACK, whiteness_lags=(1, 2, 3)),
                        ri_pair=ri_c, pha=maps_s, sat=sat_in)["denoised"]
        ).astype(np.float64)

runs_r = {}
f_final = f"{OUT}/arch_real_final.npy"
assert os.path.exists(f_final), "run run_arch_ablation.py first"
runs_r["final"] = np.load(f_final).astype(np.float64)
for name, a_in, sat_in in [("sat", amp, sat_real),
                           ("npy+sat", amp_npy, sat_real)]:
    print(f"\n=== real {name} ===", flush=True)
    # RI stays TIFF-derived; calibrate it onto the chosen amplitude's scale:
    ri_in = ri
    if a_in is amp_npy:
        import tifffile
        re_abs = np.abs(np.stack([tifffile.imread(f) for f in
                 sorted(glob.glob("data/tiff/*_real.tiff"))])).astype(np.float32)
        im_abs = np.abs(np.stack([tifffile.imread(f) for f in
                 sorted(glob.glob("data/tiff/*_imgy.tiff"))])).astype(np.float32)
        ri_in = calibrate_ri(a_in, re_abs, im_abs, mode="l1")
    runs_r[name] = cached(f"a4_real_{name.replace('+','_')}.npy",
        lambda: denoise(a_in, TrainConfig(**STACK, whiteness_lags=(3, 4, 5)),
                        ri_pair=ri_in, pha=maps_real, sat=sat_in)["denoised"]
        ).astype(np.float64)

# ── tables ──
def scale_match(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        s = float((x[c]*ref[c]).sum() / max((x[c]**2).sum(), 1e-9))
        out[c] = x[c]*s
    return out
def psnr(x, ref):
    return 10*np.log10(ref.max()**2 / max(((x-ref)**2).mean(), 1e-12))

lines = []
rois_s, rs_s = find_top_k_rois(amp_s[1].astype(np.float64))
# bright-tail-focused error: RMSE on the top-1% clean pixels (where clipping bites)
thr99 = np.quantile(clean, 0.99)
bright = clean >= thr99
hdr = ("  {:<14}".format("Method") + "".join(f"{c:>11}" for c in
       ["PSNR(HH)", "PSNR(HV)", "SSIM(HH)", "EPI(HH)", "EPI(HV)",
        "ENL(HH)", "ENL(HV)", "RMSEbright"]))
print("\nSynthetic-GT with uint8 clipping (vs known clean; scale-matched)")
print(hdr); lines += ["Synthetic-GT with uint8 clipping", hdr]
for name, d in runs_s.items():
    ds = scale_match(d, clean)
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[0], ds[0]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1]),
            enl_roi_multi(d[0], rois_s, rs_s), enl_roi_multi(d[1], rois_s, rs_s),
            float(np.sqrt(((ds - clean)**2)[bright].mean()))]
    line = "  {:<14}".format(name) + "".join(f"{v:>11.4f}" for v in vals)
    print(line); lines.append(line)

rois_r, rs_r = find_top_k_rois(amp[1])
def ratio_enl(d, ref, c):
    eps = 1e-3
    rI = (ref[c].astype(np.float64)**2 + eps) / (d[c]**2 + eps)
    v = rI[(d[c] > 2) & (ref[c] > 0)]
    return (v.mean()/v.std())**2
hdr = ("  {:<14}".format("Method") + "".join(f"{c:>12}" for c in
       ["corr(HV,VH)", "ENL-ROI(HH)", "ENL-ROI(HV)", "EPI(HH)", "EPI(HV)",
        "ENLr(HH)", "ENLr(HV)"]))
print("\nReal patch (each vs its own noisy reference)")
print(hdr); lines += ["", "Real patch", hdr]
for name, d in runs_r.items():
    ref = amp_npy if name.startswith("npy") else amp
    rec = reciprocity_metrics(d[1], d[2])
    # ENL-ROI locations chosen once on the tiff amp (same spatial ROIs)
    vals = [rec["corr"], enl_roi_multi(d[0], rois_r, rs_r),
            enl_roi_multi(d[1], rois_r, rs_r),
            epi_metric(ref[0].astype(np.float64), d[0]),
            epi_metric(ref[1].astype(np.float64), d[1]),
            ratio_enl(d, ref, 0), ratio_enl(d, ref, 1)]
    line = "  {:<14}".format(name) + "".join(f"{v:>12.4f}" for v in vals)
    print(line); lines.append(line)

with open(f"{OUT}/metrics_a4_sat.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nWrote", f"{OUT}/metrics_a4_sat.txt")
