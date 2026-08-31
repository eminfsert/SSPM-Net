"""Track B sanity: the scene trainer in its DEGENERATE case (1 patch,
batch=4 x 256 px crops = same pixel throughput as one 512 px full-image
step) must roughly reproduce the single-patch pixel+group results on both
protocols. Validates the crop/batch machinery before the real 16-patch
scene data arrives."""
import os, sys
import numpy as np

sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import (denoise_scene, TrainConfig, load_quadpol_tiffs,
                     load_quadpol_phase)
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps, _local_coherence
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
amp, ri = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")

# synthetic substrate (identical construction to run_arch_ablation)
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
rng = np.random.default_rng(7)
z = simulate(0.5*(lo+hi))
amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(z.real).astype(np.float32),
                      np.abs(z.imag).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)

BASE = {"dropout_style": "pixel", "norm": "group"}
STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, polish=0.5, edge_boost=1.0)


def cached(fname, fn):
    f = os.path.join(OUT, fname)
    if os.path.exists(f):
        print(f"[cached] {fname}", flush=True)
        return np.load(f)
    out = fn()
    np.save(f, out)
    return out


print("\n=== scene-sanity synth (1 patch, crop batch) ===", flush=True)
d_s = cached("scene1_synth_pg.npy", lambda: denoise_scene(
    [{"amp": amp_sim, "ri": ri_sim, "pha": maps_sim, "sat": None}],
    TrainConfig(**STACK, whiteness_lags=(1, 2, 3), model_cfg=BASE),
    crop=256, batch=4)["denoised"][0]).astype(np.float64)
print("\n=== scene-sanity real (1 patch, crop batch) ===", flush=True)
d_r = cached("scene1_real_pg.npy", lambda: denoise_scene(
    [{"amp": amp, "ri": ri, "pha": pha, "sat": None}],
    TrainConfig(**STACK, whiteness_lags=(3, 4, 5), model_cfg=BASE),
    crop=256, batch=4)["denoised"][0]).astype(np.float64)

ref_s = np.load(f"{OUT}/arch_synth_pixel_group.npy").astype(np.float64)
ref_r = np.load(f"{OUT}/arch_real_pixel_group.npy").astype(np.float64)

def scale_match(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        s = float((x[c]*ref[c]).sum() / max((x[c]**2).sum(), 1e-9))
        out[c] = x[c]*s
    return out
def psnr(x, ref):
    return 10*np.log10(ref.max()**2 / max(((x-ref)**2).mean(), 1e-12))

lines = ["Scene-trainer degenerate sanity (1 patch, batch=4x256 crops) vs "
         "single-patch trainer, both pixel+group full stack:", ""]
rois_s, rs_s = find_top_k_rois(amp_sim[1].astype(np.float64))
for name, d in [("single-patch", ref_s), ("scene-1patch", d_s)]:
    ds = scale_match(d, clean)
    lines.append(f"  synth {name:<13} PSNR {psnr(ds[0],clean[0]):.3f}/"
                 f"{psnr(ds[1],clean[1]):.3f}  SSIM {ssim_metric(clean[0],ds[0]):.4f}"
                 f"/{ssim_metric(clean[1],ds[1]):.4f}  EPI {epi_metric(clean[0],ds[0]):.4f}"
                 f"/{epi_metric(clean[1],ds[1]):.4f}  ENL {enl_roi_multi(d[0],rois_s,rs_s):.0f}"
                 f"/{enl_roi_multi(d[1],rois_s,rs_s):.0f}")
rois_r, rs_r = find_top_k_rois(amp[1])
def ratio_enl(d, c):
    eps = 1e-3
    rI = (amp[c].astype(np.float64)**2 + eps) / (d[c]**2 + eps)
    v = rI[(d[c] > 2) & (amp[c] > 0)]
    return (v.mean()/v.std())**2
for name, d in [("single-patch", ref_r), ("scene-1patch", d_r)]:
    rec = reciprocity_metrics(d[1], d[2])
    lines.append(f"  real  {name:<13} corr {rec['corr']:.4f}  ENL-ROI "
                 f"{enl_roi_multi(d[0],rois_r,rs_r):.3f}/{enl_roi_multi(d[1],rois_r,rs_r):.3f}"
                 f"  EPI {epi_metric(amp[0],d[0]):.4f}/{epi_metric(amp[1],d[1]):.4f}"
                 f"  ENLr {ratio_enl(d,0):.4f}/{ratio_enl(d,1):.4f}")
print("\n" + "\n".join(lines))
with open(f"{OUT}/metrics_scene_sanity.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
