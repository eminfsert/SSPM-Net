"""Round 2: how far can the phase-feedback smoothing boost go?
Same simulation as run_phase_synth.py (identical seed/noise), b in {5, 8}."""
import os, sys
import numpy as np
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_phase
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps, _local_coherence
from sspmnet.metrics import epi_metric, ssim_metric, find_top_k_rois, enl_roi_multi

OUT = "results/ri_compare"
clean = np.load(os.path.join(OUT, "denoised_baseline.npy")).astype(np.float64)
xpol = 0.5*(clean[1]+clean[2]); clean[1]=xpol; clean[2]=xpol
rng = np.random.default_rng(7)
def cn(shape, sigma):
    return rng.normal(0, sigma/np.sqrt(2), shape) + 1j*rng.normal(0, sigma/np.sqrt(2), shape)
sigma_n = 11.161                      # calibrated in round 1
def simulate(s):
    g = np.stack([cn(clean[0].shape, 1.0) for _ in range(4)]); g[2]=g[1]
    n = np.stack([cn(clean[0].shape, s) for _ in range(4)])
    return clean*g + n
# consume the calibration draws NOT — round 1 reset rng before the final z:
z = simulate(sigma_n)
amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(z.real).astype(np.float32),
                      np.abs(z.imag).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)
# verify identical noise to round 1
ref = np.load(os.path.join(OUT, "synth_phase_noisy.npy"))
print("noisy identical to round 1:", np.allclose(amp_sim, ref))

W = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
         guide_cv_protect=0.3)
variants = {
    "PH-b5-f0.5": TrainConfig(**W, phase_smooth_boost=5.0, phase_fidelity=0.5),
    "PH-b8-f0.5": TrainConfig(**W, phase_smooth_boost=8.0, phase_fidelity=0.5),
}
def sm(x, ref):
    out = x.copy()
    for c in range(x.shape[0]):
        s = float((x[c]*ref[c]).sum()/max((x[c]**2).sum(),1e-9)); out[c]=x[c]*s
    return out
def psnr(x, ref):
    return 10*np.log10(ref.max()**2/max(((x-ref)**2).mean(),1e-12))
rois, rs = find_top_k_rois(amp_sim[1].astype(np.float64))
for name, cfg in variants.items():
    print(f"\n=== synthetic {name} ===", flush=True)
    res = denoise(amp_sim, cfg, ri_pair=ri_sim, pha=maps_sim)
    d = res["denoised"].astype(np.float64)
    np.save(os.path.join(OUT, f"synth_{name.replace('.','p').replace('-','_')}.npy"), d)
    ds = sm(d, clean)
    print(f"{name}: PSNR(HH)={psnr(ds[0],clean[0]):.4f} PSNR(HV)={psnr(ds[1],clean[1]):.4f} "
          f"SSIM(HH)={ssim_metric(clean[0],ds[0]):.4f} SSIM(HV)={ssim_metric(clean[1],ds[1]):.4f} "
          f"EPI(HH)={epi_metric(clean[0],ds[0]):.4f} EPI(HV)={epi_metric(clean[1],ds[1]):.4f} "
          f"ENLroi(HV)={enl_roi_multi(d[1],rois,rs):.2f} ENLroi(HH)={enl_roi_multi(d[0],rois,rs):.2f}")
print("done")
