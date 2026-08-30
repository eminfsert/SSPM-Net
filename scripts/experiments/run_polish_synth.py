"""GT validation of the residual-speckle refinements on top of PH-b3-f0.5:

    A = self-referential NLM (refresh=100, mix=0.7, lambda 0.5->1.5,
        phase-adaptive sigma)
    B = final non-local polish (post-hoc here — mathematically identical to
        polish= in TrainConfig, applied on the normalized scale)
    C = ratio-whiteness loss (0.05; simulated speckle is white, so this is
        its fair test bed)

Same simulation as run_phase_synth2.py (sigma_n=11.161, seed 7); the base
PH-b3 is RE-RUN on this exact noise so every row shares one realization.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps
from sspmnet.losses import nl_polish, _box_blur
from sspmnet.metrics import epi_metric, ssim_metric, find_top_k_rois, enl_roi_multi

OUT = "results/ri_compare"
clean = np.load(os.path.join(OUT, "denoised_baseline.npy")).astype(np.float64)
xpol = 0.5*(clean[1]+clean[2]); clean[1]=xpol; clean[2]=xpol
rng = np.random.default_rng(7)
def cn(shape, s):
    return rng.normal(0, s/np.sqrt(2), shape) + 1j*rng.normal(0, s/np.sqrt(2), shape)
sigma_n = 11.161
g = np.stack([cn(clean[0].shape, 1.0) for _ in range(4)]); g[2]=g[1]
n = np.stack([cn(clean[0].shape, sigma_n) for _ in range(4)])
z = clean*g + n
amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(z.real).astype(np.float32),
                      np.abs(z.imag).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)

PH = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
          guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5)
A = dict(nl_self_refresh=100, nl_self_mix=0.7, nlm_lambda_end=1.5,
         nlm_sigma_noise=1.0)
train_variants = {
    "base(PH-b3)": TrainConfig(**PH),
    "+A(selfNLM)": TrainConfig(**PH, **A),
    "+C(white)":   TrainConfig(**PH, whiteness_lambda=0.05),
}

results = {}
for name, cfg in train_variants.items():
    print(f"\n=== synthetic {name} ===", flush=True)
    res = denoise(amp_sim, cfg, ri_pair=ri_sim, pha=maps_sim)
    results[name] = res["denoised"].astype(np.float64)
    np.save(os.path.join(OUT, f"polish_synth_{name.strip('+').split('(')[0]}.npy"),
            res["denoised"])

# ── post-hoc polish (B): identical to TrainConfig(polish=...), on the
#    normalized scale, protect = span-CV gate max phase 'det' map ──
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q99 = np.quantile(amp_sim, 0.99, axis=(1, 2), keepdims=True)
ri_t = torch.from_numpy(np.clip(ri_sim/np.maximum(q99[None],1e-9),0,5)).float().to(dev)
g_amp = (ri_t ** 2).mean(dim=(0, 1)).sqrt()[None, None]
mu_g = _box_blur(g_amp, 9, 1); m2_g = _box_blur(g_amp**2, 9, 1)
cv_g = torch.sqrt((m2_g-mu_g**2).clamp(min=0))/(mu_g+1e-6)
prot = torch.sigmoid((cv_g-0.3)/(0.25*0.3))
det_t = torch.from_numpy(maps_sim["det"]).float()[None,None].to(dev)
prot = torch.maximum(prot, det_t)

def posthoc_polish(d_np, strength):
    dn = torch.from_numpy(
        np.clip(d_np/np.maximum(q99,1e-9),0,1)).float()[None].to(dev)
    out = nl_polish(dn, window=9, sigma=0.1, strength=strength,
                    protect=prot).clamp(0,1)
    return out[0].cpu().numpy().astype(np.float64) * q99

for src in ["base(PH-b3)", "+A(selfNLM)"]:
    for s_ in (0.3, 0.5):
        results[f"{src}+B{s_}"] = posthoc_polish(results[src], s_)

def sm(x, ref):
    out = x.copy()
    for c in range(4):
        k = float((x[c]*ref[c]).sum()/max((x[c]**2).sum(),1e-9)); out[c]=x[c]*k
    return out
def psnr(x, ref):
    return 10*np.log10(ref.max()**2/max(((x-ref)**2).mean(),1e-12))

rois, rs = find_top_k_rois(amp_sim[1].astype(np.float64))
cols = ["PSNR(HH)","PSNR(HV)","SSIM(HH)","SSIM(HV)","EPI(HH)","EPI(HV)",
        "ENLroi(HV)","ENLroi(HH)"]
hdr = "  {:<20}".format("Method") + "".join(f"{c:>11}" for c in cols)
print("\nGround-truth metrics (vs KNOWN clean; scale-matched) + ENL")
print(hdr); lines=[hdr]
for name, d in results.items():
    ds = sm(d, clean)
    vals = [psnr(ds[0],clean[0]), psnr(ds[1],clean[1]),
            ssim_metric(clean[0],ds[0]), ssim_metric(clean[1],ds[1]),
            epi_metric(clean[0],ds[0]), epi_metric(clean[1],ds[1]),
            enl_roi_multi(d[1],rois,rs), enl_roi_multi(d[0],rois,rs)]
    line = "  {:<20}".format(name) + "".join(f"{v:>11.4f}" for v in vals)
    print(line); lines.append(line)
with open(os.path.join(OUT,"metrics_polish_synth.txt"),"w") as f:
    f.write("\n".join(lines)+"\n")
for name in ["base(PH-b3)+B0.3","base(PH-b3)+B0.5","+A(selfNLM)+B0.3","+A(selfNLM)+B0.5"]:
    safe = name.replace("(","_").replace(")","_").replace("+","p").replace(".","")
    np.save(os.path.join(OUT,f"polish_synth_{safe}.npy"), results[name])
print("done")
