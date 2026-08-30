"""GT sweep of the edge-sharpness fidelity loss on top of the full combo
(PH-b3 + whiteness + polish 0.5, polish now edge-protected)."""
import os, sys
import numpy as np
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig
from sspmnet.complex_data import calibrate_ri
from sspmnet.phase_data import phase_feedback_maps
from sspmnet.metrics import epi_metric, ssim_metric, find_top_k_rois, enl_roi_multi

OUT="results/ri_compare"
clean=np.load(f"{OUT}/denoised_baseline.npy").astype(np.float64)
xpol=0.5*(clean[1]+clean[2]); clean[1]=xpol; clean[2]=xpol
rng=np.random.default_rng(7)
def cn(sh,s): return rng.normal(0,s/np.sqrt(2),sh)+1j*rng.normal(0,s/np.sqrt(2),sh)
g=np.stack([cn(clean[0].shape,1.0) for _ in range(4)]); g[2]=g[1]
n=np.stack([cn(clean[0].shape,11.161) for _ in range(4)])
z=clean*g+n; amp_sim=np.abs(z).astype(np.float32)
ri_sim=calibrate_ri(amp_sim,np.abs(z.real).astype(np.float32),np.abs(z.imag).astype(np.float32))
maps=phase_feedback_maps(z=z)

COMBO=dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
           guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
           whiteness_lambda=0.05, polish=0.5)
def sm(x,ref):
    o=x.copy()
    for c in range(4): o[c]=x[c]*float((x[c]*ref[c]).sum()/max((x[c]**2).sum(),1e-9))
    return o
def psnr(x,ref): return 10*np.log10(ref.max()**2/max(((x-ref)**2).mean(),1e-12))
rois,rs=find_top_k_rois(amp_sim[1].astype(np.float64))
lines=[]
for lam in (0.5, 1.0, 2.0):
    print(f"\n=== edge lambda={lam} ===", flush=True)
    res=denoise(amp_sim, TrainConfig(**COMBO, edge_sharp_lambda=lam),
                ri_pair=ri_sim, pha=maps)
    d=res["denoised"].astype(np.float64)
    np.save(f"{OUT}/polish_synth_edge{str(lam).replace('.','p')}.npy", d)
    ds=sm(d,clean)
    line=(f"edge{lam}: PSNR={psnr(ds[0],clean[0]):.4f}/{psnr(ds[1],clean[1]):.4f} "
          f"SSIM={ssim_metric(clean[0],ds[0]):.4f}/{ssim_metric(clean[1],ds[1]):.4f} "
          f"EPI={epi_metric(clean[0],ds[0]):.4f}/{epi_metric(clean[1],ds[1]):.4f} "
          f"ENL={enl_roi_multi(d[0],rois,rs):.1f}/{enl_roi_multi(d[1],rois,rs):.1f}")
    print(line); lines.append(line)
open(f"{OUT}/metrics_edge_synth.txt","w").write("\n".join(lines)+"\n")
print("done  (reference final: PSNR=23.6383/21.3888 SSIM=0.7815/0.7943 EPI=0.8273/0.7911 ENL=141.3/162.6)")
