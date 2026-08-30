"""Missing GT leg: the amplitude-only BASELINE on the same simulated noisy
patch, so baseline vs final can be compared with TRUE PSNR/SSIM/EPI."""
import os, sys
import numpy as np
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig
from sspmnet.metrics import epi_metric, ssim_metric, find_top_k_rois, enl_roi_multi

OUT="results/ri_compare"
clean = np.load(f"{OUT}/denoised_baseline.npy").astype(np.float64)
xpol=0.5*(clean[1]+clean[2]); clean[1]=xpol; clean[2]=xpol
rng = np.random.default_rng(7)
def cn(sh,s): return rng.normal(0,s/np.sqrt(2),sh)+1j*rng.normal(0,s/np.sqrt(2),sh)
g=np.stack([cn(clean[0].shape,1.0) for _ in range(4)]); g[2]=g[1]
n=np.stack([cn(clean[0].shape,11.161) for _ in range(4)])
z=clean*g+n; amp_sim=np.abs(z).astype(np.float32)

print("=== synthetic baseline (amplitude-only) ===", flush=True)
res = denoise(amp_sim, TrainConfig(iters=700))
d = res["denoised"].astype(np.float64)
np.save(f"{OUT}/polish_synth_baseline.npy", d)

def sm(x,ref):
    o=x.copy()
    for c in range(4): o[c]=x[c]*float((x[c]*ref[c]).sum()/max((x[c]**2).sum(),1e-9))
    return o
def psnr(x,ref): return 10*np.log10(ref.max()**2/max(((x-ref)**2).mean(),1e-12))
rois,rs=find_top_k_rois(amp_sim[1].astype(np.float64))
rows={"baseline":d, "final":np.load(f"{OUT}/polish_synth_CpB05.npy").astype(np.float64)}
lines=[]
for nm,dd in rows.items():
    ds=sm(dd,clean)
    line=(f"{nm:<10} PSNR(HH)={psnr(ds[0],clean[0]):.4f} PSNR(HV)={psnr(ds[1],clean[1]):.4f} "
          f"SSIM(HH)={ssim_metric(clean[0],ds[0]):.4f} SSIM(HV)={ssim_metric(clean[1],ds[1]):.4f} "
          f"EPI(HH)={epi_metric(clean[0],ds[0]):.4f} EPI(HV)={epi_metric(clean[1],ds[1]):.4f} "
          f"ENL(HH)={enl_roi_multi(dd[0],rois,rs):.1f} ENL(HV)={enl_roi_multi(dd[1],rois,rs):.1f}")
    print(line); lines.append(line)
open(f"{OUT}/metrics_baseline_vs_final_synth.txt","w").write("\n".join(lines)+"\n")
