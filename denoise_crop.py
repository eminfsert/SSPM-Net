"""
Denoise overlapping quad-pol amplitude patches with SSPM-Net and stitch each
channel back into its full image.

Input layout (one folder per channel, identical patch grid):
    {prefix}_{hh,hv,vh,vv}_amp_4x4/patches/*_patch_<row>_<col>_<size>_<ch>_amp.tiff
    {prefix}_{hh,hv,vh,vv}_amp_4x4/full/*.tiff        (optional, for comparison)

Resumable: every patch result (4, H, W) is written atomically to
OUT/patches/*.npy as soon as it is finished; patches whose .npy already exists
are skipped on restart.

Usage:
    python denoise_crop.py                       # denoise missing patches, then merge
    python denoise_crop.py --merge-only          # only stitch what is already saved
    python denoise_crop.py --out /content/drive/MyDrive/sspm_results
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import tifffile
from skimage import io

CHANNELS = ("hh", "hv", "vh", "vv")               # SSPM-Net order
PATCH_RE = re.compile(r"patch_(\d+)_(\d+)_(\d+)_")


def parse_patch(path):
    r, c, s = map(int, PATCH_RE.search(os.path.basename(path)).groups())
    return r, c, s


def patch_key(path):
    """Channel-independent id, e.g. '2_patch_3072_4096_512'."""
    name = os.path.basename(path)
    return name[:PATCH_RE.search(name).end() - 1]


def atomic_save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def ramp_1d(size, overlap, taper_start, taper_end):
    """Raised-cosine ramps over the overlap; no taper on the image border side."""
    w = np.ones(size, dtype=np.float64)
    if overlap > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)
        if taper_start:
            w[:overlap] = ramp
        if taper_end:
            w[-overlap:] = ramp[::-1]
    return w


def collect_patches(prefix):
    """-> sorted list of (key, (r, c, s), {ch: path})."""
    per_ch = {}
    for ch in CHANNELS:
        files = glob.glob(os.path.join(f"{prefix}_{ch}_amp_4x4", "patches", "*.tif*"))
        if not files:
            raise SystemExit(f"no patches for channel {ch} under {prefix}_{ch}_amp_4x4/patches")
        per_ch[ch] = {patch_key(f): f for f in files}
    keys = set(per_ch["hh"])
    for ch in CHANNELS:
        if set(per_ch[ch]) != keys:
            raise SystemExit(f"patch grid of {ch} differs from hh")
    return sorted(((k, parse_patch(per_ch["hh"][k]), {ch: per_ch[ch][k] for ch in CHANNELS})
                   for k in keys), key=lambda t: t[1])


def denoise_patches(patches, out_dir, iters, device):
    import torch
    from sspmnet import denoise, TrainConfig

    pdir = os.path.join(out_dir, "patches")
    os.makedirs(pdir, exist_ok=True)
    n = len(patches)
    for i, (key, _, files) in enumerate(patches, 1):
        dst = os.path.join(pdir, key + "_sspm.npy")
        if os.path.exists(dst):
            print(f"[{i}/{n}] skip (exists): {key}", flush=True)
            continue

        amp = np.stack([io.imread(files[ch]).astype(np.float32) for ch in CHANNELS])
        t0 = time.time()
        print(f"[{i}/{n}] denoising {key} ...", flush=True)
        res = denoise(amp, TrainConfig(iters=iters, device=device), verbose=False)
        den = res["denoised"].astype(np.float32)   # (4, H, W) [HH, HV, VH, VV]

        np.save(os.path.join(pdir, key + "_loss.npy"),
                np.asarray(res["loss_hist"], dtype=np.float32))
        atomic_save_npy(dst, den)                   # written last: marks patch done
        print(f"    saved -> {dst}  ({time.time() - t0:.0f}s)", flush=True)
        if device != "cpu":
            torch.cuda.empty_cache()


def merge(patches, out_dir, prefix):
    pdir = os.path.join(out_dir, "patches")
    geo = [g for _, g, _ in patches]
    r0 = min(g[0] for g in geo)
    c0 = min(g[1] for g in geo)
    size = geo[0][2]
    rows = sorted({g[0] for g in geo})
    stride = rows[1] - rows[0] if len(rows) > 1 else size
    H = max(g[0] for g in geo) - r0 + size
    W = max(g[1] for g in geo) - c0 + size
    overlap = size - stride

    acc = np.zeros((len(CHANNELS), H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    missing = []
    for key, (r, c, s), _ in patches:
        p = os.path.join(pdir, key + "_sspm.npy")
        if not os.path.exists(p):
            missing.append(key)
            continue
        y, x = r - r0, c - c0
        w = np.outer(ramp_1d(s, overlap, y > 0, y + s < H),
                     ramp_1d(s, overlap, x > 0, x + s < W))
        acc[:, y:y + s, x:x + s] += np.load(p) * w
        wsum[y:y + s, x:x + s] += w

    if missing:
        print(f"[merge] {len(missing)} patch(es) not denoised yet; covered area only.")
    full = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), 0).astype(np.float32)

    tag = os.path.basename(prefix)
    np.save(os.path.join(out_dir, f"{tag}_quadpol_sspm_r{r0}_c{c0}.npy"), full)
    for k, ch in enumerate(CHANNELS):
        base = os.path.join(out_dir, f"{tag}_{ch}_amp_4x4_sspm_r{r0}_c{c0}")
        tifffile.imwrite(base + ".tiff", full[k])                     # float32
        u8 = np.clip(np.rint(full[k]), 0, 255).astype(np.uint8)
        tifffile.imwrite(base + "_uint8.tiff", u8)
        io.imsave(base + ".png", u8, check_contrast=False)

        refs = sorted(glob.glob(os.path.join(f"{prefix}_{ch}_amp_4x4", "full", "*.tif*")))
        if refs:
            side = np.concatenate([io.imread(refs[0]), u8], axis=1)
            io.imsave(os.path.join(out_dir, f"{tag}_{ch}_noisy_vs_sspm.png"), side,
                      check_contrast=False)

    with open(os.path.join(out_dir, "merge_info.json"), "w") as fh:
        json.dump({"channels": list(CHANNELS), "origin_rc": [r0, c0], "shape": [H, W],
                   "patch": size, "stride": stride, "n_patches": len(patches),
                   "missing": missing}, fh, indent=2)
    print(f"[merge] saved {len(CHANNELS)} channels to {out_dir}  shape={full.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="crop_2",
                    help="channel folders are <prefix>_<ch>_amp_4x4")
    ap.add_argument("--out", default="results/crop_2_quadpol_4x4")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    patches = collect_patches(args.prefix)
    os.makedirs(args.out, exist_ok=True)
    if not args.merge_only:
        denoise_patches(patches, args.out, args.iters, args.device)
    merge(patches, args.out, args.prefix)


if __name__ == "__main__":
    main()
