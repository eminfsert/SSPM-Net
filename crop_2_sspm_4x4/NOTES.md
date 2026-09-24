# crop_2 quad-pol SSPM-Net denoise — progress notes

**Goal:** denoise the 16 overlapping 512px patches of crop_2 (4x4 grid, stride 256,
full image 1280x1280, origin r3072 c4096) with SSPM-Net (HH, HV, VH, VV together),
then stitch each channel back into one full image.

**How:** `python denoise_crop.py --out crop_2_sspm_4x4` (iters=1000, cuda).
- Per patch: `crop_2_sspm_4x4/patches/<key>_sspm.npy` (4,512,512 [HH,HV,VH,VV]) + `<key>_loss.npy`
- Each finished patch is committed & pushed to this branch individually.
- Resumable: existing `_sspm.npy` patches are skipped on restart.
- Merge (raised-cosine blending over overlaps) runs after the last patch;
  `python denoise_crop.py --out crop_2_sspm_4x4 --merge-only` redoes only the stitching.

**Why a new output folder:** the first attempt wrote to `results/`, which is in
`.gitignore`, so nothing was committed and the results were lost when the session crashed.

**Resume after a crash:**
1. `git clone -b crop2-quadpol-denoise https://github.com/eminfsert/SSPM-Net.git && cd SSPM-Net`
2. `gh auth login -h github.com -p https -w && gh auth setup-git`
3. `python denoise_crop.py --out crop_2_sspm_4x4` — only the missing patches get processed
4. Commit & push the new files in `crop_2_sspm_4x4/`.
