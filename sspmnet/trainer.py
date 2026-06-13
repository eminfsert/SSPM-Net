"""
Zero-shot training / inference loop for SSPM-Net.

A single SAR image is denoised by training the network from scratch on that
image alone (no clean references, no pre-training). The pipeline mirrors the
configuration used for the thesis results:

    pre-warmup (bilateral target)   -> quench random-init chaos
    SSPM masking + losses           -> the self-supervised objective
    speckle factorization + histogram matching (-> Rayleigh)
    non-local self-similarity loss
    EMA (Polyak averaging) of weights
    cosine LR schedule
    fixed-budget stop: the final output is the single EMA checkpoint at the
        last iteration (no metric-based / per-group checkpoint selection)
    test-time augmentation (D4 x MC-dropout) at final inference

``denoise()`` is the single entry point.
"""
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .model import SSPMNet
from .masking import QuadPolSpatialMasker
from .losses import (
    MaskedL1Loss, adaptive_tv_loss, polarization_consistency_loss, bound_loss,
    polarimetric_nl_loss, compute_reference_histogram, compute_soft_histogram,
    warmup_target_4ch,
)


@dataclass
class TrainConfig:
    """Hyper-parameters for the zero-shot training loop (thesis defaults)."""

    iters: int = 700
    lr: float = 1e-4
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    init_seed: int = 42

    # Masking
    mask_keep_prob: float = 0.7

    # Total-variation (edge-aware) schedule
    tv_mult: float = 10.0
    lambda_tv_start: float = 0.4
    lambda_tv_end: float = 0.03

    # Other loss weights
    lambda_pol: float = 0.1         # HV ~ VH reciprocity
    bound_lambda: float = 2.0       # keep output in [0, 1]
    nlm_lambda: float = 0.5         # non-local self-similarity
    nlm_window: int = 7
    nlm_sigma: float = 0.1

    # Pre-warmup
    pre_warmup: int = 50

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.99

    # Speckle factorization + histogram matching
    use_speckle_factor: bool = True
    s_init: float = 1.0
    s_lr_mult: float = 5.0
    lambda_fact: float = 1.0
    lambda_mask_fact: float = 1.0
    hist_lambda: float = 1.0
    hist_recip_weight: float = 0.5
    hist_bins: int = 64
    hist_range: float = 3.0
    looks_ref: int = 1

    # Final inference
    use_tta: bool = True
    tta_mc_passes: int = 4
    n_inference: int = 32           # used when use_tta is False

    # Reporting
    snapshot_every: int = 100


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def denoise(amp_4ch_raw, cfg: TrainConfig = None, on_snapshot=None, verbose=True):
    """Zero-shot denoise one quad-pol amplitude image.

    Parameters
    ----------
    amp_4ch_raw : np.ndarray, shape (4, H, W)
        Quad-pol amplitude, channel order [HH, HV, VH, VV]. Any positive
        scale (the loop normalizes internally by the per-channel 99th
        percentile and restores the scale on output).
    cfg : TrainConfig
        Training hyper-parameters (defaults reproduce the thesis pipeline).
    on_snapshot : callable or None
        Optional callback ``f(step, denoised_np, noisy_np, info)`` invoked at
        every ``snapshot_every`` step (for visualization).
    verbose : bool
        Print per-snapshot progress.

    Returns
    -------
    dict with keys:
        'denoised'  : np.ndarray (4, H, W) — denoised amplitude (input scale)
        'stop_step' : int   — iteration the output was taken at (= cfg.iters)
        'loss_hist' : list  — total loss per iteration
    """
    if cfg is None:
        cfg = TrainConfig()
    device = _resolve_device(cfg.device)

    if cfg.init_seed is not None:
        torch.manual_seed(cfg.init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.init_seed)

    amp_4ch_raw = np.asarray(amp_4ch_raw, dtype=np.float32)
    H_d, W_d = amp_4ch_raw.shape[1], amp_4ch_raw.shape[2]
    q99 = np.quantile(amp_4ch_raw, 0.99, axis=(1, 2), keepdims=True)
    amp_norm = np.clip(amp_4ch_raw / np.maximum(q99, 1e-9), 0.0, 5.0)
    noisy_t = torch.from_numpy(amp_norm).unsqueeze(0).to(device)

    model = SSPMNet(Config()).to(device)
    masker = QuadPolSpatialMasker(keep_prob=cfg.mask_keep_prob).to(device)
    crit = MaskedL1Loss()
    n_iters = cfg.iters

    # ── Optimizer (+ optional speckle-factor tensor with its own LR) ──
    if cfg.use_speckle_factor:
        S_real = torch.full((1, 4, H_d, W_d), cfg.s_init,
                            dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.AdamW([
            {"params": model.parameters(), "lr": cfg.lr},
            {"params": [S_real], "lr": cfg.lr * cfg.s_lr_mult},
        ], weight_decay=1e-5)
    else:
        S_real = None
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iters, eta_min=cfg.lr / 100.0)

    # ── Pre-warmup: pull the random init toward a bilateral-smoothed target ──
    if cfg.pre_warmup > 0:
        warmup_t = torch.from_numpy(
            warmup_target_4ch(amp_norm)).float().unsqueeze(0).to(device)
        opt_pw = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        for _ in range(cfg.pre_warmup):
            model.train()
            loss_pw = ((model(noisy_t) - warmup_t) ** 2).mean()
            opt_pw.zero_grad()
            loss_pw.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pw.step()
        if verbose:
            print(f"  [pre-warmup] {cfg.pre_warmup} steps, final loss={loss_pw.item():.5f}")

    # ── EMA copies ──
    if cfg.use_ema:
        model_ema = copy.deepcopy(model)
        for p in model_ema.parameters():
            p.requires_grad = False
        S_real_ema = S_real.detach().clone() if S_real is not None else None
    else:
        model_ema = None
        S_real_ema = None

    # ── Reference (Rayleigh) histogram ──
    if cfg.hist_lambda > 0:
        h_ref_t, bin_centers_t, hist_step = compute_reference_histogram(
            looks=cfg.looks_ref, n_bins=cfg.hist_bins,
            range_max=cfg.hist_range, device=device)

    loss_hist = []

    if verbose:
        print(f"  [train] iters={n_iters} lr={cfg.lr} tv_mult={cfg.tv_mult} "
              f"speckle_factor={cfg.use_speckle_factor} hist={cfg.hist_lambda} "
              f"nl={cfg.nlm_lambda} ema={cfg.use_ema} tta={cfg.use_tta}")

    for step in range(n_iters):
        model.train()
        masker.train()
        ltv = math.cos(math.pi * step / max(n_iters, 1))
        lambda_tv = cfg.lambda_tv_end + 0.5 * (cfg.lambda_tv_start - cfg.lambda_tv_end) * (1.0 + ltv)

        m = masker(noisy_t)
        d = model(m["masked_input"])

        # Co-pol blind-spot (HH, VV)
        l_hh = crit(d[:, 0:1], noisy_t[:, 0:1], m["mask_hh"])
        l_vv = crit(d[:, 3:4], noisy_t[:, 3:4], m["mask_vv"])
        loss_copol = (l_hh + l_vv) / 2

        # Cross-pol Noise2Noise (HV <-> VH via the synchronized mask)
        mxp = m["mask_xpol"]
        l_hv = crit(d[:, 1:2], noisy_t[:, 2:3], mxp)
        l_vh = crit(d[:, 2:3], noisy_t[:, 1:2], mxp)
        loss_xpol = (l_hv + l_vh) / 2

        # Regularization
        l_tv = adaptive_tv_loss(d, noisy_t)
        l_pol = polarization_consistency_loss(d)
        l_bound = bound_loss(d)
        l_nl = (polarimetric_nl_loss(d, noisy_t, cfg.nlm_window, cfg.nlm_sigma)
                if cfg.nlm_lambda > 0 else torch.tensor(0.0, device=device))

        # Speckle factorization + histogram matching
        if cfg.use_speckle_factor and S_real is not None:
            S_pos = torch.nn.functional.softplus(S_real)
            l_fact = ((d * S_pos - noisy_t) ** 2).mean()
            if cfg.hist_lambda > 0:
                histos, marg = [], 0.0
                for c in range(4):
                    s_c = torch.clamp(S_pos[:, c], 0.0, cfg.hist_range)
                    h_c = compute_soft_histogram(s_c, bin_centers_t, hist_step)
                    marg = marg + ((h_c - h_ref_t) ** 2).sum()
                    histos.append(h_c)
                l_hist = marg / 4.0 + cfg.hist_recip_weight * ((histos[1] - histos[2]) ** 2).sum()
            else:
                l_hist = torch.tensor(0.0, device=device)
            eff_mask_w = cfg.lambda_mask_fact
        else:
            l_fact = torch.tensor(0.0, device=device)
            l_hist = torch.tensor(0.0, device=device)
            eff_mask_w = 1.0

        eff_tv = lambda_tv * cfg.tv_mult
        loss = (eff_mask_w * (loss_copol + loss_xpol)
                + eff_tv * l_tv + cfg.lambda_pol * l_pol
                + cfg.bound_lambda * l_bound
                + cfg.nlm_lambda * l_nl
                + cfg.hist_lambda * l_hist
                + cfg.lambda_fact * l_fact)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_hist.append(float(loss.item()))

        # EMA update
        if cfg.use_ema and model_ema is not None:
            with torch.no_grad():
                for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                    p_ema.data.mul_(cfg.ema_decay).add_(p.data, alpha=1.0 - cfg.ema_decay)
                if S_real is not None and S_real_ema is not None:
                    S_real_ema.mul_(cfg.ema_decay).add_(S_real.detach(), alpha=1.0 - cfg.ema_decay)

        # ── Periodic snapshot (visualization only; no metric-based selection) ──
        if (step + 1) % cfg.snapshot_every == 0 or step == 0:
            inf_model = model_ema if (cfg.use_ema and model_ema is not None) else model
            with torch.no_grad():
                inf_model.train()                 # MC-dropout active
                acc = torch.zeros_like(noisy_t)
                for _ in range(8):
                    acc += inf_model(noisy_t).clamp(0, 1)
                acc /= 8
            d_np = acc[0].cpu().numpy()

            info = {"step": step + 1, "iters": n_iters, "loss": float(loss.item())}
            if verbose:
                print(f"  step {step+1:>4d}/{n_iters} loss={loss.item():.4f}")
            if on_snapshot is not None:
                on_snapshot(step + 1, d_np, noisy_t[0].cpu().numpy(), info)

    # ── Final inference: single (EMA) checkpoint at the last step + TTA ──
    final_model = model_ema if (cfg.use_ema and model_ema is not None) else model

    def _final_infer():
        final_model.train()                       # MC-dropout active
        with torch.no_grad():
            if cfg.use_tta:
                acc_f = torch.zeros_like(noisy_t)
                cnt = 0
                for k_rot in range(4):
                    for do_flip in (False, True):
                        x_aug = torch.rot90(noisy_t, k_rot, dims=[-2, -1])
                        if do_flip:
                            x_aug = torch.flip(x_aug, dims=[-1])
                        for _ in range(cfg.tta_mc_passes):
                            out = final_model(x_aug).clamp(0, 1)
                            if do_flip:
                                out = torch.flip(out, dims=[-1])
                            out = torch.rot90(out, -k_rot, dims=[-2, -1])
                            acc_f += out
                            cnt += 1
                return acc_f / cnt
            acc_f = torch.zeros_like(noisy_t)
            for _ in range(cfg.n_inference):
                acc_f += final_model(noisy_t).clamp(0, 1)
            return acc_f / cfg.n_inference

    acc = _final_infer()                          # all 4 channels, same model
    if verbose:
        src = "EMA" if (cfg.use_ema and model_ema is not None) else "raw"
        print(f"  [final] {src} weights @ step {n_iters}"
              + (" + D4xMC-dropout TTA" if cfg.use_tta else ""))

    denoised = acc[0].cpu().numpy() * q99.squeeze()[:, None, None]

    del model, masker, opt, noisy_t, acc
    if model_ema is not None:
        del model_ema
    if S_real is not None:
        del S_real
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "denoised": denoised.astype(np.float32),
        "stop_step": int(n_iters),
        "loss_hist": loss_hist,
    }
