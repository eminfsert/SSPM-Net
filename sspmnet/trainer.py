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
    warmup_target_4ch, guide_edge_weights, modulate_edge_weights, _box_blur,
    nl_polish, ratio_whiteness_loss, edge_fidelity_loss,
)
from .phase_data import phase_feedback_maps
from .complex_data import _L1_RATIO


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

    # Complex (real/imaginary) auxiliary supervision — active only when
    # ``denoise(..., ri_pair=...)`` is given (see sspmnet.complex_data)
    ri_weight: float = 0.6          # share of the masked loss on RI targets
    guide_tv: bool = True           # multi-look guided TV edge weights
    guide_alpha: float = 3.0        # guided-TV edge sensitivity
    guide_nlm: bool = True          # multi-look reference for the NL loss
    ri_mode: str = "targets"        # "targets": RI as extra masked targets;
                                    # "merlin": MERLIN-style input separation
                                    # — train on ONE complex component,
                                    # supervise with the OTHER (full-pixel,
                                    # no masking; inference averages both)
    merlin_recip_weight: float = 0.5  # cross-pol share of the reciprocal
                                    # channel's opposite-component target
    merlin_loss: str = "l1"         # "l1" (median convention) or "nll":
                                    # Gaussian negative log-likelihood of the
                                    # target component given the predicted
                                    # amplitude (MERLIN's original loss;
                                    # unbiased — d^2 converges to the true
                                    # reflectivity, no calibration constant)
    guide_cv_protect: float = 0.0   # Lee-style heterogeneity gate threshold
                                    # on the guide's local CV (0 disables;
                                    # ~0.3 for the 8-look span)

    # Phase feedback — active only when ``denoise(..., pha=...)`` is given
    # (see sspmnet.phase_data). The HV-VH reciprocity coherence gives a
    # per-pixel "is this value noise?" map: reciprocity makes the two
    # cross-pol channels share the same complex speckle, so their phase
    # DISAGREES only where additive thermal/system noise dominates.
    phase_win: int = 7              # circular-statistics window
    phase_smooth_boost: float = 1.5 # TV/NLM boost where noise-dominated:
                                    # x (1 + b * (1 - snr)); 0 disables
    phase_surface_boost: float = 0.0  # extra TV/NLM boost from the HH-VV
                                    # co-pol (surface-scattering) coherence
    phase_protect: float = 0.0      # TV/NLM protection from the spatial
                                    # phase-coherence (deterministic
                                    # scatterer) map: x (1 - p * det)
    phase_fidelity: float = 0.5     # down-weight of the CROSS-POL data
                                    # term where noise-dominated (the N2N
                                    # target there is thermal noise, whose
                                    # positive median biases dark pixels
                                    # up): x (1 - f * (1 - snr)), mean-1
                                    # renormalized; 0 disables

    # Residual-speckle refinement
    nl_self_refresh: int = 0        # every N steps, rebuild the non-local
                                    # reference from the EMA model's own
                                    # output (0 = keep the static guide) —
                                    # a cleaner reference gives far more
                                    # accurate similar-patch weights
    nl_self_mix: float = 0.7        # share of the EMA output in the
                                    # refreshed reference (the rest stays
                                    # on the initial guide, avoiding
                                    # self-confirmation drift)
    nlm_lambda_end: float = 0.0     # >0: ramp nlm_lambda linearly to this
                                    # value over training (pull harder as
                                    # the reference gets cleaner)
    nlm_sigma_noise: float = 0.0    # >0 with pha: per-pixel NLM sigma
                                    # sigma*(1 + k*(1-snr)) — noise-
                                    # dominated pixels accept more
                                    # neighbors and average harder
    whiteness_lambda: float = 0.0   # spatial-whiteness penalty on the
                                    # ratio image (see losses.py)
    whiteness_lags: tuple = (1, 2, 3)  # autocorrelation lags to penalize —
                                    # keep ABOVE the speckle correlation
                                    # length: (1,2,3) for white simulated
                                    # speckle, (3,4,5) for the oversampled
                                    # real data (lag-1 autocorr ~0.5 there)
    edge_sharp_lambda: float = 0.0  # [NEGATIVE RESULT — keep 0] gradient-
                                    # matching edge loss: every guide's own
                                    # gradients carry speckle, so matching
                                    # them re-injects noise (ENL collapses)
                                    # and can crush dark channels; kept for
                                    # the record
    edge_boost: float = 0.0         # edge-masked unsharp of the FINAL
                                    # output: d + k*M*(d - blur(d)), mask M
                                    # from the multi-look span + phase snr
                                    # gradients — restores the edge
                                    # steepness that TV/NLM/TTA soften
                                    # (GT-validated: raises PSNR/SSIM/EPI
                                    # at ~no ENL cost; ~1.0 recommended)
    edge_boost_dark: float = 0.2    # dark-suppression of the boost mask:
                                    # x mu_ch/(mu_ch+t) — without it dark
                                    # cross-pol pixels get amplified
                                    # residue (real ENLr(HV) collapses)
    edge_phase_weight: float = 0.3  # share of the phase snr-coherence
                                    # gradient fused into the edge weights
                                    # (edge evidence independent of
                                    # amplitude speckle; needs pha)
    polish: float = 0.0             # final-stage non-local refinement
                                    # strength in [0, 1] (0 = off); edge/
                                    # point pixels are protected by a CV
                                    # gate (+ the phase 'det' map)
    polish_window: int = 9
    polish_sigma: float = 0.1

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

    # Model architecture overrides: kwargs forwarded to sspmnet.Config
    # (e.g. {"dropout_style": "pixel", "norm": "group"}); None = defaults
    model_cfg: dict = None

    # Reporting
    snapshot_every: int = 100


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def denoise(amp_4ch_raw, cfg: TrainConfig = None, on_snapshot=None, verbose=True,
            ri_pair=None, pha=None, sat=None):
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
    ri_pair : np.ndarray, shape (2, 4, H, W), optional
        Calibrated |Re| / |Im| pseudo-amplitude pair on the amplitude scale
        (from ``sspmnet.complex_data.load_quadpol_tiffs``). When given, the
        masked losses additionally use these two independent Noise2Noise
        targets, and the TV / non-local regularizers are steered by a
        multi-look guide built from them (see ``TrainConfig.ri_weight``,
        ``guide_tv``, ``guide_nlm``).
    pha : np.ndarray (4, H, W), or dict, optional
        Folded quad-pol phase in [0, pi] (from
        ``sspmnet.phase_data.load_quadpol_phase``), or a precomputed map
        dict from ``phase_feedback_maps``. Enables the per-pixel phase
        feedback: regularization is boosted where the HV-VH reciprocity
        coherence says the observation is noise-dominated, and the
        cross-pol data term is down-weighted there (see the ``phase_*``
        fields of ``TrainConfig``).
    sat : np.ndarray (4, H, W) bool, optional
        Per-channel saturation mask (True = the uint8 source was clipped at
        255; from ``load_quadpol_tiffs(..., return_sat=True)``). Saturated
        pixels are excluded from the MERLIN L1 data term — their targets
        carry a truncated bright tail. Regularizers still cover them.

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

    sat_keep = None
    if sat is not None:
        sat_keep = torch.from_numpy(
            (~np.asarray(sat, dtype=bool)).astype(np.float32)
        ).unsqueeze(0).to(device)                  # (1, 4, H, W): 1 = usable

    # ── Complex (RI) auxiliaries: independent pseudo-amplitude targets,
    #    multi-look edge guide for TV, multi-look reference for the NL loss ──
    ar_t = ai_t = tv_weights = nl_ref = None
    if ri_pair is not None:
        ri_pair = np.asarray(ri_pair, dtype=np.float32)
        ri_norm = np.clip(ri_pair / np.maximum(q99[None], 1e-9), 0.0, 5.0)
        ri_t = torch.from_numpy(ri_norm).to(device)          # (2, 4, H, W)
        ar_t = ri_t[0:1]                                     # (1, 4, H, W)
        ai_t = ri_t[1:2]
        with torch.no_grad():
            if cfg.guide_tv:
                # ~8-look span guide (2 RI looks x 4 channels)
                guide = (ri_t ** 2).mean(dim=(0, 1)).sqrt()[None, None]
                tv_weights = guide_edge_weights(
                    guide, alpha=cfg.guide_alpha,
                    cv_protect=cfg.guide_cv_protect or None)
            if cfg.guide_nlm:
                # per-channel 2-look reference, lightly smoothed
                nl_ref = _box_blur(0.5 * (ar_t + ai_t), k=3, passes=1)
    merlin = ri_pair is not None and cfg.ri_mode == "merlin"

    # ── Phase feedback maps: per-pixel noise/structure evidence from the
    #    folded quad-pol phase (see sspmnet.phase_data) ──
    phase_factor = fid_w = nl_w = det_t = nl_sigma_map = None
    if pha is not None:
        pm = pha if isinstance(pha, dict) else phase_feedback_maps(
            pha=np.asarray(pha, dtype=np.float32), win=cfg.phase_win)
        snr_t = torch.from_numpy(pm["snr"]).float()[None, None].to(device)
        noise_map = 1.0 - snr_t                       # 1 = noise-dominated
        det_t = torch.from_numpy(pm["det"]).float()[None, None].to(device)
        boost = 1.0 + cfg.phase_smooth_boost * noise_map
        if cfg.phase_surface_boost > 0:
            surf_t = torch.from_numpy(pm["surface"]).float()[None, None].to(device)
            boost = boost + cfg.phase_surface_boost * surf_t
        if cfg.phase_protect > 0:
            boost = boost * (1.0 - cfg.phase_protect * det_t)
        phase_factor = boost                          # (1, 1, H, W)
        nl_w = phase_factor / phase_factor.mean()     # mean-1 for L_nl
        if cfg.phase_fidelity > 0:
            fw = 1.0 - cfg.phase_fidelity * noise_map
            fid_w = fw / fw.mean()                    # mean-1 data weight
        if cfg.nlm_sigma_noise > 0:
            nl_sigma_map = cfg.nlm_sigma * (1.0 + cfg.nlm_sigma_noise * noise_map)
        if tv_weights is None:
            # no multi-look guide available: build edge weights from the
            # (1-look) span so the phase factor has something to modulate
            span_g = (noisy_t ** 2).mean(dim=1, keepdim=True).sqrt()
            tv_weights = guide_edge_weights(
                span_g, alpha=cfg.guide_alpha,
                cv_protect=cfg.guide_cv_protect or None)
        tv_weights = modulate_edge_weights(tv_weights, phase_factor)

    nl_ref0 = nl_ref            # initial (static) non-local reference

    # ── Edge-sharpness fidelity: precompute the guide's log gradients and
    #    the edge weights (amplitude multi-look + optional phase snr) ──
    guide_log = edge_w = edge_full = None
    if cfg.edge_sharp_lambda > 0 or cfg.edge_boost > 0:
        with torch.no_grad():
            if ri_pair is not None:
                g_span = (ri_t ** 2).mean(dim=(0, 1)).sqrt()[None, None]
                # per-channel gradient TARGET: each channel's own 2-look mean
                g_ch = _box_blur(0.5 * (ar_t + ai_t), k=3, passes=2)
            else:
                g_span = (noisy_t ** 2).mean(dim=1, keepdim=True).sqrt()
                g_ch = _box_blur(noisy_t, k=3, passes=2)
            guide_log = 0.5 * torch.log(g_ch ** 2 + 1e-4)
            # edge LOCATIONS from the reliable multi-look span
            span_log = 0.5 * torch.log(
                _box_blur(g_span, k=3, passes=2) ** 2 + 1e-6)
            e_h = (span_log[:, :, 1:, :] - span_log[:, :, :-1, :]).abs()
            e_w = (span_log[:, :, :, 1:] - span_log[:, :, :, :-1]).abs()
            if pha is not None and cfg.edge_phase_weight > 0:
                snr_s = _box_blur(snr_t, k=3, passes=1)
                p_h = (snr_s[:, :, 1:, :] - snr_s[:, :, :-1, :]).abs()
                p_w = (snr_s[:, :, :, 1:] - snr_s[:, :, :, :-1]).abs()
                e_h = e_h / (e_h.mean() + 1e-8) \
                    + cfg.edge_phase_weight * p_h / (p_h.mean() + 1e-8)
                e_w = e_w / (e_w.mean() + 1e-8) \
                    + cfg.edge_phase_weight * p_w / (p_w.mean() + 1e-8)
            norm = 0.5 * (e_h.mean() + e_w.mean()) + 1e-8
            e_h, e_w = e_h / norm, e_w / norm
            # rational soft mask: ~0 on speckle-level gradients, ->1 on edges
            edge_w = (e_h ** 2 / (1.0 + e_h ** 2), e_w ** 2 / (1.0 + e_w ** 2))
            # full-pixel edge map (for polish protection)
            ef = torch.zeros_like(g_span)
            ef[:, :, 1:, :] = torch.maximum(ef[:, :, 1:, :], edge_w[0])
            ef[:, :, :-1, :] = torch.maximum(ef[:, :, :-1, :], edge_w[0])
            ef[:, :, :, 1:] = torch.maximum(ef[:, :, :, 1:], edge_w[1])
            ef[:, :, :, :-1] = torch.maximum(ef[:, :, :, :-1], edge_w[1])
            edge_full = ef

    model = SSPMNet(Config(**(cfg.model_cfg or {}))).to(device)
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
        # With RI data the bilateral runs on the 2-look mean (less speckle);
        # its small scale offset vs. amplitude is corrected by main training.
        warm_src = ri_norm.mean(axis=0) if ri_pair is not None else amp_norm
        warmup_t = torch.from_numpy(
            warmup_target_4ch(warm_src)).float().unsqueeze(0).to(device)
        opt_pw = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        for i_pw in range(cfg.pre_warmup):
            model.train()
            # In MERLIN mode the network sees component inputs, so warm up
            # on them too (alternating), not on the amplitude.
            x_pw = ri_t[i_pw % 2:i_pw % 2 + 1] if merlin else noisy_t
            loss_pw = ((model(x_pw) - warmup_t) ** 2).mean()
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

    # MERLIN mode: the network is trained on component inputs, so inference
    # averages the predictions from BOTH components (as in MERLIN's test time)
    infer_inputs = [ar_t, ai_t] if merlin else [noisy_t]

    if verbose:
        ri_msg = (f" RI(mode={cfg.ri_mode},w={cfg.ri_weight},"
                  f"guide_tv={cfg.guide_tv},"
                  f"guide_nlm={cfg.guide_nlm})" if ri_pair is not None else "")
        if pha is not None:
            ri_msg += (f" PH(b={cfg.phase_smooth_boost},"
                       f"s={cfg.phase_surface_boost},p={cfg.phase_protect},"
                       f"f={cfg.phase_fidelity})")
        if cfg.nl_self_refresh > 0 or cfg.polish > 0 or cfg.whiteness_lambda > 0:
            if cfg.edge_sharp_lambda > 0 or cfg.edge_boost > 0:
                ri_msg += (f" EDGE(l={cfg.edge_sharp_lambda},"
                           f"boost={cfg.edge_boost},pw={cfg.edge_phase_weight})")
            ri_msg += (f" RF(selfref={cfg.nl_self_refresh},"
                       f"lam_end={cfg.nlm_lambda_end},"
                       f"white={cfg.whiteness_lambda},polish={cfg.polish})")
        print(f"  [train] iters={n_iters} lr={cfg.lr} tv_mult={cfg.tv_mult} "
              f"speckle_factor={cfg.use_speckle_factor} hist={cfg.hist_lambda} "
              f"nl={cfg.nlm_lambda} ema={cfg.use_ema} tta={cfg.use_tta}{ri_msg}")

    for step in range(n_iters):
        model.train()
        masker.train()
        ltv = math.cos(math.pi * step / max(n_iters, 1))
        lambda_tv = cfg.lambda_tv_end + 0.5 * (cfg.lambda_tv_start - cfg.lambda_tv_end) * (1.0 + ltv)

        if merlin:
            # ── MERLIN-style input separation: the input is ONE complex
            #    component's pseudo-amplitude, the target is the OTHER, so
            #    the input carries none of the target's noise and EVERY
            #    pixel supervises (no masking needed). ──
            k = step % 2
            x_in, tgt = ri_t[k:k + 1], ri_t[1 - k:2 - k]
            d = model(x_in)

            if cfg.merlin_loss == "nll":
                # Gaussian NLL of the raw component c ~ N(0, A^2/2) given
                # the predicted amplitude A = d (MERLIN's original loss;
                # E-unbiased: d^2 -> true reflectivity). Targets arrive on
                # the L1 convention (x _L1_RATIO); undo it to recover the
                # raw component scale.
                c = tgt / _L1_RATIO
                v = d ** 2 + 1e-3
                nll = 0.5 * torch.log(v) + (c ** 2) / v

                def pair_loss(ch, rec_ch):
                    w_r = cfg.merlin_recip_weight
                    if rec_ch is None:
                        return nll[:, ch:ch + 1].mean()
                    c_r = tgt[:, rec_ch:rec_ch + 1] / _L1_RATIO
                    v_c = v[:, ch:ch + 1]
                    nll_r = 0.5 * torch.log(v_c) + (c_r ** 2) / v_c
                    if fid_w is not None:            # phase feedback
                        return ((1 - w_r) * (nll[:, ch:ch + 1] * fid_w).mean()
                                + w_r * (nll_r * fid_w).mean())
                    return ((1 - w_r) * nll[:, ch:ch + 1].mean()
                            + w_r * nll_r.mean())

                loss_copol = (pair_loss(0, None) + pair_loss(3, None)) / 2
                loss_xpol = (pair_loss(1, 2) + pair_loss(2, 1)) / 2
            else:
                # Weighted mean over usable pixels: excludes saturated
                # TARGET pixels (sat_keep of the target channel) and, on
                # cross-pol, applies the mean-1 phase fidelity weight.
                def _wmean(t, tgt_ch, use_fid=False):
                    w = None
                    if sat_keep is not None:
                        w = sat_keep[:, tgt_ch:tgt_ch + 1]
                    if use_fid and fid_w is not None:
                        w = fid_w if w is None else w * fid_w
                    if w is None:
                        return t.mean()
                    return (t * w).sum() / w.sum().clamp(min=1.0)

                # Co-pol: full-pixel L1 N2N vs. the opposite component
                l_hh = _wmean((d[:, 0:1] - tgt[:, 0:1]).abs(), 0)
                l_vv = _wmean((d[:, 3:4] - tgt[:, 3:4]).abs(), 3)
                loss_copol = (l_hh + l_vv) / 2

                # Cross-pol: opposite component of own channel +
                # (reciprocity) of the reciprocal channel — both
                # independent of the input. With phase feedback, pixels
                # whose observation is noise-dominated (low HV-VH phase
                # coherence) contribute less to the data term.
                w_r = cfg.merlin_recip_weight
                l_hv = ((1 - w_r) * _wmean((d[:, 1:2] - tgt[:, 1:2]).abs(), 1, True)
                        + w_r * _wmean((d[:, 1:2] - tgt[:, 2:3]).abs(), 2, True))
                l_vh = ((1 - w_r) * _wmean((d[:, 2:3] - tgt[:, 2:3]).abs(), 2, True)
                        + w_r * _wmean((d[:, 2:3] - tgt[:, 1:2]).abs(), 1, True))
                loss_xpol = (l_hv + l_vh) / 2
        else:
            m = masker(noisy_t)
            d = model(m["masked_input"])

            def masked_loss(pred, tgt_ch, mask):
                """Masked L1 vs. the amplitude target of channel ``tgt_ch``;
                with RI data, mixed with the two independent |Re|/|Im|
                pseudo-amplitude targets of the same channel."""
                l_amp = crit(pred, noisy_t[:, tgt_ch:tgt_ch + 1], mask)
                if ar_t is None or cfg.ri_weight <= 0:
                    return l_amp
                l_ri = 0.5 * (crit(pred, ar_t[:, tgt_ch:tgt_ch + 1], mask)
                              + crit(pred, ai_t[:, tgt_ch:tgt_ch + 1], mask))
                return (1.0 - cfg.ri_weight) * l_amp + cfg.ri_weight * l_ri

            # Co-pol blind-spot (HH, VV)
            l_hh = masked_loss(d[:, 0:1], 0, m["mask_hh"])
            l_vv = masked_loss(d[:, 3:4], 3, m["mask_vv"])
            loss_copol = (l_hh + l_vv) / 2

            # Cross-pol Noise2Noise (HV <-> VH via the synchronized mask)
            mxp = m["mask_xpol"]
            l_hv = masked_loss(d[:, 1:2], 2, mxp)
            l_vh = masked_loss(d[:, 2:3], 1, mxp)
            loss_xpol = (l_hv + l_vh) / 2

        # Regularization
        l_tv = adaptive_tv_loss(d, noisy_t, weights=tv_weights)
        l_pol = polarization_consistency_loss(d)
        l_bound = bound_loss(d)
        nlm_lam = cfg.nlm_lambda
        if cfg.nlm_lambda_end > 0 and n_iters > 1:
            nlm_lam = (cfg.nlm_lambda + (cfg.nlm_lambda_end - cfg.nlm_lambda)
                       * step / (n_iters - 1))
        l_nl = (polarimetric_nl_loss(d, nl_ref if nl_ref is not None else noisy_t,
                                     cfg.nlm_window, cfg.nlm_sigma,
                                     pixel_weight=nl_w, sigma_map=nl_sigma_map)
                if cfg.nlm_lambda > 0 else torch.tensor(0.0, device=device))
        l_white = (ratio_whiteness_loss(d, noisy_t, lags=cfg.whiteness_lags)
                   if cfg.whiteness_lambda > 0
                   else torch.tensor(0.0, device=device))
        l_edge = (edge_fidelity_loss(d, guide_log, edge_w)
                  if cfg.edge_sharp_lambda > 0
                  else torch.tensor(0.0, device=device))

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
                + nlm_lam * l_nl
                + cfg.whiteness_lambda * l_white
                + cfg.edge_sharp_lambda * l_edge
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

        # ── Self-referential non-local reference refresh ──
        if (cfg.nl_self_refresh > 0 and cfg.nlm_lambda > 0
                and (step + 1) % cfg.nl_self_refresh == 0):
            ref_model = model_ema if (cfg.use_ema and model_ema is not None) else model
            was_training = ref_model.training
            ref_model.eval()                       # deterministic (no dropout)
            with torch.no_grad():
                pred = torch.zeros_like(noisy_t)
                for x_base in infer_inputs:
                    pred += ref_model(x_base).clamp(0, 1)
                pred /= len(infer_inputs)
            if was_training:
                ref_model.train()
            base = nl_ref0 if nl_ref0 is not None else noisy_t
            nl_ref = cfg.nl_self_mix * pred + (1.0 - cfg.nl_self_mix) * base

        # ── Periodic snapshot (visualization only; no metric-based selection) ──
        if (step + 1) % cfg.snapshot_every == 0 or step == 0:
            inf_model = model_ema if (cfg.use_ema and model_ema is not None) else model
            with torch.no_grad():
                inf_model.train()                 # MC-dropout active
                acc = torch.zeros_like(noisy_t)
                n_pass = 8 // len(infer_inputs)
                for x_base in infer_inputs:
                    for _ in range(n_pass):
                        acc += inf_model(x_base).clamp(0, 1)
                acc /= n_pass * len(infer_inputs)
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
            acc_f = torch.zeros_like(noisy_t)
            cnt = 0
            if cfg.use_tta:
                for x_base in infer_inputs:
                    for k_rot in range(4):
                        for do_flip in (False, True):
                            x_aug = torch.rot90(x_base, k_rot, dims=[-2, -1])
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
            for x_base in infer_inputs:
                for _ in range(max(cfg.n_inference // len(infer_inputs), 1)):
                    acc_f += final_model(x_base).clamp(0, 1)
                    cnt += 1
            return acc_f / cnt

    acc = _final_infer()                          # all 4 channels, same model

    # ── Final-stage non-local polish (edge-/point-protected) ──
    if cfg.polish > 0:
        with torch.no_grad():
            if ri_pair is not None:
                g_amp = (ri_t ** 2).mean(dim=(0, 1)).sqrt()[None, None]
            else:
                g_amp = (noisy_t ** 2).mean(dim=1, keepdim=True).sqrt()
            mu_g = _box_blur(g_amp, k=9, passes=1)
            m2_g = _box_blur(g_amp ** 2, k=9, passes=1)
            cv_g = torch.sqrt((m2_g - mu_g ** 2).clamp(min=0)) / (mu_g + 1e-6)
            thr = cfg.guide_cv_protect if cfg.guide_cv_protect > 0 else 0.3
            prot = torch.sigmoid((cv_g - thr) / (0.25 * thr))
            if det_t is not None:
                prot = torch.maximum(prot, det_t)
            if edge_full is not None:
                prot = torch.maximum(prot, edge_full)
            acc = nl_polish(acc, window=cfg.polish_window,
                            sigma=cfg.polish_sigma, strength=cfg.polish,
                            protect=prot).clamp(0, 1)

    # ── Edge-masked unsharp: restore edge steepness on the final output ──
    if cfg.edge_boost > 0 and edge_full is not None:
        with torch.no_grad():
            m_edge = _box_blur(edge_full, k=3, passes=1)   # widen the mask
            if cfg.edge_boost_dark > 0:                    # per-channel dark gate
                mu_ch = _box_blur(acc, k=9, passes=1)
                m_edge = m_edge * (mu_ch / (mu_ch + cfg.edge_boost_dark))
            acc = (acc + cfg.edge_boost * m_edge
                   * (acc - _box_blur(acc, k=3, passes=1))).clamp(0, 1)

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
