"""
Scene-scale zero-shot training: one model, many patches of the SAME scene.

Rationale: the single-patch trainer learns the internal statistics of one
512x512 image. When the full radar scene is available as multiple patches,
training the SAME zero-shot objective on random crops drawn from the whole
patch pool multiplies the internal training data without breaking the
zero-shot claim — no external data, no clean references; the training set
IS the image being denoised (ZSSR-style internal learning).

Semantics follow ``sspmnet.trainer.denoise`` exactly (same losses, same
defaults, same MERLIN / phase-feedback / refinement machinery), with the
data term evaluated on a batch of random crops per step instead of one
full image. Per-pixel state (speckle factor, TV guide weights, phase maps,
saturation masks) is precomputed full-size per patch and cropped in sync.

Normalization uses ONE scene-level per-channel q99 (not per patch), so all
patches live on a consistent radiometric scale.

Fair-comparison note: with ``batch=4, crop=256`` each step sees the same
number of pixels as one full 512x512 step of the single-patch trainer.
"""
import copy
import math

import numpy as np
import torch

from .config import Config
from .model import SSPMNet
from .masking import QuadPolSpatialMasker
from .losses import (
    MaskedL1Loss, adaptive_tv_loss, polarization_consistency_loss,
    bound_loss, polarimetric_nl_loss, guide_edge_weights,
    modulate_edge_weights, ratio_whiteness_loss, nl_polish,
    compute_reference_histogram, compute_soft_histogram, warmup_target_4ch,
    _box_blur,
)
from .phase_data import phase_feedback_maps
from .trainer import TrainConfig, _resolve_device


def _prep_patch(entry, q99, cfg, device):
    """Precompute every full-size per-patch tensor the loop needs."""
    d = {}
    amp = np.asarray(entry["amp"], dtype=np.float32)
    amp_norm = np.clip(amp / np.maximum(q99, 1e-9), 0.0, 5.0)
    d["noisy"] = torch.from_numpy(amp_norm).unsqueeze(0).to(device)
    d["amp_norm"] = amp_norm
    d["H"], d["W"] = amp.shape[1], amp.shape[2]

    d["sat_keep"] = None
    if entry.get("sat") is not None:
        d["sat_keep"] = torch.from_numpy(
            (~np.asarray(entry["sat"], dtype=bool)).astype(np.float32)
        ).unsqueeze(0).to(device)

    d["ri"] = d["ar"] = d["ai"] = d["tvw"] = d["nl_ref"] = None
    if entry.get("ri") is not None:
        ri_norm = np.clip(np.asarray(entry["ri"], dtype=np.float32)
                          / np.maximum(q99[None], 1e-9), 0.0, 5.0)
        ri_t = torch.from_numpy(ri_norm).to(device)
        d["ri"] = ri_t
        d["ri_norm"] = ri_norm
        d["ar"], d["ai"] = ri_t[0:1], ri_t[1:2]
        with torch.no_grad():
            if cfg.guide_tv:
                if cfg.polgroup_guides:
                    g_co = (ri_t[:, [0, 3]] ** 2).mean(dim=(0, 1)).sqrt()[None, None]
                    g_x = (ri_t[:, [1, 2]] ** 2).mean(dim=(0, 1)).sqrt()[None, None]
                    w_co = guide_edge_weights(g_co, alpha=cfg.guide_alpha,
                                              cv_protect=cfg.guide_cv_protect or None)
                    w_x = guide_edge_weights(g_x, alpha=cfg.guide_alpha,
                                             cv_protect=cfg.guide_cv_protect or None)
                    d["tvw"] = tuple(torch.cat([c, x, x, c], dim=1)
                                     for c, x in zip(w_co, w_x))
                else:
                    guide = (ri_t ** 2).mean(dim=(0, 1)).sqrt()[None, None]
                    d["tvw"] = guide_edge_weights(
                        guide, alpha=cfg.guide_alpha,
                        cv_protect=cfg.guide_cv_protect or None)
            if cfg.guide_nlm:
                d["nl_ref"] = _box_blur(0.5 * (d["ar"] + d["ai"]), k=3, passes=1)

    d["phase_factor"] = d["fid_w"] = d["nl_w"] = d["det"] = d["snr"] = None
    d["nl_sigma_map"] = d["helix"] = d["fact_w"] = d["tgt_db2"] = None
    if entry.get("pha") is not None:
        pm = entry["pha"] if isinstance(entry["pha"], dict) else \
            phase_feedback_maps(pha=np.asarray(entry["pha"], np.float32),
                                win=cfg.phase_win)
        snr_t = torch.from_numpy(pm["snr"]).float()[None, None].to(device)
        noise_map = 1.0 - snr_t
        d["snr"] = snr_t
        d["det"] = torch.from_numpy(pm["det"]).float()[None, None].to(device)
        boost = 1.0 + cfg.phase_smooth_boost * noise_map
        if cfg.phase_surface_boost > 0:
            surf = torch.from_numpy(pm["surface"]).float()[None, None].to(device)
            boost = boost + cfg.phase_surface_boost * surf
        if cfg.phase_protect > 0:
            boost = boost * (1.0 - cfg.phase_protect * d["det"])
        d["phase_factor"] = boost
        d["nl_w"] = boost / boost.mean()
        if cfg.phase_fidelity > 0:
            fw = 1.0 - cfg.phase_fidelity * noise_map
            d["fid_w"] = fw / fw.mean()
        if cfg.nlm_sigma_noise > 0:
            d["nl_sigma_map"] = cfg.nlm_sigma * (1.0 + cfg.nlm_sigma_noise * noise_map)
        if d["tvw"] is None:
            span_g = (d["noisy"] ** 2).mean(dim=1, keepdim=True).sqrt()
            d["tvw"] = guide_edge_weights(span_g, alpha=cfg.guide_alpha,
                                          cv_protect=cfg.guide_cv_protect or None)
        d["tvw"] = modulate_edge_weights(d["tvw"], d["phase_factor"])
        if cfg.phase_helix_protect > 0 and "helix" in pm:      # D3
            hx = torch.from_numpy(pm["helix"]).float()[None, None].to(device)
            d["helix"] = hx
            keep = 1.0 - cfg.phase_helix_protect * hx
            k_h, k_w = modulate_edge_weights(
                (torch.ones_like(keep[:, :, 1:]), torch.ones_like(keep[:, :, :, 1:])),
                keep)
            w_h, w_w = d["tvw"]
            w_h = w_h.expand(-1, 4, -1, -1).clone()
            w_w = w_w.expand(-1, 4, -1, -1).clone()
            w_h[:, 1:3] = w_h[:, 1:3] * k_h
            w_w[:, 1:3] = w_w[:, 1:3] * k_w
            d["tvw"] = (w_h, w_w)
        if cfg.fact_snr_gate > 0:                                # D4
            d["fact_w"] = (1.0 - cfg.fact_snr_gate) + cfg.fact_snr_gate * snr_t
    if cfg.xpol_target_debias > 0 and d["ri"] is not None:       # D1
        from .complex_data import estimate_thermal_sigma
        s_th = estimate_thermal_sigma(
            amp, pm["snr"] if entry.get("pha") is not None else None)
        s_n = torch.from_numpy(
            (s_th / q99[1:3, 0, 0]).astype(np.float32)).to(device)
        d["tgt_db2"] = cfg.xpol_target_debias * (s_n ** 2).view(1, 2, 1, 1)

    # edge mask for edge_boost / polish protection (as in trainer.py)
    d["edge_full"] = None
    if cfg.edge_boost > 0:
        with torch.no_grad():
            if d["ri"] is not None:
                g_span = (d["ri"] ** 2).mean(dim=(0, 1)).sqrt()[None, None]
            else:
                g_span = (d["noisy"] ** 2).mean(dim=1, keepdim=True).sqrt()
            span_log = 0.5 * torch.log(
                _box_blur(g_span, k=3, passes=2) ** 2 + 1e-6)
            e_h = (span_log[:, :, 1:, :] - span_log[:, :, :-1, :]).abs()
            e_w = (span_log[:, :, :, 1:] - span_log[:, :, :, :-1]).abs()
            if d["snr"] is not None and cfg.edge_phase_weight > 0:
                snr_s = _box_blur(d["snr"], k=3, passes=1)
                p_h = (snr_s[:, :, 1:, :] - snr_s[:, :, :-1, :]).abs()
                p_w = (snr_s[:, :, :, 1:] - snr_s[:, :, :, :-1]).abs()
                e_h = e_h / (e_h.mean() + 1e-8) \
                    + cfg.edge_phase_weight * p_h / (p_h.mean() + 1e-8)
                e_w = e_w / (e_w.mean() + 1e-8) \
                    + cfg.edge_phase_weight * p_w / (p_w.mean() + 1e-8)
            norm = 0.5 * (e_h.mean() + e_w.mean()) + 1e-8
            e_h, e_w = e_h / norm, e_w / norm
            ew = (e_h ** 2 / (1.0 + e_h ** 2), e_w ** 2 / (1.0 + e_w ** 2))
            ef = torch.zeros_like(g_span)
            ef[:, :, 1:, :] = torch.maximum(ef[:, :, 1:, :], ew[0])
            ef[:, :, :-1, :] = torch.maximum(ef[:, :, :-1, :], ew[0])
            ef[:, :, :, 1:] = torch.maximum(ef[:, :, :, 1:], ew[1])
            ef[:, :, :, :-1] = torch.maximum(ef[:, :, :, :-1], ew[1])
            d["edge_full"] = ef
    return d


def _crop_map(t, y, x, c):
    """Crop a (1, C, H, W) map; None passes through."""
    return None if t is None else t[:, :, y:y + c, x:x + c]


def _crop_tvw(tvw, y, x, c):
    """Crop the (weight_h, weight_w) TV edge-weight pair to a c x c crop."""
    if tvw is None:
        return None
    wh, ww = tvw
    return (wh[:, :, y:y + c - 1, x:x + c], ww[:, :, y:y + c, x:x + c - 1])


def denoise_scene(patches, cfg: TrainConfig = None, crop: int = 256,
                  batch: int = 4, verbose: bool = True):
    """Zero-shot denoise a set of quad-pol patches from ONE scene.

    Parameters
    ----------
    patches : list of dicts (``load_scene_patches`` format): 'amp' required,
        'ri' / 'pha' / 'sat' optional (must be present for every patch to be
        used — a patch missing them falls back to the amp-only paths for
        that feature).
    cfg : TrainConfig — same fields as the single-patch trainer.
    crop, batch : random-crop size and crops per step. batch=4, crop=256
        matches the pixel throughput of one 512x512 full-image step.

    Returns
    -------
    dict: 'denoised' — list of (4, H, W) float32 arrays (input scale, same
    order as ``patches``), 'stop_step', 'loss_hist'.
    """
    if cfg is None:
        cfg = TrainConfig()
    device = _resolve_device(cfg.device)
    if cfg.init_seed is not None:
        torch.manual_seed(cfg.init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.init_seed)
    rng = np.random.default_rng(cfg.init_seed if cfg.init_seed is not None else 0)

    # ── scene-level per-channel q99 ──
    all_amp = np.concatenate(
        [np.asarray(e["amp"], np.float32).reshape(4, -1) for e in patches], axis=1)
    q99 = np.quantile(all_amp, 0.99, axis=1, keepdims=True)[:, :, None]  # (4,1,1)
    del all_amp

    P = [_prep_patch(e, q99, cfg, device) for e in patches]
    nP = len(P)
    have_ri = all(d["ri"] is not None for d in P)
    have_pha = all(d["phase_factor"] is not None for d in P)
    merlin = have_ri and cfg.ri_mode == "merlin"

    model = SSPMNet(Config(**(cfg.model_cfg or {}))).to(device)
    masker = QuadPolSpatialMasker(keep_prob=cfg.mask_keep_prob).to(device)
    crit = MaskedL1Loss()
    n_iters = cfg.iters
    # D2: the snr map as the cross-pol branch's extra input plane
    use_aux = bool((cfg.model_cfg or {}).get("xpol_snr_input")) and have_pha

    # ── per-patch speckle factor bank ──
    if cfg.use_speckle_factor:
        S_bank = [torch.full((1, 4, d["H"], d["W"]), cfg.s_init,
                             dtype=torch.float32, device=device,
                             requires_grad=True) for d in P]
        opt = torch.optim.AdamW(
            [{"params": model.parameters(), "lr": cfg.lr},
             {"params": S_bank, "lr": cfg.lr * cfg.s_lr_mult}],
            weight_decay=1e-5)
    else:
        S_bank = None
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iters, eta_min=cfg.lr / 100.0)

    def sample_crops():
        """B random (patch, y, x) crop origins, 8-px aligned."""
        out = []
        for _ in range(batch):
            pi = int(rng.integers(0, nP))
            d = P[pi]
            y = int(rng.integers(0, max((d["H"] - crop) // 8 + 1, 1))) * 8
            x = int(rng.integers(0, max((d["W"] - crop) // 8 + 1, 1))) * 8
            out.append((pi, y, x))
        return out

    def stack(field, crops, chan4=True):
        ts = []
        for pi, y, x in crops:
            t = P[pi][field]
            if t is None:
                return None
            ts.append(t[:, :, y:y + crop, x:x + crop])
        return torch.cat(ts, dim=0)

    # ── Pre-warmup on bilateral targets (crops) ──
    if cfg.pre_warmup > 0:
        for d in P:
            src = d["ri_norm"].mean(axis=0) if d["ri"] is not None else d["amp_norm"]
            d["warm"] = torch.from_numpy(
                warmup_target_4ch(src)).float().unsqueeze(0).to(device)
        opt_pw = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        for i_pw in range(cfg.pre_warmup):
            model.train()
            crops = sample_crops()
            if merlin:
                k = i_pw % 2
                x_pw = torch.cat([P[pi]["ri"][k:k + 1, :, y:y + crop, x:x + crop]
                                  for pi, y, x in crops], dim=0)
            else:
                x_pw = stack("noisy", crops)
            tgt_pw = torch.cat([P[pi]["warm"][:, :, y:y + crop, x:x + crop]
                                for pi, y, x in crops], dim=0)
            aux_pw = stack("snr", crops) if use_aux else None
            loss_pw = ((model(x_pw, aux=aux_pw) - tgt_pw) ** 2).mean()
            opt_pw.zero_grad()
            loss_pw.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pw.step()
        if verbose:
            print(f"  [pre-warmup] {cfg.pre_warmup} steps, "
                  f"final loss={loss_pw.item():.5f}")

    model_ema = None
    if cfg.use_ema:
        model_ema = copy.deepcopy(model)
        for p_ in model_ema.parameters():
            p_.requires_grad = False

    if cfg.hist_lambda > 0:
        h_ref_t, bin_centers_t, hist_step = compute_reference_histogram(
            looks=cfg.looks_ref, n_bins=cfg.hist_bins,
            range_max=cfg.hist_range, device=device)

    if verbose:
        print(f"  [scene-train] {nP} patches, crop={crop} batch={batch} "
              f"iters={n_iters} merlin={merlin} pha={have_pha} "
              f"model_cfg={cfg.model_cfg}")

    loss_hist = []
    for step in range(n_iters):
        model.train()
        masker.train()
        ltv = math.cos(math.pi * step / max(n_iters, 1))
        lambda_tv = cfg.lambda_tv_end + 0.5 * (cfg.lambda_tv_start
                                               - cfg.lambda_tv_end) * (1.0 + ltv)
        crops = sample_crops()
        noisy_b = stack("noisy", crops)
        fid_b = stack("fid_w", crops)
        satk_b = stack("sat_keep", crops)
        aux_b = stack("snr", crops) if use_aux else None
        fact_b = stack("fact_w", crops)
        tvw_b = None
        if P[crops[0][0]]["tvw"] is not None:
            whs, wws = [], []
            for pi, y, x in crops:
                wh, ww = _crop_tvw(P[pi]["tvw"], y, x, crop)
                whs.append(wh)
                wws.append(ww)
            tvw_b = (torch.cat(whs, 0), torch.cat(wws, 0))

        if merlin:
            k = step % 2
            x_in = torch.cat([P[pi]["ri"][k:k + 1, :, y:y + crop, x:x + crop]
                              for pi, y, x in crops], dim=0)
            tgt = torch.cat([P[pi]["ri"][1 - k:2 - k, :, y:y + crop, x:x + crop]
                             for pi, y, x in crops], dim=0)
            d_out = model(x_in, aux=aux_b)

            def _wmean(t, tgt_ch, use_fid=False):
                w = None
                if satk_b is not None:
                    chs = tgt_ch if isinstance(tgt_ch, (list, tuple)) else [tgt_ch]
                    w = satk_b[:, chs].prod(dim=1, keepdim=True)
                if use_fid and fid_b is not None:
                    w = fid_b if w is None else w * fid_b
                if w is None:
                    return t.mean()
                return (t * w).sum() / w.sum().clamp(min=1.0)

            l_hh = _wmean((d_out[:, 0:1] - tgt[:, 0:1]).abs(), 0)
            l_vv = _wmean((d_out[:, 3:4] - tgt[:, 3:4]).abs(), 3)
            loss_copol = (l_hh + l_vv) / 2
            t_x = tgt[:, 1:3]
            if cfg.xpol_target_debias > 0:                       # D1
                db2 = torch.cat([P[pi]["tgt_db2"].expand(1, 2, crop, crop)
                                 for pi, _, _ in crops], dim=0)
                t_x = torch.sqrt((t_x ** 2 - db2).clamp(min=0.0))
            if cfg.xpol_fused_target:                            # D1
                t_f = 0.5 * (t_x[:, 0:1] + t_x[:, 1:2])
                l_hv = _wmean((d_out[:, 1:2] - t_f).abs(), [1, 2], True)
                l_vh = _wmean((d_out[:, 2:3] - t_f).abs(), [1, 2], True)
            else:
                w_r = cfg.merlin_recip_weight
                l_hv = ((1 - w_r) * _wmean((d_out[:, 1:2] - t_x[:, 0:1]).abs(), 1, True)
                        + w_r * _wmean((d_out[:, 1:2] - t_x[:, 1:2]).abs(), 2, True))
                l_vh = ((1 - w_r) * _wmean((d_out[:, 2:3] - t_x[:, 1:2]).abs(), 2, True)
                        + w_r * _wmean((d_out[:, 2:3] - t_x[:, 0:1]).abs(), 1, True))
            loss_xpol = (l_hv + l_vh) / 2
        else:
            ar_b = stack("ar", crops)
            ai_b = stack("ai", crops)
            m = masker(noisy_b)
            d_out = model(m["masked_input"], aux=aux_b)

            def masked_loss(pred, tgt_ch, mask):
                l_amp = crit(pred, noisy_b[:, tgt_ch:tgt_ch + 1], mask)
                if ar_b is None or cfg.ri_weight <= 0:
                    return l_amp
                l_ri = 0.5 * (crit(pred, ar_b[:, tgt_ch:tgt_ch + 1], mask)
                              + crit(pred, ai_b[:, tgt_ch:tgt_ch + 1], mask))
                return (1.0 - cfg.ri_weight) * l_amp + cfg.ri_weight * l_ri

            l_hh = masked_loss(d_out[:, 0:1], 0, m["mask_hh"])
            l_vv = masked_loss(d_out[:, 3:4], 3, m["mask_vv"])
            loss_copol = (l_hh + l_vv) / 2
            mxp = m["mask_xpol"]
            l_hv = masked_loss(d_out[:, 1:2], 2, mxp)
            l_vh = masked_loss(d_out[:, 2:3], 1, mxp)
            loss_xpol = (l_hv + l_vh) / 2

        # ── regularizers on the crop batch ──
        l_tv = adaptive_tv_loss(d_out, noisy_b, weights=tvw_b)
        l_pol = polarization_consistency_loss(d_out)
        l_bound = bound_loss(d_out)
        nlm_lam = cfg.nlm_lambda
        if cfg.nlm_lambda_end > 0 and n_iters > 1:
            nlm_lam = (cfg.nlm_lambda + (cfg.nlm_lambda_end - cfg.nlm_lambda)
                       * step / (n_iters - 1))
        nlw_b = stack("nl_w", crops)
        nlsig_b = stack("nl_sigma_map", crops)
        nlref_b = stack("nl_ref", crops)
        _dg = [[0, 3], [1, 2]] if cfg.polgroup_guides else None
        l_nl = (polarimetric_nl_loss(
                    d_out, nlref_b if nlref_b is not None else noisy_b,
                    cfg.nlm_window, cfg.nlm_sigma,
                    pixel_weight=nlw_b, sigma_map=nlsig_b, dist_groups=_dg)
                if cfg.nlm_lambda > 0 else torch.tensor(0.0, device=device))
        l_white = (ratio_whiteness_loss(d_out, noisy_b, lags=cfg.whiteness_lags,
                                        per_channel=cfg.polgroup_guides)
                   if cfg.whiteness_lambda > 0
                   else torch.tensor(0.0, device=device))

        if cfg.use_speckle_factor and S_bank is not None:
            S_crops = torch.cat(
                [S_bank[pi][:, :, y:y + crop, x:x + crop]
                 for pi, y, x in crops], dim=0)
            S_pos = torch.nn.functional.softplus(S_crops)
            sq_f = (d_out * S_pos - noisy_b) ** 2
            if fact_b is not None:                               # D4
                sq_f = torch.cat([sq_f[:, 0:1], sq_f[:, 1:3] * fact_b,
                                  sq_f[:, 3:4]], dim=1)
            l_fact = sq_f.mean()
            if cfg.hist_lambda > 0:
                histos, marg = [], 0.0
                for c in range(4):
                    s_c = torch.clamp(S_pos[:, c], 0.0, cfg.hist_range)
                    h_c = compute_soft_histogram(
                        s_c, bin_centers_t, hist_step,
                        weight=fact_b[:, 0] if (fact_b is not None
                                                and c in (1, 2)) else None)
                    marg = marg + ((h_c - h_ref_t) ** 2).sum()
                    histos.append(h_c)
                l_hist = (marg / 4.0 + cfg.hist_recip_weight
                          * ((histos[1] - histos[2]) ** 2).sum())
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
                + cfg.hist_lambda * l_hist
                + cfg.lambda_fact * l_fact)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_hist.append(float(loss.item()))

        if cfg.use_ema and model_ema is not None:
            with torch.no_grad():
                for p_ema, p_ in zip(model_ema.parameters(), model.parameters()):
                    p_ema.data.mul_(cfg.ema_decay).add_(
                        p_.data, alpha=1.0 - cfg.ema_decay)

        if verbose and ((step + 1) % cfg.snapshot_every == 0 or step == 0):
            print(f"  step {step+1:>4d}/{n_iters} loss={loss.item():.4f}",
                  flush=True)

    # ── Final inference per patch: EMA + D4 x MC-dropout TTA ──
    final_model = model_ema if (cfg.use_ema and model_ema is not None) else model
    outs = []
    for d in P:
        infer_inputs = ([d["ri"][0:1], d["ri"][1:2]] if merlin
                        else [d["noisy"]])
        final_model.train()
        aux_p = d["snr"] if use_aux else None
        with torch.no_grad():
            acc = torch.zeros_like(d["noisy"])
            cnt = 0
            if cfg.use_tta:
                for x_base in infer_inputs:
                    for k_rot in range(4):
                        for do_flip in (False, True):
                            x_aug = torch.rot90(x_base, k_rot, dims=[-2, -1])
                            a_aug = (torch.rot90(aux_p, k_rot, dims=[-2, -1])
                                     if aux_p is not None else None)
                            if do_flip:
                                x_aug = torch.flip(x_aug, dims=[-1])
                                if a_aug is not None:
                                    a_aug = torch.flip(a_aug, dims=[-1])
                            for _ in range(cfg.tta_mc_passes):
                                out = final_model(x_aug, aux=a_aug).clamp(0, 1)
                                if do_flip:
                                    out = torch.flip(out, dims=[-1])
                                acc += torch.rot90(out, -k_rot, dims=[-2, -1])
                                cnt += 1
            else:
                for x_base in infer_inputs:
                    for _ in range(max(cfg.n_inference // len(infer_inputs), 1)):
                        acc += final_model(x_base, aux=aux_p).clamp(0, 1)
                        cnt += 1
            acc /= cnt

            if cfg.polish > 0:
                if d["ri"] is not None:
                    g_amp = (d["ri"] ** 2).mean(dim=(0, 1)).sqrt()[None, None]
                else:
                    g_amp = (d["noisy"] ** 2).mean(dim=1, keepdim=True).sqrt()
                mu_g = _box_blur(g_amp, k=9, passes=1)
                m2_g = _box_blur(g_amp ** 2, k=9, passes=1)
                cv_g = torch.sqrt((m2_g - mu_g ** 2).clamp(min=0)) / (mu_g + 1e-6)
                thr = cfg.guide_cv_protect if cfg.guide_cv_protect > 0 else 0.3
                prot = torch.sigmoid((cv_g - thr) / (0.25 * thr))
                if d["det"] is not None:
                    prot = torch.maximum(prot, d["det"])
                if d["edge_full"] is not None:
                    prot = torch.maximum(prot, d["edge_full"])
                if d["helix"] is not None:                       # D3
                    prot = prot.expand(-1, 4, -1, -1).clone()
                    prot[:, 1:3] = torch.maximum(
                        prot[:, 1:3], cfg.phase_helix_protect * d["helix"])
                acc = nl_polish(acc, window=cfg.polish_window,
                                sigma=cfg.polish_sigma, strength=cfg.polish,
                                protect=prot,
                                dist_groups=([[0, 3], [1, 2]]
                                             if cfg.polgroup_guides else None)
                                ).clamp(0, 1)

            if cfg.edge_boost > 0 and d["edge_full"] is not None:
                m_edge = _box_blur(d["edge_full"], k=3, passes=1)
                if cfg.edge_boost_dark > 0:
                    mu_ch = _box_blur(acc, k=9, passes=1)
                    m_edge = m_edge * (mu_ch / (mu_ch + cfg.edge_boost_dark))
                acc = (acc + cfg.edge_boost * m_edge
                       * (acc - _box_blur(acc, k=3, passes=1))).clamp(0, 1)

        out_np = acc[0].cpu().numpy() * q99.squeeze()[:, None, None]
        if cfg.thermal_debias > 0:
            from .complex_data import estimate_thermal_sigma
            snr_np = (d["snr"][0, 0].cpu().numpy()
                      if d.get("snr") is not None else None)
            amp_raw = d["noisy"][0].cpu().numpy() \
                * q99.squeeze()[:, None, None]      # clip at 5*q99 is far
            s_th = estimate_thermal_sigma(amp_raw, snr_np)  # above the dark
            for c in (1, 2):                                # pixels used
                out_np[c] = np.sqrt(np.maximum(
                    out_np[c] ** 2 - cfg.thermal_debias * s_th ** 2, 0.0))
        outs.append(out_np.astype(np.float32))

    if verbose:
        print(f"  [final] {'EMA' if cfg.use_ema else 'raw'} weights, "
              f"{nP} patches denoised"
              + (" + D4xMC-dropout TTA" if cfg.use_tta else ""))

    del model, masker, opt
    if model_ema is not None:
        del model_ema
    if S_bank is not None:
        del S_bank
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"denoised": outs, "stop_step": int(n_iters),
            "loss_hist": loss_hist}
