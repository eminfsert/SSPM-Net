"""
Architectural configuration for SSPM-Net.

Only the *model* hyper-parameters live here. Training hyper-parameters
(iterations, learning rate, loss weights, ...) live in
``sspmnet.trainer.TrainConfig``.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Model architecture configuration."""

    # ── Input ────────────────────────────────────────────────────────
    num_polarizations: int = 4          # HH, HV, VH, VV
    in_channels: int = 4

    # ── Frequency decomposition (fixed 1-level Haar DWT) ─────────────
    wavelet: str = "haar"
    wavelet_levels: int = 1

    # ── CNN branch — processes the HIGH-frequency detail sub-bands ───
    #    (class is named ``LowFreqBranch`` for historical reasons)
    low_freq_channels: int = 64
    low_freq_num_blocks: int = 5
    # per-block conv dilation, e.g. [1, 2, 4, 8, 1] widens the CNN receptive
    # field ~2.4x with zero extra parameters; None = all 1 (historical)
    low_freq_dilations: List[int] = None

    # ── Swin branch — processes the LL (low-frequency) sub-band ──────
    #    (class is named ``HighFreqBranch`` for historical reasons)
    high_freq_embed_dim: int = 96
    high_freq_depths: List[int] = field(default_factory=lambda: [2, 2])
    high_freq_num_heads: List[int] = field(default_factory=lambda: [3, 6])
    high_freq_window_size: int = 8
    high_freq_mlp_ratio: float = 4.0
    high_freq_drop_rate: float = 0.0
    high_freq_attn_drop_rate: float = 0.1

    # ── Reconstruction ───────────────────────────────────────────────
    recon_channels: int = 64

    # ── Cross-polarization attention ─────────────────────────────────
    cross_attn_dim: int = 64
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1

    # ── Monte-Carlo dropout (Self2Self) ──────────────────────────────
    dropout_rate: float = 0.3
    # "band": historical nn.Dropout2d — on the 1-channel LL output this
    # zeroes the WHOLE low-frequency band with prob p; "pixel": nn.Dropout
    # (i.i.d. per element). See sspmnet.layers.
    dropout_style: str = "band"

    # ── Normalization layers ─────────────────────────────────────────
    # "batch": historical BatchNorm2d (batch-1 statistics; EMA never
    # updates the running buffers); "group": GroupNorm(min(8,C), C).
    norm: str = "batch"

    # ── Cross-pol pair input ─────────────────────────────────────────
    # When True the cross-pol branch receives BOTH reciprocal planes
    # (self first, reciprocal second): HV/VH are the same physical channel
    # measured twice with independent thermal noise, so the second plane
    # is a genuine extra look the branch can fuse spatially.
    xpol_pair_input: bool = False

    # ── Device ───────────────────────────────────────────────────────
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
