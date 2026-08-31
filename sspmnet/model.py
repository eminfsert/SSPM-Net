"""
SSPM-Net — physics-aware zero-shot quad-pol SAR despeckling network.

Pipeline (input already normalized to [0, 1]):

    Input (B, 4, H, W)  [HH, HV, VH, VV]
        HH, VV -> CopolBranch   (shared weights)
        HV, VH -> CrosspolBranch(shared weights)
    -> Cross-Polarization Attention
    -> Per-channel Refinement x4
    -> Output (B, 4, H, W)

Each branch does: DWT -> Swin on LL (global context) + CNN on the detail
sub-bands (local edges) -> inverse DWT -> feature map.

Note: the sub-module class names ``HighFreqBranch`` (Swin, on LL) and
``LowFreqBranch`` (CNN, on the detail bands) are kept for historical
stability; their roles are as described above.
"""
import torch
import torch.nn as nn

from .config import Config
from .low_freq_branch import LowFreqBranch
from .high_freq_branch import HighFreqBranch
from .reconstruction import ReconstructionLayer
from .cross_attention import CrossPolarizationAttention
from .freq_decomposition import FrequencyDecomposition
from .layers import make_norm, make_dropout


class DenoiseBranch(nn.Module):
    """Single-channel wavelet denoising branch.

    Swin Transformer -> LL sub-band (global context, low noise)
    CNN              -> LH+HL+HH sub-bands (local edges, high noise)
    """

    def __init__(self, cfg: Config, feat_out: int = 64):
        super().__init__()
        drop_p = cfg.dropout_rate

        self.freq_decomp = FrequencyDecomposition()

        # Swin processes the LL sub-band (1 channel)
        self.ll_branch = HighFreqBranch(
            in_channels=1,
            embed_dim=cfg.high_freq_embed_dim,
            depths=cfg.high_freq_depths,
            num_heads=cfg.high_freq_num_heads,
            window_size=cfg.high_freq_window_size,
            mlp_ratio=cfg.high_freq_mlp_ratio,
            drop_rate=cfg.high_freq_drop_rate,
            attn_drop_rate=cfg.high_freq_attn_drop_rate,
            out_channels=1,
        )
        self.drop_ll = make_dropout(drop_p, cfg.dropout_style)

        # CNN processes the detail sub-bands (3 channels: LH, HL, HH)
        self.hf_branch = LowFreqBranch(
            in_channels=3,
            mid_channels=cfg.low_freq_channels,
            num_blocks=cfg.low_freq_num_blocks,
            out_channels=3,
            dropout=drop_p,
            dropout_style=cfg.dropout_style,
            norm=cfg.norm,
        )
        self.drop_hf = make_dropout(drop_p, cfg.dropout_style)

        self.reconstruction = ReconstructionLayer(
            in_channels=1,
            mid_channels=cfg.recon_channels,
            out_channels=feat_out,
            norm=cfg.norm,
        )
        self.drop_recon = make_dropout(drop_p, cfg.dropout_style)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> feature map (B, D, H, W)."""
        _, _, H, W = x.shape
        low, high = self.freq_decomp.decompose(x)

        ll_out = self.drop_ll(self.ll_branch(low))
        hf_out = self.drop_hf(self.hf_branch(high))

        return self.drop_recon(
            self.reconstruction(ll_out, hf_out, output_size=(H, W))
        )


class ChannelRefinement(nn.Module):
    """Per-channel refinement: feature map -> 1-channel output."""

    def __init__(self, feat_dim: int = 64, dropout: float = 0.3,
                 dropout_style: str = "band", norm: str = "batch"):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            make_norm(feat_dim, norm),
            nn.LeakyReLU(0.2, inplace=True),
            make_dropout(dropout, dropout_style),
            nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1, bias=False),
            make_norm(feat_dim // 2, norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(x)


class SSPMNet(nn.Module):
    """Physics-aware zero-shot quad-pol SAR denoiser (SSPM-Net).

    Operates purely in normalized [0, 1] space. Input channel order is
    [HH, HV, VH, VV]. Output has the same shape.
    """

    def __init__(self, cfg: Config = None):
        super().__init__()
        if cfg is None:
            cfg = Config()

        D = cfg.cross_attn_dim

        # Asymmetric branches: co-pol (HH/VV) and cross-pol (HV/VH)
        self.copol_branch = DenoiseBranch(cfg, feat_out=D)
        self.xpol_branch = DenoiseBranch(cfg, feat_out=D)

        self.cross_attn = CrossPolarizationAttention(
            feat_dim=D,
            num_heads=cfg.cross_attn_heads,
            num_pols=4,
            dropout=cfg.cross_attn_dropout,
        )

        self.refinements = nn.ModuleList([
            ChannelRefinement(feat_dim=D, dropout=cfg.dropout_rate,
                              dropout_style=cfg.dropout_style, norm=cfg.norm)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, H, W) in [0, 1] -> denoised (B, 4, H, W)."""
        hh = x[:, 0:1]
        hv = x[:, 1:2]
        vh = x[:, 2:3]
        vv = x[:, 3:4]

        feat_hh = self.copol_branch(hh)
        feat_vv = self.copol_branch(vv)     # shares weights with HH
        feat_hv = self.xpol_branch(hv)
        feat_vh = self.xpol_branch(vh)      # shares weights with HV

        features = self.cross_attn([feat_hh, feat_hv, feat_vh, feat_vv])
        outputs = [self.refinements[i](features[i]) for i in range(4)]
        return torch.cat(outputs, dim=1)


# Backward-compatible alias (the class was historically named SARDenoiser)
SARDenoiser = SSPMNet
