"""
CNN branch.

Despite the historical name ``LowFreqBranch``, in SSPM-Net this branch
processes the HIGH-frequency detail sub-bands (LH + HL + HH) where local
edges and speckle live. It is a lightweight residual CNN.
"""
import torch
import torch.nn as nn

from .layers import make_norm, make_dropout


class ResidualBlock(nn.Module):
    """Pre-activation residual block: (Norm -> LReLU -> Conv) x2 + skip."""

    def __init__(self, channels: int, norm: str = "batch"):
        super().__init__()
        self.block = nn.Sequential(
            make_norm(channels, norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            make_norm(channels, norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LowFreqBranch(nn.Module):
    """Residual CNN over the detail sub-bands.

    Input projection -> N x ResidualBlock (with dropout) -> output projection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        num_blocks: int = 5,
        out_channels: int = None,
        dropout: float = 0.3,
        dropout_style: str = "band",
        norm: str = "batch",
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.proj_in = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)

        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock(mid_channels, norm=norm))
            if (i + 1) % 2 == 0:                 # dropout every 2 blocks
                layers.append(make_dropout(dropout, dropout_style))
        self.blocks = nn.Sequential(*layers)

        self.proj_out = nn.Sequential(
            make_norm(mid_channels, norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj_in(x)
        feat = self.blocks(feat)
        return self.proj_out(feat)
