"""
CNN branch.

Despite the historical name ``LowFreqBranch``, in SSPM-Net this branch
processes the HIGH-frequency detail sub-bands (LH + HL + HH) where local
edges and speckle live. It is a lightweight residual CNN.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block: (BN -> LReLU -> Conv) x2 + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
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
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.proj_in = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)

        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock(mid_channels))
            if (i + 1) % 2 == 0:                 # dropout every 2 blocks
                layers.append(nn.Dropout2d(p=dropout))
        self.blocks = nn.Sequential(*layers)

        self.proj_out = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj_in(x)
        feat = self.blocks(feat)
        return self.proj_out(feat)
