"""
SSPM-Net — physics-aware zero-shot quad-pol SAR despeckling.

Quick start
-----------
    import numpy as np
    from sspmnet import denoise, TrainConfig

    amp = np.load("data/example_quadpol.npy")      # (4, H, W): HH, HV, VH, VV
    result = denoise(amp, TrainConfig(iters=1000))
    denoised = result["denoised"]                  # (4, H, W)
"""
from .config import Config
from .model import SSPMNet, SARDenoiser, DenoiseBranch, ChannelRefinement
from .masking import QuadPolSpatialMasker, BernoulliMasker
from .trainer import TrainConfig, denoise
from .complex_data import load_quadpol_tiffs, calibrate_ri
from . import metrics, losses, complex_data

__all__ = [
    "Config",
    "SSPMNet", "SARDenoiser", "DenoiseBranch", "ChannelRefinement",
    "QuadPolSpatialMasker", "BernoulliMasker",
    "TrainConfig", "denoise",
    "load_quadpol_tiffs", "calibrate_ri",
    "metrics", "losses", "complex_data",
]

__version__ = "1.0.0"
