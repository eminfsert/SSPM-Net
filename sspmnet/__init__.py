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
from .complex_data import (load_quadpol_tiffs, calibrate_ri,
                           load_scene_patches, load_quadpol_slc)
from .scene_trainer import denoise_scene
from .phase_data import load_quadpol_phase, phase_feedback_maps
from . import metrics, losses, complex_data, phase_data, spectral

__all__ = [
    "Config",
    "SSPMNet", "SARDenoiser", "DenoiseBranch", "ChannelRefinement",
    "QuadPolSpatialMasker", "BernoulliMasker",
    "TrainConfig", "denoise",
    "load_quadpol_tiffs", "calibrate_ri", "load_quadpol_slc",
    "load_scene_patches", "denoise_scene",
    "load_quadpol_phase", "phase_feedback_maps",
    "metrics", "losses", "complex_data", "phase_data", "spectral",
]

__version__ = "1.0.0"
