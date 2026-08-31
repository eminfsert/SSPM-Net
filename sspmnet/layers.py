"""Small factory helpers for ablatable architecture choices.

``norm``:
    "batch" — nn.BatchNorm2d (historical default; note: with batch size 1
        and train()-mode MC-dropout inference it acts on single-image batch
        statistics, and the EMA model never updates its buffers)
    "group" — nn.GroupNorm(min(8, C), C): batch-size independent, no
        running buffers, consistent between training and EMA inference.

``dropout_style``:
    "band"  — nn.Dropout2d (historical default). NOTE: on the 1-channel LL
        output this zeroes the ENTIRE low-frequency band with prob p, and
        on the 3-channel detail output it drops whole sub-bands.
    "pixel" — nn.Dropout: i.i.d. per-element masking (Self2Self style).
"""
import torch.nn as nn


def make_norm(channels: int, kind: str = "batch") -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"unknown norm kind: {kind!r}")


def make_dropout(p: float, style: str = "band") -> nn.Module:
    if style == "band":
        return nn.Dropout2d(p=p)
    if style == "pixel":
        return nn.Dropout(p=p)
    raise ValueError(f"unknown dropout style: {style!r}")
