"""Patch group ``fast_init`` (opt-in): fast weight initialisation when the weights will be loaded anyway.

Building OpenFold3 draws every weight with ``scipy.stats.truncnorm`` before a checkpoint overwrites it
(about 43 s on 2 CPU cores for the full 368M-parameter model, on every start). The replacement draws from
the *same distribution* (truncated normal, same std and truncation points) with PyTorch's native sampler.
Initialisation stays valid for from-scratch training too, but the random stream differs, so seeded
from-scratch initialisations are no longer reproducible against upstream; hence opt-in.
"""

from __future__ import annotations

import functools
import math

import torch

from .._patch import Patch

GROUP = "fast_init"
SUMMARY = "OPT-IN: PyTorch-native truncated-normal weight init (same distribution, different random stream)"

_TRUNC_STD = (
    0.8796256610342398  # std of N(0,1) truncated to [-2, 2] (== scipy truncnorm.std(a=-2, b=2))
)


def _trunc_normal(orig):
    @functools.wraps(orig)
    def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
        if weights.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            return orig(weights, scale, fan)
        import openfold3.core.model.primitives.initialization as init

        f = init._calculate_fan(weights.shape, fan)
        std = math.sqrt(scale / max(1, f)) / _TRUNC_STD
        with torch.no_grad():
            torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-2 * std, b=2 * std)

    return trunc_normal_init_


def targets() -> list[Patch]:
    return [
        Patch(
            "initialization.trunc_normal_init_",
            GROUP,
            ("openfold3.core.model.primitives.initialization", None, "trunc_normal_init_"),
            _trunc_normal,
        )
    ]
