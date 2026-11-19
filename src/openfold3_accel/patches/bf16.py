"""Patch group ``bf16_diffusion`` (opt-in, approximate): run the diffusion denoiser under CPU bf16 autocast.

On CPUs with AMX / AVX512-BF16 the denoiser GEMMs (which dominate CPU inference after caching) are
several times faster in bfloat16. AlphaFold3-style models are trained with bf16 in this module, but
results are *not* bit-identical, so this group is never enabled by default. Enable with
``openfold3_accel.enable(["diffusion_cache", "bf16_diffusion"])`` or ``OPENFOLD3_ACCEL_GROUPS``.
Layer norms and softmax stay in fp32 (autocast policy); coordinates are returned as fp32.
"""

from __future__ import annotations

import functools
import os
import threading
import warnings

import torch

from .._patch import Patch

_ROLLOUT_LOCK = threading.RLock()
GROUP = "bf16_diffusion"
SUMMARY = "OPT-IN, approximate: bf16 autocast for the diffusion denoiser on AMX/AVX512-BF16 CPUs"


def cpu_supports_bf16() -> bool:
    """True if this CPU has AMX-BF16 or AVX512-BF16 (where bf16 GEMMs are actually faster)."""
    for name in ("_is_amx_tile_supported", "_is_avx512_bf16_supported", "_is_amx_bf16_supported"):
        f = getattr(torch.cpu, name, None)
        try:
            if f is not None and f():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


_WARNED = False


def _forward(orig):
    @functools.wraps(orig)
    def forward(self, batch, xl_noisy, *args, **kwargs):
        if (
            torch.is_grad_enabled()
            or self.training
            or xl_noisy.device.type != "cpu"
            or not _on_cpu(self)
        ):
            return orig(self, batch, xl_noisy, *args, **kwargs)
        if not _bf16_ok():
            return orig(self, batch, xl_noisy, *args, **kwargs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = orig(self, batch, xl_noisy, *args, **kwargs)
        return out.to(xl_noisy.dtype)

    return forward


def _rollout(orig):
    """Open the autocast context once per rollout so its weight-cast cache persists across steps.

    Autocast only caches fp32->bf16 weight casts for leaf tensors that require grad. Under no_grad inference
    the weights are cast again on every call (about a fifth of CPU time in our profile), so the weights'
    ``requires_grad`` flag is raised for the duration of the rollout (no graph is recorded: grad is off) and
    restored afterwards. Policy: bf16 rollouts are serialised per process by a lock, and other threads must not
    read or change those flags meanwhile. ``OPENFOLD3_ACCEL_NO_CASTCACHE=1`` skips the flag toggle entirely.
    """

    @functools.wraps(orig)
    def _sample_rollout(self, *args, **kwargs):
        if torch.is_grad_enabled() or self.training or not _on_cpu(self) or not _bf16_ok():
            return orig(self, *args, **kwargs)
        if os.environ.get("OPENFOLD3_ACCEL_NO_CASTCACHE") == "1":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return orig(self, *args, **kwargs)
        # The flag toggle is process-global state on shared parameters: serialise rollouts so two threads can
        # never interleave toggling and restoring, and so a restore never reverts another thread's change.
        with _ROLLOUT_LOCK:
            params = [p for p in self.parameters() if p.is_floating_point() and not p.requires_grad]
            for p in params:
                p.requires_grad_(True)
            try:
                with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
                    return orig(self, *args, **kwargs)
            finally:
                for p in params:
                    p.requires_grad_(False)

    return _sample_rollout


def _on_cpu(module) -> bool:
    """True when the module's weights live on the CPU (the only place this group helps)."""
    p = next(module.parameters(), None)
    return p is not None and p.device.type == "cpu"


def _bf16_ok() -> bool:
    global _WARNED
    if cpu_supports_bf16() or os.environ.get("OPENFOLD3_ACCEL_BF16_FORCE") == "1":
        return True
    if not _WARNED:
        _WARNED = True
        warnings.warn(
            "bf16_diffusion: this CPU lacks AMX/AVX512-BF16, so bf16 would be slower and less "
            "accurate; running fp32. Set OPENFOLD3_ACCEL_BF16_FORCE=1 to override.",
            RuntimeWarning,
            stacklevel=2,
        )
    return False


def targets() -> list[Patch]:
    return [
        Patch(
            "SampleDiffusion._sample_rollout[bf16]",
            GROUP,
            (
                "openfold3.core.model.structure.diffusion_module",
                "SampleDiffusion",
                "_sample_rollout",
            ),
            _rollout,
        ),
        Patch(
            "DiffusionModule.forward",
            GROUP,
            ("openfold3.core.model.structure.diffusion_module", "DiffusionModule", "forward"),
            _forward,
        ),
    ]
