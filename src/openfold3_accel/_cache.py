"""Rollout-scoped memoisation of tensors that do not change between denoising steps.

OpenFold3's diffusion sampler calls the denoiser ~200 times per sample with *identical* trunk
conditioning (``zij``, ``si_trunk``, reference-conformer features, masks). Only the noisy
coordinates and the noise-level embedding change. Many sub-expressions of the denoiser depend only
on the static inputs (pair biases, ``LayerNorm(z)``, the atom-pair MLP, the ``s``-dependent half of
every atom-level adaptive layer-norm, ...). They are recomputed on every step by upstream code.

This module provides the small amount of machinery that lets patched layers compute such terms
once per rollout:

* a :class:`RolloutCache` that is *active only inside* a patched ``SampleDiffusion._sample_rollout``
  (inference, ``torch.no_grad``), so training, activation checkpointing and any other use of the
  layers is untouched;
* a registry of tensors known to be static (matched by object identity *and* ``_version`` so
  in-place edits invalidate), and a byte budget so that very large inputs degrade gracefully
  (entries that do not fit are simply recomputed, exactly as upstream does).
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import Tensor

_ACTIVE: contextvars.ContextVar[RolloutCache | None] = contextvars.ContextVar(
    "openfold3_accel_rollout_cache", default=None
)

#: Fraction of *available* RAM the cache may use when no explicit budget is configured.
_RAM_FRACTION = 0.25
_MAX_DEFAULT_BYTES = 8 * 2**30


def _available_ram_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def default_budget_bytes() -> int:
    """Cache budget: ``OPENFOLD3_ACCEL_CACHE_MB`` if set, else 25% of available RAM (max 8 GiB)."""
    env = os.environ.get("OPENFOLD3_ACCEL_CACHE_MB")
    if env:
        try:
            return max(0, int(float(env) * 2**20))
        except ValueError:
            pass
    avail = _available_ram_bytes()
    if avail is None:
        return 2 * 2**30
    return int(min(_MAX_DEFAULT_BYTES, avail * _RAM_FRACTION))


def _nbytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, tuple | list):
        return sum(_nbytes(v) for v in value)
    return 0


def _tensors(value: Any) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, tuple | list):
        for v in value:
            yield from _tensors(v)


def _ver(t: Tensor) -> int:
    """Version counter of ``t``; -1 for inference tensors, which do not track one.

    Inference tensors (created under ``torch.inference_mode()``, e.g. by Lightning's ``Trainer.predict``)
    raise on ``_version``. They cannot be modified outside inference mode, and OpenFold3 never mutates
    the rollout's conditioning tensors in place, so identity alone is used for them.
    """
    try:
        return t._version
    except RuntimeError:
        return -1


class RolloutCache:
    """Per-rollout memo table. Not thread-safe by design: one cache lives in one context."""

    def __init__(self, budget_bytes: int | None = None):
        self.budget = default_budget_bytes() if budget_bytes is None else budget_bytes
        self.used = 0
        self.hits = 0
        self.misses = 0
        self.overflow = 0
        self.by_key: dict[str, list[int]] = {}
        self._static: dict[int, tuple[Tensor, int]] = {}
        self._memo: dict[Any, Any] = {}

    # -- static-tensor registry ---------------------------------------------------------------
    def mark_static(self, *tensors: Tensor | None) -> None:
        """Declare tensors constant for the lifetime of the rollout (strong refs are kept)."""
        for t in tensors:
            if isinstance(t, Tensor):
                self._static[id(t)] = (t, _ver(t))

    def is_static(self, t: Tensor | None) -> bool:
        if not isinstance(t, Tensor):
            return False
        entry = self._static.get(id(t))
        return entry is not None and entry[0] is t and entry[1] == _ver(t)

    # -- memoisation ----------------------------------------------------------------------------
    def memo(self, key: Any, compute: Callable[[], Any], *, mark: bool = True) -> Any:
        """Return ``compute()``, computing it at most once per rollout (budget permitting).

        Tensors in the result are registered as static so that downstream layers can memoise on
        them too. When the budget is exhausted the value is returned uncached and *not*
        registered, so everything derived from it is also recomputed - i.e. upstream behaviour.
        """
        name = str(key[0]) if isinstance(key, tuple) else str(key)
        counts = self.by_key.setdefault(name, [0, 0])
        if key in self._memo:
            self.hits += 1
            counts[0] += 1
            return self._memo[key]
        self.misses += 1
        counts[1] += 1
        value = compute()
        nb = _nbytes(value)
        if self.used + nb > self.budget:
            self.overflow += 1
            return value
        self.used += nb
        self._memo[key] = value
        if mark:
            self.mark_static(*_tensors(value))
        return value

    def stats(self) -> dict:
        return dict(
            by_key={k: tuple(v) for k, v in self.by_key.items()},
            hits=self.hits,
            misses=self.misses,
            overflow=self.overflow,
            used_bytes=self.used,
            budget_bytes=self.budget,
        )


def _exporting() -> bool:
    """True while tracing / exporting / compiling (ONNX, TorchScript, ``torch.export``, ``torch.compile``).

    Identity-keyed memoisation is meaningless inside a trace, so upstream code runs instead.
    """
    try:
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            return True
    except Exception:  # noqa: BLE001
        pass
    for name in ("is_exporting", "is_compiling"):
        f = getattr(torch.compiler, name, None)
        try:
            if f is not None and f():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def active() -> RolloutCache | None:
    """The cache of the rollout currently executing in this context, if any (and only at inference)."""
    c = _ACTIVE.get()
    if c is None or torch.is_grad_enabled() or _exporting():
        return None
    return c


@contextlib.contextmanager
def rollout_scope(budget_bytes: int | None = None) -> Iterator[RolloutCache]:
    """Activate a fresh :class:`RolloutCache`; it is dropped (freeing its memory) on exit."""
    cache = RolloutCache(budget_bytes)
    token = _ACTIVE.set(cache)
    try:
        yield cache
    finally:
        _ACTIVE.reset(token)
        LAST_STATS.update(cache.stats())


#: Statistics of the most recently completed rollout (for diagnostics / benchmarks).
LAST_STATS: dict[str, int] = {}
