"""Drop-in CPU-oriented inference acceleration for OpenFold3.

Usage::

    import openfold3_accel   # activates automatically (set OPENFOLD3_ACCEL_DISABLE=1 to prevent)
    import openfold3         # (order does not matter; patches resolve lazily)

See :func:`status`, :func:`enable` and :func:`disable`.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from . import _compat, _patch, _selfcheck
from .patches import DEFAULT_GROUPS, group_defs

try:
    from importlib.metadata import version as _version

    __version__ = _version("openfold3-accel")
except Exception:  # noqa: BLE001
    __version__ = "0.0.0+unknown"

__all__ = [
    "enable",
    "disable",
    "status",
    "Status",
    "enabled",
    "disabled",
    "groups",
    "__version__",
]


def groups() -> dict[str, str]:
    """Available patch groups and a one-line description of each."""
    return {name: summary for name, (summary, _make) in group_defs().items()}


@dataclass(frozen=True)
class Status:
    """Snapshot returned by :func:`status`."""

    enabled: bool
    reason: str
    openfold3_version: str | None
    openfold3_supported: bool
    openfold3_supported_range: str
    torch_version: str
    groups: dict[str, str] = field(default_factory=dict)
    patches: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"openfold3_accel {__version__}: {'ENABLED' if self.enabled else 'disabled'} ({self.reason})",
            f"  openfold3: {self.openfold3_version} (supported {self.openfold3_supported_range}: "
            f"{self.openfold3_supported})",
            f"  torch: {self.torch_version}",
        ]
        lines += [f"  group {k}: {v}" for k, v in self.groups.items()]
        lines += [f"    patch {k}: {v}" for k, v in self.patches.items()]
        return "\n".join(lines)


def _selected(selection: Iterable[str] | None) -> list[str]:
    avail = list(group_defs())
    if selection is not None:
        names = list(selection)
    else:
        env = os.environ.get("OPENFOLD3_ACCEL_GROUPS", "")
        names = [g.strip() for g in env.split(",") if g.strip()] or list(DEFAULT_GROUPS)
        skip = {
            g.strip() for g in os.environ.get("OPENFOLD3_ACCEL_SKIP", "").split(",") if g.strip()
        }
        names = [g for g in names if g not in skip]
    unknown = [g for g in names if g not in avail]
    if unknown:
        raise ValueError(f"unknown patch group(s) {unknown}; available: {avail}")
    return names


def enable(groups: Iterable[str] | None = None) -> Status:
    """Apply the optimisations (idempotent). Unsupported setups are left untouched and reported.

    Parameters
    ----------
    groups
        Subset of :func:`groups` to enable. Default: all (honouring ``OPENFOLD3_ACCEL_GROUPS`` and
        ``OPENFOLD3_ACCEL_SKIP``).
    """
    _patch.apply(group_defs(), _selected(groups), _selfcheck.run)
    return status()


def disable(groups: Iterable[str] | None = None) -> Status:
    """Restore the original OpenFold3 callables (removing stacked wrappers in a safe order)."""
    for w in _patch.revert(groups):
        import warnings

        warnings.warn(w, RuntimeWarning, stacklevel=2)
    return status()


@contextlib.contextmanager
def disabled(groups: Iterable[str] | None = None) -> Iterator[None]:
    """Temporarily run upstream OpenFold3 code (e.g. for A/B comparisons), then re-enable."""
    before = [n for n, g in _patch.STATE.groups.items() if g.state == "active"]
    names = before if groups is None else [n for n in groups if n in before]
    disable(names)
    try:
        yield
    finally:
        enable(names)


@contextlib.contextmanager
def enabled(groups: Iterable[str] | None = None) -> Iterator[None]:
    """Temporarily enable (and afterwards restore) the optimisations."""
    before = {n for n, g in _patch.STATE.groups.items() if g.state == "active"}
    enable(groups)
    try:
        yield
    finally:
        now = [n for n, g in _patch.STATE.groups.items() if g.state == "active" and n not in before]
        disable(now)


def status() -> Status:
    import torch

    v = _compat.openfold3_version()
    ok, _ = _compat.version_supported(v)
    lo = ".".join(map(str, _compat.MIN_OPENFOLD3))
    hi = ".".join(map(str, _compat.MAX_OPENFOLD3_EXCLUSIVE))
    return Status(
        enabled=_patch.STATE.enabled,
        reason=_patch.STATE.reason,
        openfold3_version=v,
        openfold3_supported=ok,
        openfold3_supported_range=f">={lo},<{hi}",
        torch_version=torch.__version__,
        groups={k: g.state for k, g in _patch.STATE.groups.items()},
        patches={p.name: p.state for g in _patch.STATE.groups.values() for p in g.patches},
    )


if os.environ.get("OPENFOLD3_ACCEL_DISABLE", "") not in ("1", "true", "True", "yes"):
    enable()
