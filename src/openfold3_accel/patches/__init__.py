"""Patch groups. Each module exposes ``GROUP``, ``SUMMARY`` and ``targets()``."""

from __future__ import annotations

from collections.abc import Callable

from .._patch import Patch


def group_defs() -> dict[str, tuple[str, Callable[[], list[Patch]]]]:
    from . import attention, bf16, diffusion, loading

    return {m.GROUP: (m.SUMMARY, m.targets) for m in (diffusion, bf16, attention, loading)}


#: groups that are on by default (the rest are opt-in because they change numerics)
DEFAULT_GROUPS = ("diffusion_cache",)
