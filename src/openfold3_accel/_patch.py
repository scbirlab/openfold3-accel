"""Reversible, idempotent, group-wise patching of OpenFold3 internals.

Design
------
* A *patch* replaces one attribute (a method on a class, or a function in a module) with a
  replacement built from the original by a *factory*.
* Patches are organised in *groups* (``diffusion_cache``, ``attention``, ...). A group is applied
  atomically: every target of the group must match a verified source fingerprint, otherwise the
  whole group is left untouched and the reason is reported by :func:`openfold3_accel.status`.
* After applying a group, a numerical self-check compares the replacement with the original on
  tiny random modules. A failing check rolls the group back.
* :func:`revert` restores the *exact original function objects* (unless somebody else has wrapped
  them since, in which case it warns and leaves them).

Nothing here touches model parameters, buffers, ``state_dict`` layout, RNG state or thread counts.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from . import _compat

_LOCK = threading.RLock()


@dataclass
class Patch:
    """One replaced attribute.

    ``spec`` is ``(module, class_or_None, attribute)``; it is also what
    ``scripts/fingerprint_versions.py`` uses to fingerprint upstream sources without importing them.
    """

    name: str
    group: str
    spec: tuple[str, str | None, str]
    factory: Callable[[Callable], Callable]
    state: str = "inactive"
    original: Callable | None = None
    replacement: Callable | None = None
    owner: Any = None
    attr: str | None = None

    def resolve(self) -> tuple[Any, str]:
        import importlib

        module, cls, attr = self.spec
        owner = importlib.import_module(module)
        return (getattr(owner, cls) if cls else owner), attr


@dataclass
class GroupState:
    name: str
    summary: str
    state: str = "inactive"  # inactive | active | skipped: <why>
    patches: list[Patch] = field(default_factory=list)
    seq: int = (
        -1
    )  # application order; wrappers stacked on one attribute must be removed in reverse order


_SEQ = itertools.count()


@dataclass
class _State:
    reason: str = "not yet enabled"
    groups: dict[str, GroupState] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return any(g.state == "active" for g in self.groups.values())


STATE = _State()


def _rollback(patches: Iterable[Patch]) -> None:
    for p in patches:
        if p.state == "active" and p.owner is not None and p.original is not None:
            if p.owner.__dict__.get(p.attr) is p.replacement:
                setattr(p.owner, p.attr, p.original)
            p.state = "inactive"


def _apply_group(group: GroupState, selfcheck: Callable[[str], str | None] | None) -> None:
    patches = group.patches
    try:
        for p in patches:
            owner, attr = p.resolve()
            p.owner, p.attr = owner, attr
            fn = owner.__dict__.get(attr)
            if fn is None:
                group.state = f"skipped: {p.name} is not defined where expected"
                return
            fp = _compat.fingerprint(fn)
            if fp is None:
                group.state = f"skipped: source of {p.name} unavailable, cannot verify"
                return
            if fp not in _compat.FINGERPRINTS.get(p.name, frozenset()):
                group.state = f"skipped: unrecognised source fingerprint {fp} for {p.name}"
                return
            p.original = fn
        # every target verified: now (and only now) install the replacements
        for p in patches:
            p.replacement = p.factory(p.original)
            setattr(p.owner, p.attr, p.replacement)
            p.state = "active"
    except Exception as exc:  # noqa: BLE001 - never break the user's import
        _rollback(patches)
        group.state = f"skipped: patching failed: {exc!r}"
        return
    problem = None
    if selfcheck is not None:
        try:
            problem = selfcheck(group.name)
        except Exception as exc:  # noqa: BLE001
            problem = f"self-check raised {exc!r}"
    if problem:
        _rollback(patches)
        group.state = f"skipped: self-check failed: {problem}"
        return
    group.state = "active"
    group.seq = next(_SEQ)


def apply(
    group_defs: dict[str, tuple[str, Callable[[], list[Patch]]]],
    selected: Iterable[str],
    selfcheck: Callable[[str], str | None] | None,
) -> None:
    """Apply the selected groups (idempotent per group)."""
    with _LOCK:
        v = _compat.openfold3_version()
        ok, why = _compat.version_supported(v)
        if not ok:
            STATE.reason = why
            return
        for name in selected:
            summary, make = group_defs[name]
            g = STATE.groups.get(name)
            if g is not None and g.state == "active":
                continue
            g = GroupState(name=name, summary=summary, patches=make())
            STATE.groups[name] = g
            _apply_group(g, selfcheck)
        n_active = sum(1 for g in STATE.groups.values() if g.state == "active")
        STATE.reason = "ok" if n_active else "no patch group matched a verified fingerprint"


def revert(groups: Iterable[str] | None = None) -> list[str]:
    """Restore original callables; returns warnings for patches that others re-wrapped.

    Groups may wrap the *same* attribute (``bf16_diffusion`` wraps ``diffusion_cache``'s rollout). A wrapper
    can only be removed from the top, so removing a group unwinds every active group applied at or after it
    (last in, first out) and then re-applies the groups that were meant to stay, in their original order.
    A patch that cannot be removed (someone else wrapped it) is reported and its group stays ``active``.
    """
    with _LOCK:
        warnings: list[str] = []
        active = sorted(
            (g for g in STATE.groups.values() if g.state == "active"), key=lambda g: g.seq
        )
        wanted = {g.name for g in active} if groups is None else set(groups)
        remove = [g for g in active if g.name in wanted]
        if not remove:
            return warnings
        first = min(g.seq for g in remove)
        unwound = [g for g in reversed(active) if g.seq >= first]
        # Dry run first so the operation is atomic: if anything we installed has been wrapped by third-party
        # code, change nothing (a half-reverted group would be skipped by a later enable() forever).
        cur = {}
        conflicts = []
        for g in unwound:
            for p in reversed(g.patches):
                if p.state != "active":
                    continue
                key = (id(p.owner), p.attr)
                now = cur.get(key, p.owner.__dict__.get(p.attr))
                if now is p.replacement:
                    cur[key] = p.original
                else:
                    conflicts.append(f"{p.name} was re-wrapped by other code; nothing was reverted")
        if conflicts:
            return conflicts
        for g in unwound:
            for p in reversed(g.patches):
                if p.state != "active":
                    continue
                if p.owner.__dict__.get(p.attr) is p.replacement:
                    setattr(p.owner, p.attr, p.original)
                    p.state = "inactive"
                else:
                    warnings.append(f"{p.name} was re-wrapped by other code; left in place")
            if all(p.state != "active" for p in g.patches):
                g.state = "inactive"
        if (
            not warnings
        ):  # re-apply survivors (their self-check already passed; fingerprints are re-verified)
            for g in reversed(unwound):
                if g.name not in wanted:
                    _apply_group(g, None)
        STATE.reason = "disabled" if not STATE.enabled else "ok"
        return warnings
