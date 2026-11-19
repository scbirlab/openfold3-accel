"""Version gating and source fingerprints for patch targets.

OpenFold3's internals are private and change between releases. Instead of trusting a version range
alone, every patch is applied only if

* the installed ``openfold3`` is inside the tested range, **and**
* the whitespace-normalised source of the exact function being replaced matches a fingerprint that
  the replacement has been verified against.

Anything else is left untouched (the original OpenFold3 code runs) and is reported by
:func:`openfold3_accel.status`. Regenerate the table for new releases with
``python scripts/fingerprint_versions.py`` after reading the upstream diff of each target.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from importlib import metadata

#: Inclusive lower / exclusive upper bound of OpenFold3 releases this package is tested with.
MIN_OPENFOLD3 = (0, 5, 0)
MAX_OPENFOLD3_EXCLUSIVE = (0, 6, 0)

#: Known-good source fingerprints per patch target (see ``scripts/fingerprint_versions.py``).
#: ``dev`` fingerprints are those of the commit pinned in ``OPENFOLD3_COMMIT``.
FINGERPRINTS: dict[str, frozenset[str]] = {  # openfold3 0.5.0 and 0.5.1 (identical)
    "SampleDiffusion._sample_rollout": frozenset({"8bd6b31fdcda"}),
    "DiffusionConditioning.forward": frozenset({"f087b75e4de9"}),
    "DiffusionTransformer.forward": frozenset({"595d4303fea0"}),
    "DiffusionAttentionPairBias._prep_bias": frozenset({"bdc267d1d3cf"}),
    "AdaLN.forward": frozenset({"7757a8da48bc"}),
    "ConditionedTransitionBlock._transition": frozenset({"ad932f303448"}),
    "AtomAttentionEncoder.get_atom_reps": frozenset({"6fea8030df81"}),
    "CrossAttentionPairBias.forward": frozenset({"0f7d5d1d435d"}),
    "SampleDiffusion._sample_rollout[bf16]": frozenset({"8bd6b31fdcda"}),
    "DiffusionModule.forward": frozenset({"8f7737923ccd"}),
    "attention._attention": frozenset({"bfffc382fcae"}),
    "initialization.trunc_normal_init_": frozenset({"a7a22ae83444"}),
}


def fingerprint(fn) -> str | None:
    """Whitespace-insensitive SHA-256 prefix of ``fn``'s source, or ``None`` if unavailable."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    return fingerprint_source(src)


def fingerprint_source(src: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", src).encode()).hexdigest()[:12]


def parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for tok in re.split(r"[.+\-]", v):
        m = re.match(r"\d+", tok)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts)


def openfold3_version() -> str | None:
    try:
        return metadata.version("openfold3")
    except metadata.PackageNotFoundError:
        return None


def version_supported(v: str | None) -> tuple[bool, str]:
    if v is None:
        return False, "openfold3 is not installed"
    t = parse_version(v)
    if not t:
        return False, f"unparseable openfold3 version {v!r}"
    if t < MIN_OPENFOLD3 or t >= MAX_OPENFOLD3_EXCLUSIVE:
        lo = ".".join(map(str, MIN_OPENFOLD3))
        hi = ".".join(map(str, MAX_OPENFOLD3_EXCLUSIVE))
        return False, f"openfold3 {v} is outside the tested range [{lo}, {hi})"
    return True, "ok"
