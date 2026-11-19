"""Numerical smoke test run each time a patch group is enabled.

It builds tiny random modules from the *installed* OpenFold3, runs them inside a rollout scope with the
conditioning tensors registered static, and compares cached against upstream output. A failure rolls the
group back. The scope of this check is deliberately small (the cheap row-wise layers, plus the cache
mechanics, called twice so that the cached path really is exercised); full-path equivalence is covered by
``tests/test_equivalence.py``.
"""

from __future__ import annotations


def _diffusion_cache() -> str | None:
    import importlib

    import torch

    from . import _cache

    norm = importlib.import_module("openfold3.core.model.primitives.normalization")
    trans = importlib.import_module("openfold3.core.model.layers.transition")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        ada = norm.AdaLN(c_a=8, c_s=8).eval()
        ctb = trans.ConditionedTransitionBlock(c_a=8, c_s=8, n=2).eval()
        for p in list(ada.parameters()) + list(ctb.parameters()):
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        a = torch.randn(1, 5, 8, generator=g)
        s = torch.randn(1, 5, 8, generator=g)
        mask = torch.ones(1, 5, 1)
        ref_a = ada.forward.__wrapped__(ada, a, s) if hasattr(ada.forward, "__wrapped__") else None
        ref_c = (
            ctb._transition.__wrapped__(ctb, a, s, mask)
            if hasattr(ctb._transition, "__wrapped__")
            else None
        )
        if ref_a is None or ref_c is None:
            return "patched callables lack __wrapped__ (cannot obtain upstream reference)"
        with _cache.rollout_scope() as ctx:
            ctx.mark_static(s)
            for _ in range(2):  # second pass reads the cache
                out_a = ada(a, s)
                out_c = ctb._transition(a, s, mask)
            hits = ctx.hits
        if hits < 2:
            return f"cache mechanics not exercised (hits={hits})"
        if not torch.allclose(out_a, ref_a, atol=1e-6, rtol=1e-5):
            return "AdaLN cached output differs from upstream"
        if not torch.allclose(out_c, ref_c, atol=1e-6, rtol=1e-5):
            return "ConditionedTransitionBlock cached output differs from upstream"
        with torch.inference_mode():  # Lightning's Trainer.predict path
            ai, si = a.clone(), s.clone()
            with _cache.rollout_scope() as ctx:
                ctx.mark_static(si)
                ada(ai, si)
                ada(ai, si)
    return None


def run(group: str) -> str | None:
    """Return ``None`` if the group passes, else a short reason (the group is then rolled back).

    The check builds random modules, which draws from the torch, NumPy (scipy's truncated normal) and Python
    global RNGs; all three are saved and restored so enabling the package never perturbs a seeded process.
    """
    if group != "diffusion_cache":
        return None
    import contextlib
    import random

    import torch

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            torch.random.fork_rng(devices=[])
        )  # CPU generator only; the check is CPU-only
        py_state = random.getstate()
        stack.callback(random.setstate, py_state)
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            np = None
        if np is not None:
            np_state = np.random.get_state()
            stack.callback(np.random.set_state, np_state)
        try:
            return _diffusion_cache()
        except Exception as e:  # noqa: BLE001
            return f"self-check raised {type(e).__name__}: {e}"
