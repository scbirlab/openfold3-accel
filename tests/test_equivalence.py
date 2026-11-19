import torch

import openfold3_accel as acc


def test_exact_group_matches_upstream(tiny):
    model, batch, rollout = tiny
    acc.disable()
    ref = rollout()
    acc.enable(["diffusion_cache"])
    try:
        assert acc.status().groups["diffusion_cache"] == "active"
        out = rollout()
    finally:
        acc.disable()
    mask = batch["atom_mask"].bool()[0]
    assert torch.allclose(out[:, :, mask], ref[:, :, mask], atol=1e-5, rtol=1e-5)
    assert (ref - ref.mean()).abs().max() > 0.1  # non-vacuous


EXPECTED_CACHE_PATHS = (
    "cond_zij",
    "ln_z",
    "diff_bias",
    "adaln",
    "ctb_gate",
    "atom_reps",
    "geom",
    "xbias",
    "xq",
    "xk",
    "xout",
)


def test_every_cache_path_is_exercised(tiny):
    """Equivalence only means something if each cached path really served hits."""
    from openfold3_accel import _cache

    _model, _batch, rollout = tiny
    acc.enable(["diffusion_cache"])
    try:
        _cache.LAST_STATS.clear()
        rollout()
        by_key = _cache.LAST_STATS["by_key"]
    finally:
        acc.disable()
    for name in EXPECTED_CACHE_PATHS:
        assert name in by_key, f"cache path {name!r} never ran: {sorted(by_key)}"
        hits, misses = by_key[name]
        assert misses >= 1 and hits >= 1, f"{name}: hits={hits} misses={misses}"


def test_inference_mode_matches_upstream(tiny):
    """OpenFold3 prediction runs under Lightning's torch.inference_mode(); version counters are absent."""
    _model, batch, rollout = tiny
    acc.disable()
    with torch.inference_mode():
        ref = rollout()
    acc.enable(["diffusion_cache"])
    try:
        from openfold3_accel import _cache

        _cache.LAST_STATS.clear()
        with torch.inference_mode():
            out = rollout()
        assert _cache.LAST_STATS["hits"] > 0
    finally:
        acc.disable()
    mask = batch["atom_mask"].bool()[0]
    assert torch.allclose(out[:, :, mask], ref[:, :, mask], atol=1e-5, rtol=1e-5)


def test_selfcheck_passes_and_detects_breakage(monkeypatch):
    from openfold3_accel import _selfcheck

    acc.enable(["diffusion_cache"])
    try:
        assert _selfcheck.run("diffusion_cache") is None
        import openfold3.core.model.primitives.normalization as nz

        orig = nz.AdaLN.forward
        with (
            monkeypatch.context() as mp
        ):  # undo before disable(), or the patch stack is left re-wrapped
            mp.setattr(nz.AdaLN, "forward", _broken(orig))
            assert _selfcheck.run("diffusion_cache") is not None
    finally:
        acc.disable()


def _broken(orig):
    import functools

    @functools.wraps(orig)
    def forward(self, a, s):
        return orig(self, a, s) + 1e-2

    return forward


def test_revert_restores_originals(tiny):
    import openfold3.core.model.layers.diffusion_conditioning as dc

    before = dc.DiffusionConditioning.forward
    acc.enable(["diffusion_cache"])
    assert dc.DiffusionConditioning.forward is not before
    acc.disable()
    assert dc.DiffusionConditioning.forward is before


def test_context_managers(tiny):
    acc.disable()
    with acc.enabled(["diffusion_cache"]):
        assert acc.status().groups["diffusion_cache"] == "active"
        with acc.disabled():
            assert acc.status().groups["diffusion_cache"] != "active"
        assert acc.status().groups["diffusion_cache"] == "active"
    assert acc.status().groups["diffusion_cache"] != "active"


def test_grad_enabled_bypasses_cache(tiny):
    model, batch, rollout = tiny
    from openfold3_accel import _cache

    acc.enable(["diffusion_cache"])
    try:
        _cache.LAST_STATS.clear()
        with torch.enable_grad():
            rollout()
        assert not _cache.LAST_STATS  # upstream path: no scope was opened
    finally:
        acc.disable()


def test_state_dict_and_checkpoint_roundtrip(tiny, tmp_path):
    model, _, rollout = tiny
    keys = list(model.model.state_dict())
    acc.enable(["diffusion_cache"])
    try:
        assert list(model.model.state_dict()) == keys
        torch.save(model.model.state_dict(), tmp_path / "w.pt")
        sd = torch.load(tmp_path / "w.pt")
        model.model.load_state_dict(sd)
        rollout()
    finally:
        acc.disable()


def test_bf16_group_is_close(tiny):
    model, batch, rollout = tiny
    acc.disable()
    ref = rollout()
    acc.enable(["diffusion_cache", "bf16_diffusion"])
    try:
        out = rollout()
    finally:
        acc.disable()
    mask = batch["atom_mask"].bool()[0]
    rms = ((out - ref)[:, :, mask] ** 2).sum(-1).mean().sqrt()
    assert rms < 0.5 * ref.abs().mean()


def test_unknown_group_rejected():
    import pytest

    with pytest.raises(ValueError):
        acc.enable(["nope"])


def test_export_and_tracing_use_original_code(tiny, monkeypatch):
    """Inside a trace/export the caches step aside (upstream code runs)."""
    from openfold3_accel import _cache

    with _cache.rollout_scope():
        assert _cache.active() is not None
        monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
        assert _cache.active() is None


def test_fused_attention_close_to_upstream(tiny):
    model, batch, rollout = tiny
    acc.disable()
    ref = rollout()
    acc.enable(["diffusion_cache", "fused_attention"])
    try:
        assert acc.status().groups["fused_attention"] == "active"
        out = rollout()
    finally:
        acc.disable()
    mask = batch["atom_mask"].bool()[0]
    assert torch.allclose(out[:, :, mask], ref[:, :, mask], atol=2e-4, rtol=1e-4)
    # guard against a vacuous pass through the silent fallback: the fused kernel must really be used
    import torch.nn.functional as F

    calls = []
    real = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    acc.enable(["diffusion_cache", "fused_attention"])
    try:
        rollout()
    finally:
        F.scaled_dot_product_attention = real
        acc.disable()
    assert calls


def test_stacked_groups_revert_in_reverse_order(tiny):
    import openfold3.core.model.structure.diffusion_module as dm

    before = dm.SampleDiffusion._sample_rollout
    acc.enable(["diffusion_cache", "bf16_diffusion", "fused_attention"])
    assert dm.SampleDiffusion._sample_rollout is not before
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a "re-wrapped, left in place" warning would fail here
        acc.disable()
    assert dm.SampleDiffusion._sample_rollout is before


def test_bf16_weight_cast_cache_restores_requires_grad(tiny):
    model, batch, rollout = tiny
    inner = model.model.sample_diffusion
    before = [p.requires_grad for p in inner.parameters()]
    acc.enable(["diffusion_cache", "bf16_diffusion"])
    try:
        rollout()
    finally:
        acc.disable()
    assert [p.requires_grad for p in inner.parameters()] == before


def test_fast_init_same_distribution():
    import importlib

    import numpy as np

    init = importlib.import_module("openfold3.core.model.primitives.initialization")
    torch.manual_seed(0)
    w_ref = torch.empty(512, 256)
    acc.disable()
    init.trunc_normal_init_(w_ref)
    acc.enable(["fast_init"])
    try:
        assert acc.status().groups["fast_init"] == "active"
        w = torch.empty(512, 256)
        init.trunc_normal_init_(w)
    finally:
        acc.disable()
    assert abs(w.std().item() - w_ref.std().item()) < 0.02 * w_ref.std().item()
    assert w.abs().max().item() <= 2 * (1 / np.sqrt(256)) / 0.8796 * 1.0001  # truncation at +-2 std
    assert abs(w.mean().item()) < 0.01


# ---- lifecycle regression tests (review findings) ---------------------------------------------------------


def _wrapper_depth(fn) -> int:
    n = 0
    while hasattr(fn, "__wrapped__"):
        fn, n = fn.__wrapped__, n + 1
    return n


def test_selective_disable_every_transition():
    """Every enable/disable order over groups that stack on one attribute leaves a consistent, correct stack."""
    import itertools

    import openfold3.core.model.structure.diffusion_module as dm

    acc.disable()
    original = dm.SampleDiffusion._sample_rollout
    stackers = ("diffusion_cache", "bf16_diffusion")
    states = [set(c) for r in range(3) for c in itertools.combinations(stackers, r)]
    for order in itertools.permutations(stackers):
        for target in states:
            acc.disable()
            for g in order:  # enable in this order
                acc.enable([g])
            assert dm.SampleDiffusion._sample_rollout is not original
            acc.disable([g for g in stackers if g not in target])
            active = {n for n, s in acc.status().groups.items() if s == "active"} & set(stackers)
            assert active == target, (order, target, active)
            assert _wrapper_depth(dm.SampleDiffusion._sample_rollout) == len(target), (
                order,
                target,
            )
            if not target:
                assert dm.SampleDiffusion._sample_rollout is original
    acc.disable()
    assert dm.SampleDiffusion._sample_rollout is original


def test_selective_disable_behaviour(tiny):
    """After disabling only the inner group, its cache must really be off (no hits), bf16 still on."""
    from openfold3_accel import _cache

    _m, _b, rollout = tiny
    acc.enable(["diffusion_cache", "bf16_diffusion"])
    try:
        acc.disable(["diffusion_cache"])
        assert acc.status().groups["diffusion_cache"] == "inactive"
        assert acc.status().groups["bf16_diffusion"] == "active"
        _cache.LAST_STATS.clear()
        rollout()
        assert not _cache.LAST_STATS.get("by_key"), (
            "diffusion_cache still running after being disabled"
        )
    finally:
        acc.disable()


def _rng_fingerprint():
    import random

    import numpy as np

    return (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())


def test_enable_disable_preserve_global_rngs():
    import random

    import numpy as np

    def seeded():
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

    acc.disable()
    seeded()
    ref = _rng_fingerprint()
    seeded()
    acc.enable(["diffusion_cache"])  # runs the self-check, which builds random modules
    assert acc.status().groups["diffusion_cache"] == "active"
    acc.disable()
    with acc.enabled(["diffusion_cache"]):
        pass
    with acc.disabled():
        pass
    assert _rng_fingerprint() == ref


def test_bf16_rollouts_from_threads_are_safe(tiny):
    import threading

    model, _b, rollout = tiny
    inner = model.model.sample_diffusion
    before = [p.requires_grad for p in inner.parameters()]
    errors, outs = [], []

    def work():
        try:
            torch.set_grad_enabled(False)  # grad mode is thread-local
            outs.append(rollout())
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    acc.enable(["diffusion_cache", "bf16_diffusion"])
    try:
        ts = [threading.Thread(target=work) for _ in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    finally:
        acc.disable()
    assert not errors, errors
    assert [p.requires_grad for p in inner.parameters()] == before


import pytest  # noqa: E402


@pytest.mark.parametrize(
    ("n_token", "batch_size", "atom_holes", "cache_mb"),
    [
        (37, 1, 0.0, None),
        (50, 2, 0.0, None),
        (48, 1, 0.15, None),
        (41, 2, 0.3, None),
        (48, 1, 0.0, "1"),
    ],
    ids=["odd-n", "batch2", "atom-holes", "batch2-odd-holes", "tiny-cache-budget"],
)
def test_equivalence_across_shapes_masks_budgets(
    make_case, monkeypatch, n_token, batch_size, atom_holes, cache_mb
):
    if cache_mb:
        monkeypatch.setenv("OPENFOLD3_ACCEL_CACHE_MB", cache_mb)
    _model, batch, rollout = make_case(
        n_token=n_token, batch_size=batch_size, atom_holes=atom_holes
    )
    acc.disable()
    ref = rollout()
    acc.enable(["diffusion_cache"])
    try:
        out = rollout()
    finally:
        acc.disable()
    mask = batch["atom_mask"].bool()
    assert mask.any()
    for b in range(mask.shape[0]):
        assert torch.allclose(
            out[b][..., mask[b], :], ref[b][..., mask[b], :], atol=1e-5, rtol=1e-5
        )


def test_third_party_wrapper_makes_disable_atomic():
    import functools
    import warnings

    import openfold3.core.model.structure.diffusion_module as dm

    acc.disable()
    acc.enable(["diffusion_cache", "bf16_diffusion"])
    ours = dm.SampleDiffusion._sample_rollout
    inner = dm.DiffusionModule.forward  # patched by bf16_diffusion

    @functools.wraps(ours)
    def theirs(*a, **k):
        return ours(*a, **k)

    dm.SampleDiffusion._sample_rollout = theirs
    try:
        with pytest.warns(RuntimeWarning, match="re-wrapped"):
            acc.disable()
        # nothing was half-reverted: every group still reports active and every attribute is unchanged
        st = acc.status().groups
        assert st["diffusion_cache"] == st["bf16_diffusion"] == "active"
        assert dm.SampleDiffusion._sample_rollout is theirs
        assert dm.DiffusionModule.forward is inner
    finally:
        dm.SampleDiffusion._sample_rollout = ours  # the third party unwraps
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        acc.disable()
    assert acc.status().groups["diffusion_cache"] == "inactive"


def test_reapply_failure_is_reported_not_raised(monkeypatch):
    from openfold3_accel import _patch

    acc.disable()
    acc.enable(["diffusion_cache", "bf16_diffusion"])
    real = _patch._apply_group

    def boom(group, selfcheck):
        group.state = "skipped: injected failure"

    monkeypatch.setattr(_patch, "_apply_group", boom)
    acc.disable(["diffusion_cache"])  # survivors (bf16) must be re-applied; that step "fails"
    monkeypatch.setattr(_patch, "_apply_group", real)
    st = acc.status().groups
    assert st["diffusion_cache"] == "inactive"
    assert st["bf16_diffusion"].startswith("skipped")
    acc.disable()


def test_bf16_rollout_bypassed_off_cpu(tiny, monkeypatch):
    import openfold3_accel.patches.bf16 as bf

    model, _b, rollout = tiny
    inner = model.model.sample_diffusion
    seen = []
    monkeypatch.setattr(bf, "_on_cpu", lambda m: False)  # pretend the weights are on an accelerator
    real_autocast = torch.autocast
    monkeypatch.setattr(
        torch, "autocast", lambda *a, **k: (seen.append(a), real_autocast(*a, **k))[1]
    )
    acc.enable(["diffusion_cache", "bf16_diffusion"])
    try:
        before = [p.requires_grad for p in inner.parameters()]
        rollout()
        assert [p.requires_grad for p in inner.parameters()] == before
        assert not any(a and a[0] == "cpu" for a in seen), (
            "CPU autocast entered for non-CPU weights"
        )
    finally:
        acc.disable()
