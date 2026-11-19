"""Benchmark worker: times one OpenFold3 diffusion rollout in a given mode and returns JSON.

Runs on a *randomly initialised* model with randomised zero-init layers and synthetic features, so it
needs no weights, no network and no input files. It reads nothing from and writes nothing to anywhere
except stdout / the ``--out`` JSON file.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
import warnings

MODES = (
    "original",
    "exact",
    "exact+fused",
    "exact+bf16",
    "exact+fused+bf16",
    "init:original",
    "init:fast_init",
)
GROUPS = {
    "original": [],
    "exact": ["diffusion_cache"],
    "exact+fused": ["diffusion_cache", "fused_attention"],
    "exact+bf16": ["diffusion_cache", "bf16_diffusion"],
    "exact+fused+bf16": ["diffusion_cache", "fused_attention", "bf16_diffusion"],
    "init:original": [],
    "init:fast_init": ["fast_init"],
}


def _build(n_token: int, steps: int, blocks: int, device: str):
    import logging

    import torch

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from openfold3.tests.utils.data_utils import random_of3_features

    torch.set_grad_enabled(False)
    cfg = OF3ProjectEntry().get_model_config_with_presets()
    cfg.settings.memory.eval.use_triton_triangle_kernels = False
    cfg.settings.memory.eval.tune_chunk_size = False
    cfg.architecture.pairformer.no_blocks = 2
    cfg.architecture.msa.msa_module.no_blocks = 1
    cfg.architecture.diffusion_module.diffusion_transformer.no_blocks = blocks
    cfg.architecture.shared.diffusion.no_full_rollout_steps = steps
    cfg.architecture.shared.num_recycles = 0
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = OpenFold3AllAtom(cfg).eval()
    for p in model.model.parameters():
        if p.abs().max() == 0:
            p.data.normal_(0, 0.05)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    batch = random_of3_features(batch_size=1, n_token=n_token, n_msa=8, n_templ=1, is_eval=True)
    model = model.to(device)
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    sd = model.model.sample_diffusion
    orig, cap = sd.forward, {}

    def grab(*a, **k):
        cap["a"], cap["k"] = a, k
        return orig(*a, **k)

    sd.forward = grab
    model.model(batch)
    sd.forward = orig
    return sd, cap, batch


def _run_init(mode, groups, repeats):
    """Time construction of the full-size model (weights are drawn, then normally overwritten by a checkpoint)."""
    import logging

    import torch

    import openfold3_accel as acc

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

    if groups:
        st = acc.enable(groups)
        bad = {g: s for g, s in st.groups.items() if g in groups and s != "active"}
        if bad:
            return {"mode": mode, "device": "cpu-init", "error": f"patches not active: {bad}"}
    cfg = OF3ProjectEntry().get_model_config_with_presets()
    times = []
    for _ in range(max(1, min(repeats, 2))):
        t = time.perf_counter()
        m = OpenFold3AllAtom(cfg)
        times.append(time.perf_counter() - t)
    n = sum(p.numel() for p in m.parameters())
    return {
        "mode": mode.replace("init:", "init "),
        "device": "cpu-init",
        "seconds_median": statistics.median(times),
        "seconds_all": times,
        "coords": [],
        "params_millions": n / 1e6,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def run(
    mode: str, device: str, n_token: int, steps: int, blocks: int, repeats: int, threads: int | None
):
    import torch

    if threads:
        torch.set_num_threads(threads)
    import openfold3_accel as acc

    acc.disable()
    groups = GROUPS[mode]
    if mode.startswith("init:"):
        return _run_init(mode, groups, repeats)
    sd, cap, batch = _build(n_token, steps, blocks, device)
    if groups:
        st = acc.enable(groups)
        bad = {g: s for g, s in st.groups.items() if g in groups and s != "active"}
        if bad:
            return {"mode": mode, "device": device, "error": f"patches not active: {bad}"}

    def once():
        torch.manual_seed(1)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = sd(*cap["a"], **cap["k"])
        if device == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t

    out, _ = once()  # warm-up
    times = [once()[1] for _ in range(repeats)]
    mask = batch["atom_mask"].bool()[0]
    return {
        "mode": mode,
        "device": device,
        "n_token": n_token,
        "steps": steps,
        "blocks": blocks,
        "seconds_median": statistics.median(times),
        "seconds_all": times,
        "coords": out[:, :, mask].double().cpu().flatten().tolist()[:200000],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-token", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ["OPENFOLD3_ACCEL_DISABLE"] = "1"
    res = run(a.mode, a.device, a.n_token, a.steps, a.blocks, a.repeats, a.threads)
    with open(a.out, "w") as f:
        json.dump(res, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
