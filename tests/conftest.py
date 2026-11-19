"""Shared fixtures: a tiny OpenFold3 model with *non-trivial* weights.

OpenFold3 zero-initialises many output layers, so a freshly built model makes the denoiser output
almost constant and any equivalence test vacuous. We therefore randomise every all-zero parameter.
"""

from __future__ import annotations

import os
import warnings

import pytest

os.environ.setdefault("OPENFOLD3_ACCEL_DISABLE", "1")  # tests control patching explicitly
warnings.filterwarnings("ignore")

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3")


@pytest.fixture(scope="session")
def tiny():
    import logging

    logging.disable(logging.CRITICAL)
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from openfold3.tests.utils.data_utils import random_of3_features

    torch.set_grad_enabled(False)
    cfg = OF3ProjectEntry().get_model_config_with_presets()
    cfg.settings.memory.eval.use_triton_triangle_kernels = False
    cfg.settings.memory.eval.tune_chunk_size = False
    cfg.architecture.pairformer.no_blocks = 2
    cfg.architecture.msa.msa_module.no_blocks = 1
    cfg.architecture.diffusion_module.diffusion_transformer.no_blocks = 3
    cfg.architecture.shared.diffusion.no_full_rollout_steps = 4
    cfg.architecture.shared.num_recycles = 0
    torch.manual_seed(0)
    model = OpenFold3AllAtom(cfg).eval()
    for p in model.model.parameters():
        if p.abs().max() == 0:
            p.data.normal_(0, 0.05)

    def make_case(n_token=48, batch_size=1, n_msa=8, atom_holes=0.0, seed=0):
        """Capture the diffusion rollout inputs for a batch of the given shape (optionally with masked atoms)."""
        torch.manual_seed(seed)
        batch = random_of3_features(
            batch_size=batch_size, n_token=n_token, n_msa=n_msa, n_templ=1, is_eval=True
        )
        if atom_holes:  # unresolved atoms inside valid tokens, plus a masked tail of the atom array
            g = torch.Generator().manual_seed(seed + 7)
            am = batch["atom_mask"].clone()
            am = am * (torch.rand(am.shape, generator=g) > atom_holes).to(am.dtype)
            am[..., -11:] = 0
            batch["atom_mask"] = am
        sd = model.model.sample_diffusion
        orig, cap = sd.forward, {}

        def grab(*a, **k):
            cap["a"], cap["k"] = a, k
            return orig(*a, **k)

        sd.forward = grab
        try:
            model.model(batch)
        finally:
            sd.forward = orig

        def rollout(seed: int = 1):
            torch.manual_seed(seed)
            return sd(*cap["a"], **cap["k"])

        return model, batch, rollout

    model_, batch_, rollout_ = make_case()
    tiny_model_cases["make"] = make_case
    return model_, batch_, rollout_


tiny_model_cases: dict = {}


@pytest.fixture(scope="session")
def make_case(tiny):
    """Factory for rollouts on other batch shapes / padding patterns (reuses the tiny model)."""
    return tiny_model_cases["make"]
