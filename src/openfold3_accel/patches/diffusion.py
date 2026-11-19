"""Patch group ``diffusion_cache``: compute step-invariant diffusion-denoiser terms once per rollout.

What is cached (all inside one ``SampleDiffusion._sample_rollout`` call, inference only):

======================================  ===================================================
upstream recomputation (every step)     replacement
======================================  ===================================================
``LayerNorm(z)`` in every               computed once; the result is registered as static
``DiffusionTransformer``
per-block pair bias ``Linear(z)``       computed once per block (24 token blocks + 6 atom
+ mask bias                             blocks)
reference-conformer atom features,      computed once; only ``q = c + Linear(r_noisy)`` is
trunk-single/pair injection, atom-pair  computed per step
MLP (``AtomAttentionEncoder``)
``s``-dependent half of every atom-     ``(gate, shift)`` computed once per layer
level ``AdaLN`` and transition gate
key-side blocking of atom features      project + normalise each atom *once*, then gather
(each atom appears in 4 key blocks)     into key blocks (row-wise ops commute with gather)
======================================  ===================================================

Everything uses the same operations on the same inputs as upstream; the only numerical difference
is the order of floating-point work inside GEMMs of different row counts (~1e-7 relative) and
values at *padded* (``atom_mask == 0``) atoms, which never influence valid atoms.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import math

import torch
import torch.nn.functional as F

from .. import _cache
from .._patch import Patch

GROUP = "diffusion_cache"
SUMMARY = "cache step-invariant diffusion denoiser terms once per rollout (exact)"


def _mod(path: str):
    return importlib.import_module(path)


def _attention_fn():
    # looked up at call time so that the ``attention`` group (if active) is honoured
    return _mod("openfold3.core.model.primitives.attention")._attention


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _adaln_terms(adaln, s):
    """The ``s``-only half of ``AdaLN.forward`` (identical operations)."""
    s = adaln.layer_norm_s(s)
    g = adaln.sigmoid(adaln.linear_g(s))
    return g, adaln.linear_s(s)


def _geometry(mask, n_query: int, n_key: int):
    """Key-block gather indices, invalid mask and pair mask for a *static, un-batched* atom mask.

    Returns ``None`` if the mask carries real batch dimensions (then upstream code is used).
    """
    if mask.numel() != mask.shape[-1]:
        return None
    blocks = _mod("openfold3.core.utils.atom_attention_block_utils")
    n_atom = mask.shape[-1]
    m1 = mask.reshape(1, n_atom)
    key_idx, invalid = blocks.get_block_indices(
        atom_mask=m1, n_query=n_query, n_key=n_key, device=mask.device
    )
    num_blocks = math.ceil(n_atom / n_query)
    pad = blocks.get_query_block_padding(n_atom=n_atom, n_query=n_query)
    pair = blocks.get_pair_atom_block_mask(
        atom_mask=m1,
        num_blocks=num_blocks,
        n_query=n_query,
        n_key=n_key,
        pad_len_right_q=pad,
        key_block_idxs=key_idx,
        invalid_mask=invalid,
    )
    return key_idx.reshape(-1).long(), invalid.reshape(-1), pair, pad, num_blocks


def _gather_keys(x, key_idx, batch_dims, num_blocks: int, n_key: int):
    """``x[*, N, C] -> [*, blocks, n_key, C]`` using shared (un-batched) indices."""
    n_atom, c = x.shape[-2:]
    flat = x.reshape(-1, n_atom, c).index_select(1, key_idx)
    return flat.reshape((*batch_dims, num_blocks, n_key, c))


# ----------------------------------------------------------------------------------------------
# replacement factories (each receives the *original* callable)
# ----------------------------------------------------------------------------------------------
def _rollout(orig):
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def _sample_rollout(self, *args, **kwargs):
        if torch.is_grad_enabled() or self.training:
            return orig(self, *args, **kwargs)
        bound = sig.bind(self, *args, **kwargs).arguments
        batch = bound.get("batch")
        with _cache.rollout_scope() as ctx:
            if isinstance(batch, dict):
                ctx.mark_static(
                    *(
                        batch.get(k)
                        for k in (
                            "atom_mask",
                            "token_mask",
                            "ref_pos",
                            "ref_mask",
                            "ref_element",
                            "ref_charge",
                            "ref_atom_name_chars",
                            "ref_space_uid",
                            "num_atoms_per_token",
                            "atom_to_token_index",
                        )
                    )
                )
            ctx.mark_static(
                bound.get("atom_mask"),
                bound.get("si_input"),
                bound.get("si_trunk"),
                bound.get("zij_trunk"),
            )
            return orig(self, *args, **kwargs)

    return _sample_rollout


def _conditioning_forward(orig):
    """Pair conditioning (relpos concat, LN, Linear, transitions over N x N) is step-invariant."""

    @functools.wraps(orig)
    def forward(self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size=None):
        ctx = _cache.active()
        if (
            ctx is None
            or not ctx.is_static(zij_trunk)
            or not ctx.is_static(batch.get("token_mask"))
        ):
            return orig(self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size)
        key = ("cond_zij", id(self), id(zij_trunk), id(batch), bool(use_conditioning))
        box: dict = {}

        def compute():
            si, zij = orig(
                self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size
            )
            box["si"] = si
            return zij

        zij = ctx.memo(key, compute)
        if "si" in box:  # first step: upstream output as computed
            return box["si"], zij
        token_mask = batch["token_mask"]
        if not use_conditioning:
            si_trunk = si_trunk * 0
        # identical single-path arithmetic to ``_embed_trunk_inputs`` / ``_forward``
        si = torch.cat([si_trunk, si_input], dim=-1)
        si = self.linear_s(self.layer_norm_s(si))
        n = 0.25 * torch.log(t / self.sigma_data)
        n = self.fourier_emb(n.unsqueeze(-1))
        si = si + self.linear_n(self.layer_norm_n(n)).unsqueeze(-2)
        for lyr in self.transition_s:
            si = si + lyr(si, mask=token_mask, chunk_size=chunk_size)
        return si, zij

    return forward


def _diffusion_transformer_forward(orig):
    @functools.wraps(orig)
    def forward(
        self,
        a,
        s,
        z,
        mask=None,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        use_triton_triangle_kernels=False,
        use_lma=False,
        use_high_precision_attention=False,
        _mask_trans=True,
    ):
        ctx = _cache.active()
        if ctx is None or not ctx.is_static(z):
            return orig(
                self,
                a,
                s,
                z,
                mask,
                use_deepspeed_evo_attention,
                use_cueq_triangle_kernels,
                use_triton_triangle_kernels,
                use_lma,
                use_high_precision_attention,
                _mask_trans,
            )
        ckpt = _mod("openfold3.core.utils.checkpointing")
        # Single layer norm for atom attention enc/dec diffusion transformer (as upstream), once.
        z = ctx.memo(("ln_z", id(self), id(z)), lambda: self.layer_norm_z(z))
        blocks = [
            functools.partial(
                b,
                s=s,
                z=z,
                mask=mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                use_high_precision_attention=use_high_precision_attention,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]
        (a,) = ckpt.checkpoint_blocks(
            blocks, args=(a,), blocks_per_ckpt=None, use_reentrant=self.use_reentrant
        )
        return a

    return forward


def _diffusion_prep_bias(orig):
    @functools.wraps(orig)
    def _prep_bias(self, a, z, mask):
        ctx = _cache.active()
        if ctx is None or not ctx.is_static(z) or (mask is not None and not ctx.is_static(mask)):
            return orig(self, a, z, mask)
        pz = _mod("openfold3.core.utils.tensor_utils").permute_final_dims

        def compute():
            # one combined bias (mask + pair), kept un-expanded over samples: broadcasting at use time
            # gives identical values and the memory of the pair bias alone
            m = mask if mask is not None else a.new_ones(a.shape[-2:-1])
            mask_bias = (self.inf * (m - 1))[..., None, None, :]
            return mask_bias + pz(self.linear_z(z), [2, 0, 1])

        combined = ctx.memo(("diff_bias", id(self), id(z), id(mask)), compute)
        return [combined]

    return _prep_bias


def _adaln_forward(orig):
    @functools.wraps(orig)
    def forward(self, a, s):
        ctx = _cache.active()
        if ctx is None or not ctx.is_static(s):
            return orig(self, a, s)
        g, shift = ctx.memo(("adaln", id(self), id(s)), lambda: _adaln_terms(self, s))
        a = self.layer_norm_a(a)
        return g * a + shift

    return forward


def _cond_transition(orig):
    @functools.wraps(orig)
    def _transition(self, a, s, mask):
        ctx = _cache.active()
        if ctx is None or not ctx.is_static(s):
            return orig(self, a, s, mask)
        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        gate = ctx.memo(("ctb_gate", id(self), id(s)), lambda: self.sigmoid(self.linear_g(s)))
        a = gate * self.linear_out(b)
        a = a * mask
        return a

    return _transition


def _atom_reps(orig):
    @functools.wraps(orig)
    def get_atom_reps(self, batch, rl=None, si_trunk=None, zij_trunk=None):
        ctx = _cache.active()
        if (
            ctx is None
            or rl is None
            or not (ctx.is_static(si_trunk) and ctx.is_static(zij_trunk))
            or not ctx.is_static(batch.get("ref_pos"))
        ):
            return orig(self, batch, rl, si_trunk, zij_trunk)
        box: dict = {}

        def compute():
            ql, cl, plm = orig(self, batch, rl, si_trunk, zij_trunk)
            box["ql"] = ql
            return cl, plm

        cl, plm = ctx.memo(("atom_reps", id(self), id(batch)), compute)
        if "ql" in box:  # first step (or cache overflow): upstream result as computed
            return box["ql"], cl, plm
        # Same arithmetic as ``NoisyPositionEmbedder.forward``: ql = cl + linear_r(rl)
        return cl + self.noisy_position_embedder.linear_r(rl), cl, plm

    return get_atom_reps


def _cross_attention_pair_bias(orig):
    def fast(self, a, z, s, mask, use_high_precision_attention, ctx):
        geom = ctx.memo(
            ("geom", id(mask), self.n_query, self.n_key),
            lambda: _geometry(mask, self.n_query, self.n_key),
            mark=False,
        )
        if geom is None:
            return None
        key_idx, _invalid, pair_mask, pad, num_blocks = geom
        batch_dims = tuple(a.shape[:-2])
        n_atom, n_dim = a.shape[-2:]
        nq, nk = self.n_query, self.n_key
        ptd = _mod("openfold3.core.utils.tensor_utils").permute_final_dims
        mha = self.mha

        # -- static: biases ----------------------------------------------------------------
        def bias_terms():
            mask_bias = (self.inf * (pair_mask - 1))[..., None, :, :]
            return mask_bias, ptd(self.linear_z(z), [2, 0, 1])

        def bias_terms_combined():
            mask_bias, zb = bias_terms()
            return mask_bias + zb

        # single combined bias; broadcast over samples at use time (same values, same memory as zb)
        biases = [ctx.memo(("xbias", id(self), id(z), id(mask)), bias_terms_combined)]

        # -- static: adaptive-LN terms for queries (blocked) and keys (un-blocked) --------------
        def q_terms():
            sp = F.pad(s, (0, 0, 0, pad), value=0.0)
            sp = sp.reshape((*s.shape[:-2], num_blocks, nq, s.shape[-1]))
            return _adaln_terms(self.layer_norm_a_q, sp)

        gq, shq = ctx.memo(("xq", id(self), id(s)), q_terms)
        gk, shk = ctx.memo(("xk", id(self), id(s)), lambda: _adaln_terms(self.layer_norm_a_k, s))
        out_gate = ctx.memo(("xout", id(self), id(s)), lambda: self.sigmoid(self.linear_ada_out(s)))

        # -- dynamic -------------------------------------------------------------------------
        ap = F.pad(a, (0, 0, 0, pad), value=0.0).reshape((*batch_dims, num_blocks, nq, n_dim))
        a_q = gq * self.layer_norm_a_q.layer_norm_a(ap) + shq

        # keys: LN / adaptive modulation / K,V projections are row-wise, so run them once per
        # atom and gather afterwards (upstream gathers first and does 4x the work).
        a_k = gk * self.layer_norm_a_k.layer_norm_a(a) + shk
        k = _gather_keys(mha.linear_k(a_k), key_idx, batch_dims, num_blocks, nk)
        v = _gather_keys(mha.linear_v(a_k), key_idx, batch_dims, num_blocks, nk)

        q = mha.linear_q(a_q)
        q = q.view(q.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
        k = k.view(k.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
        v = v.view(v.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
        q /= math.sqrt(mha.c_hidden)

        o = _attention_fn()(q, k, v, biases, use_high_precision=use_high_precision_attention)
        o = o.transpose(-2, -3)
        o = mha._wrap_up(o, a_q)

        o = o.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]
        return out_gate * o

    @functools.wraps(orig)
    def forward(self, a, z, s, mask=None, use_high_precision_attention=False):
        ctx = _cache.active()
        if (
            ctx is not None
            and mask is not None
            and ctx.is_static(z)
            and ctx.is_static(s)
            and ctx.is_static(mask)
            and s.shape[:-2] == mask.shape[:-1]
        ):
            out = fast(self, a, z, s, mask, use_high_precision_attention, ctx)
            if out is not None:
                return out
        return orig(self, a, z, s, mask, use_high_precision_attention)

    return forward


# ----------------------------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------------------------
def targets() -> list[Patch]:
    ds = "openfold3.core.model.structure.diffusion_module"
    ap = "openfold3.core.model.layers.attention_pair_bias"
    return [
        Patch(
            "SampleDiffusion._sample_rollout",
            GROUP,
            (ds, "SampleDiffusion", "_sample_rollout"),
            _rollout,
        ),
        Patch(
            "DiffusionConditioning.forward",
            GROUP,
            (
                "openfold3.core.model.layers.diffusion_conditioning",
                "DiffusionConditioning",
                "forward",
            ),
            _conditioning_forward,
        ),
        Patch(
            "DiffusionTransformer.forward",
            GROUP,
            (
                "openfold3.core.model.layers.diffusion_transformer",
                "DiffusionTransformer",
                "forward",
            ),
            _diffusion_transformer_forward,
        ),
        Patch(
            "DiffusionAttentionPairBias._prep_bias",
            GROUP,
            (ap, "DiffusionAttentionPairBias", "_prep_bias"),
            _diffusion_prep_bias,
        ),
        Patch(
            "AdaLN.forward",
            GROUP,
            ("openfold3.core.model.primitives.normalization", "AdaLN", "forward"),
            _adaln_forward,
        ),
        Patch(
            "ConditionedTransitionBlock._transition",
            GROUP,
            ("openfold3.core.model.layers.transition", "ConditionedTransitionBlock", "_transition"),
            _cond_transition,
        ),
        Patch(
            "AtomAttentionEncoder.get_atom_reps",
            GROUP,
            (
                "openfold3.core.model.layers.sequence_local_atom_attention",
                "AtomAttentionEncoder",
                "get_atom_reps",
            ),
            _atom_reps,
        ),
        Patch(
            "CrossAttentionPairBias.forward",
            GROUP,
            (ap, "CrossAttentionPairBias", "forward"),
            _cross_attention_pair_bias,
        ),
    ]
