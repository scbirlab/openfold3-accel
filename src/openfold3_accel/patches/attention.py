"""Patch group ``fused_attention`` (opt-in, tolerance-class): scaled-dot-product attention for OpenFold3's
bias-attention primitive.

Upstream ``_attention`` materialises the logits, adds every bias with an in-place add (one full pass per
bias, per head, per block, per denoising step), runs a separate softmax pass and a second matmul. On CPU
this elementwise traffic was ~16% of the rollout. PyTorch's fused ``scaled_dot_product_attention`` (with the
biases passed as an additive mask) does the same mathematics in one blocked kernel. Results agree to
floating-point rounding, not bit-for-bit, hence opt-in. Falls back to upstream for reduced-precision inputs when
``use_high_precision`` is set, when gradients are needed, and for anything unexpected.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from .._patch import Patch

GROUP = "fused_attention"
SUMMARY = "OPT-IN (rounding-level differences): fused SDPA with additive biases for the attention primitive"


def _attention(orig):
    @functools.wraps(orig)
    def attention(query, key, value, biases, use_high_precision=False):
        # ``use_high_precision`` is a no-op for fp32 inputs on CPU (upstream's autocast context targets CUDA only),
        # so fp32 can take the fused path; for reduced-precision inputs it asks for fp32 softmax, which we honour
        # by deferring to upstream.
        if (
            (use_high_precision and query.dtype != torch.float32)
            or torch.is_grad_enabled()
            or query.device.type != "cpu"
        ):
            return orig(query, key, value, biases, use_high_precision)
        try:
            mask = None
            for b in biases:
                mask = b if mask is None else mask + b
            if mask is not None:
                mask = mask.to(query.dtype)
            return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=1.0)
        except (RuntimeError, TypeError):
            return orig(query, key, value, biases, use_high_precision)

    return attention


def targets() -> list[Patch]:
    return [
        Patch(
            "attention._attention",
            GROUP,
            ("openfold3.core.model.primitives.attention", None, "_attention"),
            _attention,
        )
    ]
