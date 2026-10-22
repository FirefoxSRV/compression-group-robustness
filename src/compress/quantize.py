"""Simulated (fake) symmetric weight quantization.

A methodological choice, not a limitation: the measurement here is accuracy,
not latency or memory, and quantize-dequantize reproduces the numerical effect
exactly. Real int4/int8 kernels are CUDA-only and irrelevant to an accuracy study.
"""
from __future__ import annotations

import torch

from ..model import targeted_named_weights
from ._common import base_stats, clone_model


def fake_quantize(W: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    """Symmetric, no zero-point. Channel = output dim (dim 0 of a Linear weight)."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    if per_channel:
        scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    else:
        scale = W.abs().max().clamp(min=1e-8) / qmax
    return (W / scale).round().clamp(qmin, qmax) * scale


def apply(model, bits: int, granularity: str = "per_tensor"):
    if granularity not in ("per_tensor", "per_channel"):
        raise ValueError(f"unknown granularity {granularity!r}")

    m = clone_model(model)

    # Identity short-circuit. At bits=32, W/scale reaches ~2.1e9, past fp32's
    # 24-bit mantissa, so the round trip would carry ~1e-7 relative error rather
    # than being exactly identity -- and V3 demands identity to 1e-6.
    if bits < 32:
        per_channel = granularity == "per_channel"
        with torch.no_grad():
            for _, p in targeted_named_weights(m):
                p.copy_(fake_quantize(p.detach(), bits, per_channel))

    stats = base_stats(m)
    stats.update(bits=bits, granularity=granularity, identity=bits >= 32)
    return m, stats
