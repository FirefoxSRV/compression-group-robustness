"""Compression method registry.

Universal contract (spec 6.0). Every method:

    apply(model, **params) -> (compressed_model, stats)

  * returns a NEW model
  * NEVER mutates the input model (each `apply` deepcopies first)
  * stats always includes n_params_targeted, n_params_zeroed, achieved_sparsity

Compression is applied to a trained checkpoint and evaluated immediately.
There is no retraining afterwards -- that is the experimental design.
"""
from . import ffn, heads, magnitude, quantize

REGISTRY = {
    "magnitude": magnitude.apply,
    "heads": heads.apply,
    "ffn": ffn.apply,
    "quantize": quantize.apply,
}

# The knob each method sweeps, for the CSV `param_name` column.
PARAM_NAME = {
    "magnitude": "sparsity",
    "heads": "n_heads_removed",
    "ffn": "neuron_fraction",
    "quantize": "bits",
}

__all__ = ["REGISTRY", "PARAM_NAME", "magnitude", "heads", "ffn", "quantize"]
