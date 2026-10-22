"""Shared helpers for the compression methods."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..model import n_targeted_params, targeted_named_weights


def clone_model(model: nn.Module) -> nn.Module:
    """A detached deep copy. The no-mutation contract starts here."""
    return copy.deepcopy(model).eval()


def base_stats(model: nn.Module) -> dict:
    n_targeted = n_targeted_params(model)
    n_zero = sum(int((p == 0).sum()) for _, p in targeted_named_weights(model))
    return {
        "n_params_targeted": n_targeted,
        "n_params_zeroed": n_zero,
        "achieved_sparsity": n_zero / n_targeted if n_targeted else 0.0,
    }
