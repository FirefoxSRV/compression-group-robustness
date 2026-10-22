"""Global unstructured magnitude pruning."""
from __future__ import annotations

import torch

from ..model import targeted_named_weights
from ._common import base_stats, clone_model


def apply(model, sparsity: float):
    """Zero the globally smallest-|w| fraction of targeted weights."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0,1), got {sparsity}")

    m = clone_model(model)
    targeted = targeted_named_weights(m)
    n_targeted = sum(p.numel() for _, p in targeted)
    k = int(round(sparsity * n_targeted))

    if k > 0:
        # kthvalue rather than torch.quantile: it is exact and cheap here --
        # 21.2M fp32 on CPU is 85 MB.
        flat = torch.cat(
            [p.detach().abs().flatten().float().cpu() for _, p in targeted]
        )
        threshold = float(flat.kthvalue(k).values)
        # At 21M fp32 magnitudes, the threshold value is NOT unique: measured
        # ties run 1-3 deep. A plain `|w| > threshold` mask would zero every
        # tied weight and overshoot k, so V4 could only ever be an inequality.
        # Zeroing strictly-below plus an explicit tie budget makes it an
        # equality instead -- exactly k, always.
        n_below = int((flat < threshold).sum())
        del flat
        tie_budget = k - n_below

        with torch.no_grad():
            for _, p in targeted:
                a = p.detach().abs()          # snapshot: survives the mul_ below
                p.mul_((a >= threshold).to(p.dtype))
                if tie_budget > 0:
                    tied = (a == threshold).view(-1).nonzero(as_tuple=True)[0]
                    if tied.numel():
                        # Tensor order, then flat index -- deterministic, so the
                        # same checkpoint always yields the same pruned model.
                        take = tied[:tie_budget]
                        p.view(-1)[take] = 0.0
                        tie_budget -= take.numel()

    stats = base_stats(m)
    stats.update(requested_sparsity=sparsity, threshold=threshold if k > 0 else 0.0)
    return m, stats
