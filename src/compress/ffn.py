"""Structured FFN neuron removal.

Neuron i of a block's MLP is the pair (fc1 row i, fc2 column i): fc1 row i
computes the pre-activation, fc2 column i is the only path by which the
activation reaches the residual stream. Zeroing both removes the neuron exactly.

fc1.bias is deliberately NOT touched -- biases are outside the frozen
compression scope (spec 6.1). It costs nothing: a pruned neuron still emits the
constant GELU(bias_i), but fc2's zeroed column multiplies it by zero, so the
contribution is exactly zero anyway. Removal stays exact without widening the
scope that every other method is measured against.

Masked, not narrowed, for the same reason as `heads`: shapes stay ViT-S/16 so
achieved_sparsity means the same thing across all four methods.
"""
from __future__ import annotations

import torch

from ._common import base_stats, clone_model


def neuron_importance(model) -> list[tuple[float, int, int]]:
    """(score, block, neuron) by L2 norm of the neuron's fc1 row + fc2 column.

    Both halves matter: a neuron with a large fc1 row but a near-zero fc2 column
    already contributes nothing, and scoring on fc1 alone would rank it as
    important and spend the pruning budget elsewhere.
    """
    out = []
    for b, blk in enumerate(model.blocks):
        # (hidden,) each -- vectorized, since this runs over 18,432 neurons.
        row_sq = blk.mlp.fc1.weight.detach().pow(2).sum(dim=1)
        col_sq = blk.mlp.fc2.weight.detach().pow(2).sum(dim=0)
        scores = (row_sq + col_sq).sqrt()
        out.extend((float(s), b, i) for i, s in enumerate(scores))
    return out


def select_neurons(model, neuron_fraction: float):
    """Globally smallest neurons first, across all blocks.

    No per-block floor here, unlike `heads`: emptying one block's MLP leaves the
    attention path and the residual stream intact, so it degrades rather than
    changing the architecture. Blocks are free to give up wildly different
    shares, which is the point -- where the ranking chooses to cut is a result,
    not a nuisance to be normalized away.
    """
    if not 0.0 <= neuron_fraction < 1.0:
        raise ValueError(f"neuron_fraction must be in [0,1), got {neuron_fraction}")

    scores = neuron_importance(model)
    k = int(round(neuron_fraction * len(scores)))
    # (score, block, neuron) ordering makes ties deterministic across runs.
    return sorted(sorted(scores)[:k], key=lambda t: (t[1], t[2]))


def apply(model, neuron_fraction: float):
    m = clone_model(model)
    selected = select_neurons(m, neuron_fraction)

    per_block: dict[int, list[int]] = {}
    for _score, b, i in selected:
        per_block.setdefault(b, []).append(i)

    with torch.no_grad():
        for b, idx in per_block.items():
            idx = torch.as_tensor(idx, dtype=torch.long)
            m.blocks[b].mlp.fc1.weight[idx] = 0.0
            m.blocks[b].mlp.fc2.weight[:, idx] = 0.0

    n_hidden = m.blocks[0].mlp.fc1.weight.shape[0]
    stats = base_stats(m)
    stats.update(
        requested_fraction=neuron_fraction,
        n_neurons_removed=len(selected),
        n_neurons_total=len(m.blocks) * n_hidden,
        neurons_removed_per_block=[len(per_block.get(b, [])) for b in range(len(m.blocks))],
    )
    return m, stats
