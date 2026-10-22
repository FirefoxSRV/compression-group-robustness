"""Structured attention-head removal.

Heads are removed by masking, not by shrinking tensors. The shapes stay
ViT-S/16 so `assert_structure` keeps holding and one CSV column
(`achieved_sparsity`) stays comparable across all four methods -- a physically
narrowed qkv would make "fraction of targeted weights zeroed" mean something
different here than under magnitude pruning.

Masking is EXACT, not an approximation, and that rests on the layout asserted
in model.assert_structure: attention ends

    x = x.transpose(1, 2).reshape(B, N, attn_dim)   # heads laid out contiguously
    x = self.norm(x)                                # Identity -- asserted
    x = self.proj(x)

so head h owns exactly proj-input columns [h*head_dim : (h+1)*head_dim]. Zeroing
those columns deletes head h's contribution to the residual stream outright.
Its qkv rows are zeroed too: with proj already blocking the output they cannot
change the result, but they ARE dead weight, and leaving them out of the count
would understate how much of the model a head sweep actually removes.

(Zeroing q and k leaves head h with all-zero logits -> uniform softmax. Finite,
not NaN, and discarded by proj regardless.)
"""
from __future__ import annotations

import torch

from ._common import base_stats, clone_model


def head_slices(attn, h: int):
    """(qkv row indices, proj column indices) owned by head `h`.

    qkv.weight is (3*attn_dim, dim) ordered [q; k; v] -- the `reshape(B, N, 3,
    num_heads, head_dim)` in forward puts the 3 ahead of the head axis -- so
    head h appears once in each of the three sections.
    """
    d, a = attn.head_dim, attn.attn_dim
    rows = torch.cat([torch.arange(s * a + h * d, s * a + (h + 1) * d) for s in range(3)])
    cols = torch.arange(h * d, (h + 1) * d)
    return rows, cols


def head_importance(model) -> list[tuple[float, int, int]]:
    """(score, block, head) for every head, by L2 norm of the weights it owns.

    Data-free and deterministic, matching the magnitude family: this sweep asks
    what happens when you delete the smallest parts of the network, and a head's
    size is the norm of its own qkv rows plus proj columns. Every head holds
    identical parameter counts (head_dim is uniform, asserted), so raw norms are
    directly comparable ACROSS blocks -- no per-block normalization needed, and
    none applied, since normalizing would silently force a near-uniform
    per-block quota and hide where capacity actually is.
    """
    out = []
    for b, blk in enumerate(model.blocks):
        attn = blk.attn
        for h in range(attn.num_heads):
            rows, cols = head_slices(attn, h)
            score = (
                attn.qkv.weight.detach()[rows].pow(2).sum()
                + attn.proj.weight.detach()[:, cols].pow(2).sum()
            ).sqrt()
            out.append((float(score), b, h))
    return out


def select_heads(model, n_heads_removed: int, min_heads_per_block: int = 1):
    """Globally smallest heads first, subject to a per-block floor.

    The floor is not cosmetic: emptying a block's attention entirely turns that
    block into an MLP-only layer, a different architecture rather than a
    compressed one, and the resulting accuracy cliff would be attributed to
    "head sparsity" in the plots. With 12 blocks x 6 heads and a floor of 1,
    at most 60 heads can go -- which is exactly where the configured sweep stops.
    """
    n_blocks = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    max_removable = n_blocks * (n_heads - min_heads_per_block)
    if not 0 <= n_heads_removed <= max_removable:
        raise ValueError(
            f"n_heads_removed must be in [0, {max_removable}], got {n_heads_removed}"
        )

    remaining = [n_heads] * n_blocks
    selected = []
    # Sort by (score, block, head): the trailing keys make ties resolve
    # identically on every run and every machine.
    for _score, b, h in sorted(head_importance(model)):
        if len(selected) == n_heads_removed:
            break
        if remaining[b] <= min_heads_per_block:
            continue
        remaining[b] -= 1
        selected.append((b, h))
    return selected


def apply(model, n_heads_removed: int, min_heads_per_block: int = 1):
    m = clone_model(model)
    selected = select_heads(m, n_heads_removed, min_heads_per_block)

    with torch.no_grad():
        for b, h in selected:
            attn = m.blocks[b].attn
            rows, cols = head_slices(attn, h)
            attn.qkv.weight[rows] = 0.0
            attn.proj.weight[:, cols] = 0.0

    per_block = [0] * len(m.blocks)
    for b, _h in selected:
        per_block[b] += 1

    stats = base_stats(m)
    stats.update(
        n_heads_removed=len(selected),
        requested_heads_removed=n_heads_removed,
        heads_removed_per_block=per_block,
        removed=sorted(selected),
    )
    return m, stats
