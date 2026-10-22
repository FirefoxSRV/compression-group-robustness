"""Group-aware accuracy metrics for Waterbirds.

Pure tensor code. Never imports a model, never touches MPS-unfriendly float64:
counts accumulate as int64 on CPU and the division happens in Python floats.
"""
from __future__ import annotations

import torch

GROUP_NAMES = [
    "landbird_on_land",    # majority, aligned
    "landbird_on_water",   # MINORITY, conflicting
    "waterbird_on_land",   # MINORITY, conflicting
    "waterbird_on_water",  # majority, aligned
]
N_GROUPS = 4

# Waterbirds training-split group counts. Asserted by V1 -- if these ever
# disagree with the data, group indexing is wrong and every downstream number
# is meaningless.
TRAIN_GROUP_COUNTS = [3498, 184, 56, 1057]
_TRAIN_TOTAL = sum(TRAIN_GROUP_COUNTS)  # 4795

# Exact fractions, not rounded literals. This reproduces the WILDS
# `adj_acc_avg` definition (a weighted sum over raw counts / 4795) bit for bit,
# so `acc_adj_avg` is literally the number reported in the literature.
# The spec's rounded [0.72951, 0.03838, 0.01168, 0.22044] sums to 1.00001.
TRAIN_GROUP_FRACTIONS = [c / _TRAIN_TOTAL for c in TRAIN_GROUP_COUNTS]


def group_index(metadata: torch.Tensor) -> torch.Tensor:
    """0=landbird/land, 1=landbird/water, 2=waterbird/land, 3=waterbird/water.

    `metadata` is (N, 2) with column 0 = background/place and column 1 = y,
    matching the WILDS Waterbirds `metadata_fields = ['background', 'y']`.
    """
    background = metadata[:, 0].long()
    y = metadata[:, 1].long()
    return 2 * y + background


def per_group_counts(
    preds: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor
) -> tuple[list[int], list[int]]:
    """Return (correct_per_group, total_per_group) as Python ints."""
    preds = preds.detach().cpu().long()
    labels = labels.detach().cpu().long()
    groups = groups.detach().cpu().long()
    correct = preds.eq(labels)

    n_correct, n_total = [], []
    for g in range(N_GROUPS):
        mask = groups.eq(g)
        n_total.append(int(mask.sum()))
        n_correct.append(int(correct[mask].sum()))
    return n_correct, n_total


def adjusted_average(per_group_acc) -> float:
    """Per-group accuracy reweighted by TRAINING frequency.

    Waterbirds val/test are deliberately more group-balanced than train, so the
    plain mean over the test set is NOT the ~97% figure quoted in the
    literature. This is.
    """
    return float(sum(p * a for p, a in zip(TRAIN_GROUP_FRACTIONS, per_group_acc)))


def compute_metrics(
    preds: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor
) -> dict:
    """Full metric dict for one evaluation pass."""
    n_correct, n_total = per_group_counts(preds, labels, groups)

    # A group with no samples would make worst-group meaningless; V2 forbids it.
    per_group_acc = [
        (c / t) if t > 0 else float("nan") for c, t in zip(n_correct, n_total)
    ]

    total = sum(n_total)
    acc_avg_plain = sum(n_correct) / total if total > 0 else float("nan")
    acc_adj_avg = adjusted_average(per_group_acc)

    worst_group_id = min(range(N_GROUPS), key=lambda g: per_group_acc[g])
    acc_wg = per_group_acc[worst_group_id]

    return {
        "acc_avg_plain": acc_avg_plain,
        "acc_adj_avg": acc_adj_avg,
        "acc_wg": acc_wg,
        "worst_group_id": worst_group_id,
        # THE headline quantity: flat under pure capacity loss, widening under
        # returning shortcut reliance.
        "gap_adj": acc_adj_avg - acc_wg,
        **{f"acc_g{g}": per_group_acc[g] for g in range(N_GROUPS)},
        **{f"n_g{g}": n_total[g] for g in range(N_GROUPS)},
    }
