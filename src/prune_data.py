"""Difficulty-based data pruning (Mulchandani & Kim, ICLR 2025), general variant.

The premise: a spurious correlation is learned because of a handful of training
samples whose CORE feature is hard to learn -- where the core feature is hard,
the network prefers the simpler spurious one. Removing those samples severs the
dependency.

    d_i = || p(w_e, x_i) - onehot(y_i) ||_2       (one ERM run, fixed early epoch e)

Rank by d_i, drop the HARDEST class-balanced, retrain from scratch on the rest.

This is the `prune_general.py` variant of the reference implementation: no group
labels, no inferred domain knowledge, no human intervention. That is not an
aesthetic preference -- Phase 1's whole premise is a group-blind pipeline, and a
group-aware pruner would leak exactly the information the ERM baseline is denied.

`group_report()` exists only to describe what the group-blind rule happened to
do. Nothing in the selection path reads it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .metrics import GROUP_NAMES, N_GROUPS

N_CLASSES = 2


def load_scores(path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("difficulty", "y"):
        if k not in blob:
            raise KeyError(f"{path} is not a scores file (missing {k!r})")
    return blob


def class_balanced_budget(n_remove: int, n_classes: int = N_CLASSES) -> list[int]:
    """Split a removal budget as evenly as possible across classes.

    `n_remove = 5, n_classes = 2` -> [3, 2]. The max-min spread is at most one
    sample, which is gate V10's claim. The remainder goes to the low-numbered
    classes deterministically, so the split never depends on dict ordering.
    """
    base, rem = divmod(n_remove, n_classes)
    return [base + (1 if c < rem else 0) for c in range(n_classes)]


def select_keep_indices(
    difficulty: torch.Tensor,
    y: torch.Tensor,
    fraction: float,
    class_balanced: bool = True,
) -> dict:
    """Indices to KEEP after removing `fraction` of the split, hardest-first.

    Ties in `d_i` are broken by ascending index, so the result is a pure
    function of (difficulty, y, fraction) and reruns are bit-identical.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0,1), got {fraction}")
    n = difficulty.numel()
    assert y.numel() == n, f"difficulty {n} vs labels {y.numel()}"
    n_remove = int(round(fraction * n))
    if n_remove < 1:
        raise ValueError(f"fraction {fraction} removes 0 of {n} samples")

    if class_balanced:
        budget = class_balanced_budget(n_remove, N_CLASSES)
        drop = []
        for c, b in enumerate(budget):
            idx_c = (y == c).nonzero(as_tuple=True)[0]
            if b > idx_c.numel():
                raise ValueError(
                    f"class {c} has {idx_c.numel()} samples, cannot drop {b}; "
                    f"fraction {fraction} is too large for a class-balanced split")
            # stable=True makes the ascending sort tie-break by index; the flip
            # then puts the hardest first with ties still index-ordered.
            order = torch.argsort(difficulty[idx_c], stable=True)
            drop.append(idx_c[order.flip(0)[:b]])
        removed = torch.cat(drop).sort().values
    else:
        order = torch.argsort(difficulty, stable=True)
        removed = order.flip(0)[:n_remove].sort().values

    mask = torch.ones(n, dtype=torch.bool)
    mask[removed] = False
    keep = mask.nonzero(as_tuple=True)[0]

    assert keep.numel() + removed.numel() == n
    return {"keep": keep, "removed": removed, "n_total": n,
            "n_removed": int(removed.numel()), "fraction": fraction,
            "class_balanced": class_balanced,
            "removed_per_class": [int((y[removed] == c).sum()) for c in range(N_CLASSES)]}


def group_report(removed: torch.Tensor, group: torch.Tensor) -> dict:
    """DIAGNOSTIC ONLY. Which groups the group-blind rule happened to remove.

    Read this after the fact; never feed it back into selection.
    """
    total = [int((group == g).sum()) for g in range(N_GROUPS)]
    gone = [int((group[removed] == g).sum()) for g in range(N_GROUPS)]
    return {
        GROUP_NAMES[g]: {
            "removed": gone[g], "of": total[g],
            "pct": round(100.0 * gone[g] / total[g], 2) if total[g] else None,
        }
        for g in range(N_GROUPS)
    }


def build_for_seed(
    seed: int,
    scores_path,
    fractions,
    out_dir,
    class_balanced: bool = True,
) -> list[dict]:
    blob = load_scores(scores_path)
    d, y = blob["difficulty"], blob["y"].long()
    group = blob.get("group")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for frac in fractions:
        sel = select_keep_indices(d, y, frac, class_balanced=class_balanced)
        pct = f"{round(frac * 100):02d}"
        tag = f"prune{pct}"
        path = out_dir / f"{tag}_seed{seed}.pt"
        record = {
            **sel, "seed": seed, "tag": tag,
            "score_epoch": blob.get("score_epoch"),
            "scores_path": str(scores_path),
        }
        if group is not None:
            record["group_report"] = group_report(sel["removed"], group)
        torch.save(record, path)

        summary = {k: v for k, v in record.items()
                   if k not in ("keep", "removed")}
        summary["path"] = str(path)
        summary["n_keep"] = int(sel["keep"].numel())
        manifest.append(summary)

        print(f"[seed {seed}] {tag}: keep {summary['n_keep']}/{sel['n_total']} "
              f"(removed {sel['n_removed']}, per-class {sel['removed_per_class']}) "
              f"-> {path}")
        if group is not None:
            for name, st in record["group_report"].items():
                print(f"    {name:<20} removed {st['removed']:>4}/{st['of']:<5}"
                      f" ({st['pct']}%)")
    return manifest


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--scores", default=None,
                    help="difficulty .pt (default: from prune.scores_dir and "
                         "prune.score_epoch)")
    ap.add_argument("--fractions", type=float, nargs="*", default=None)
    ap.add_argument("--no-class-balance", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pcfg = cfg.get("prune", {})
    score_epoch = pcfg.get("score_epoch", 1)
    scores = args.scores or (
        Path(pcfg.get("scores_dir", "results/difficulty"))
        / f"difficulty_seed{args.seed}_ep{score_epoch}.pt")
    if not Path(scores).exists():
        raise SystemExit(
            f"{scores} missing -- run `python -m src.train --seed {args.seed} "
            "--score` first (it needs results/train_cache.pt)")

    manifest = build_for_seed(
        args.seed, scores,
        args.fractions or pcfg.get("fractions", [0.01, 0.02, 0.05, 0.10]),
        pcfg.get("indices_dir", "results/prune_indices"),
        class_balanced=not args.no_class_balance,
    )
    out = Path(pcfg.get("indices_dir", "results/prune_indices")) / f"manifest_seed{args.seed}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
