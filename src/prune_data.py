"""Difficulty-based data pruning (Mulchandani & Kim, ICLR 2025).

The premise: a spurious correlation is learned because of a handful of training
samples whose CORE feature is hard to learn -- where the core feature is hard,
the network prefers the simpler spurious one. Removing those samples severs the
dependency.

    d_i = || p(w_e, x_i) - onehot(y_i) ||_2       (one ERM run, fixed early epoch e)

Rank by d_i, drop the HARDEST, retrain from scratch on the rest.

TWO SELECTION RULES, chosen with `--variant`:

  * `global` -- hardest per class across the whole split, equal counts removed,
    no group labels. The `prune_general.py` path upstream, and the only rule
    here until 2026-10-27.
  * `spurious` -- hardest within each class's spurious-aligned group only,
    classes equalized. This is what the paper does on Waterbirds, which is its
    IDENTIFIABLE setting. See select_keep_indices_spurious for the quotes.

`global` was chosen first on the reasoning that Phase 1's whole premise is a
group-blind pipeline and a group-aware pruner would leak exactly the
information the ERM baseline is denied. That reasoning stands on its own terms,
but it is NOT a reproduction of this paper on Waterbirds: §6.3 says the
group-blind rule "is not feasible" there, and the Oct 26 run measured precisely
why -- it deleted 48% of g1 and 68% of the 56-sample worst group g2, and
worst-group accuracy fell monotonically with the budget.

So both rules are kept. `spurious` is the reproduction; `global` is the
contrast that demonstrates the paper's own stated limitation.

`group_report()` describes what a rule removed, per group. The `global` path
never reads it; the `spurious` path selects on group membership by design.
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


# The spurious-aligned group of each class: background matches label. With
# group = 2*y + place (src/data.py:100) that is group == 3*c, i.e. g0 for
# landbird-on-land (3498 train) and g3 for waterbird-on-water (1057).
# g1 (184) and g2 (56) are the counter-spurious groups, and g2 is the worst
# group every Phase 1 number is reported on.
ALIGNED_GROUP = {0: 0, 1: 3}


def select_keep_indices_spurious(
    difficulty: torch.Tensor,
    y: torch.Tensor,
    group: torch.Tensor,
    sparsity: float,
    equalize_classes: bool = True,
) -> dict:
    """The IDENTIFIABLE-setting rule from Mulchandani & Kim (ICLR 2025) §6.3.

    Waterbirds is their *identifiable* setting, where the paper is explicit that
    the method uses group labels (Table 1 marks it Train/Val checked):

        "our approach works with group labels ... and simply prunes those
         samples containing the hardest core features within groups containing
         the spurious feature associated with each class"

    and equally explicit that the group-blind rule does NOT work there:

        "it is not feasible to simply prune the hardest samples of the training
         data as this will primarily prune samples that do not contain the
         spurious features"

    which is exactly what `select_keep_indices` was measured doing: at a 10%
    budget it removed 48% of g1 and 68% of g2 -- the counter-spurious samples
    that hold worst-group accuracy up -- and worst-group accuracy duly fell.

    So candidates here are drawn ONLY from each class's spurious-aligned group.
    g1 and g2 are never touched, by construction.

    `equalize_classes` implements the paper's other Waterbirds-specific
    sentence, "we prune such that the total number of samples in both classes
    is the same". Note that is equal *remaining* class sizes, not equal counts
    removed -- the latter is what `select_keep_indices` does, and it preserves
    the 3682/1113 class imbalance. Equalizing is why the paper's sparsities are
    large (Fig. 8 spans 25-75%): on Waterbirds most of the budget is spent
    undoing the class imbalance. `sparsity` is therefore the pre-balancing
    fraction of each aligned pool, matching Fig. 8's "sparsities reported
    before class balancing"; `achieved_fraction` reports what actually went.

    Ties in difficulty break by ascending index, so this stays a pure function
    of its inputs and reruns are bit-identical (as gate V10b requires of the
    group-blind rule).
    """
    if not 0.0 < sparsity < 1.0:
        raise ValueError(f"sparsity must be in (0,1), got {sparsity}")
    n = difficulty.numel()
    assert y.numel() == n and group.numel() == n, "difficulty/y/group length"

    def hardest_from(pool: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return pool[:0]
        k = min(k, pool.numel())
        order = torch.argsort(difficulty[pool], stable=True)
        return pool[order.flip(0)[:k]]

    pools, base, class_n = {}, {}, {}
    for c in range(N_CLASSES):
        pools[c] = (group == ALIGNED_GROUP[c]).nonzero(as_tuple=True)[0]
        class_n[c] = int((y == c).sum())
        base[c] = int(round(sparsity * pools[c].numel()))

    drop_per_class = {c: base[c] for c in range(N_CLASSES)}
    if equalize_classes:
        remaining = {c: class_n[c] - drop_per_class[c] for c in range(N_CLASSES)}
        target = min(remaining.values())
        for c in range(N_CLASSES):
            extra = remaining[c] - target
            # Only the aligned pool is eligible, so a class whose pool runs out
            # simply cannot be brought all the way down. Report, do not fake it.
            room = pools[c].numel() - drop_per_class[c]
            drop_per_class[c] += min(extra, room)

    drop = [hardest_from(pools[c], drop_per_class[c]) for c in range(N_CLASSES)]
    removed = torch.cat(drop).sort().values

    mask = torch.ones(n, dtype=torch.bool)
    mask[removed] = False
    keep = mask.nonzero(as_tuple=True)[0]
    assert keep.numel() + removed.numel() == n

    kept_y = y[keep]
    return {"keep": keep, "removed": removed, "n_total": n,
            "n_keep": int(keep.numel()),
            "n_removed": int(removed.numel()),
            "fraction": sparsity,
            "achieved_fraction": float(removed.numel()) / n,
            "variant": "spurious",
            "equalize_classes": equalize_classes,
            "class_balanced": equalize_classes,
            "removed_per_class": [int((y[removed] == c).sum()) for c in range(N_CLASSES)],
            "kept_per_class": [int((kept_y == c).sum()) for c in range(N_CLASSES)],
            "pool_sizes": [int(pools[c].numel()) for c in range(N_CLASSES)]}


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
    variant: str = "global",
    equalize_classes: bool = True,
) -> list[dict]:
    """`variant` selects the rule:

      global    -- hardest per class across the whole split, equal counts
                   removed. The group-blind rule; what the paper calls
                   infeasible in identifiable settings like Waterbirds.
      spurious  -- hardest within each class's spurious-aligned group only,
                   classes equalized. The paper's §6.3 identifiable-setting
                   rule. See select_keep_indices_spurious.

    Tags differ by variant ("prune05" vs "spur25") so both can live in one
    results directory and be compared arm against arm.
    """
    blob = load_scores(scores_path)
    d, y = blob["difficulty"], blob["y"].long()
    group = blob.get("group")
    if variant not in ("global", "spurious"):
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "spurious" and group is None:
        raise SystemExit(
            "variant 'spurious' needs group labels, but the scores file has "
            "none. Re-run scoring with a train cache that carries `group`.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for frac in fractions:
        if variant == "spurious":
            sel = select_keep_indices_spurious(
                d, y, group, frac, equalize_classes=equalize_classes)
        else:
            sel = select_keep_indices(d, y, frac, class_balanced=class_balanced)
        pct = f"{round(frac * 100):02d}"
        tag = ("spur" if variant == "spurious" else "prune") + pct
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

        extra = ""
        if variant == "spurious":
            extra = (f" achieved {sel['achieved_fraction']:.3f}, "
                     f"kept/class {sel['kept_per_class']}")
        print(f"[seed {seed}] {tag}: keep {summary['n_keep']}/{sel['n_total']} "
              f"(removed {sel['n_removed']}, per-class {sel['removed_per_class']}"
              f"{extra}) -> {path}")
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
    ap.add_argument("--variant", choices=("global", "spurious"), default=None,
                    help="global: the group-blind rule (hardest per class "
                         "anywhere). spurious: the paper's identifiable-setting "
                         "rule -- hardest within each class's spurious-aligned "
                         "group, classes equalized. Default: prune.variant")
    ap.add_argument("--no-equalize-classes", action="store_true",
                    help="spurious variant only: skip the paper's "
                         "equal-remaining-class-size step")
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

    variant = args.variant or pcfg.get("variant", "global")
    default_fracs = (pcfg.get("spurious_sparsities", [0.25, 0.50, 0.75])
                     if variant == "spurious"
                     else pcfg.get("fractions", [0.01, 0.02, 0.05, 0.10]))
    manifest = build_for_seed(
        args.seed, scores,
        args.fractions or default_fracs,
        pcfg.get("indices_dir", "results/prune_indices"),
        class_balanced=not args.no_class_balance,
        variant=variant,
        equalize_classes=not args.no_equalize_classes,
    )
    out = (Path(pcfg.get("indices_dir", "results/prune_indices"))
           / f"manifest_{variant}_seed{args.seed}.json")
    out.write_text(json.dumps(manifest, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
