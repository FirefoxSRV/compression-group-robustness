"""Phase 2's headline quantity: how much of the robustness gain survives compression.

Let W_p(s) and W_e(s) be worst-group accuracy of the pruned and ERM models at
the same compression configuration, and

    D(s) = W_p(s) - W_e(s)        the robustness gain still present at s
    R(s) = D(s) / D(0)            the fraction of the DENSE gain retained

R ~ 1: the intervention survives deployment.
R ~ 0: compression silently undid it -- a model made robust, then made small,
       and no longer robust, with average accuracy barely moved.

R is reported per compression family, because Phase 1 already showed the
families are not interchangeable: 6 of 72 attention heads cost more worst-group
accuracy than 30% unstructured sparsity.

Two things this file refuses to do quietly:

  * R is undefined when the dense gain D(0) is not clearly positive. Dividing
    by a near-zero or negative denominator manufactures enormous ratios out of
    noise. `--min-gain` gates it and the gate is reported, not hidden.
  * Collapsed rows (V7b: `pred_mean` pinned near 0 or 1) are excluded. A model
    with one output has no group disparity left, so its "retention" is an
    artifact of a degenerate predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

JOIN_KEYS = ["method", "param_name", "param_value", "granularity"]
COLLAPSE_LO, COLLAPSE_HI = 0.05, 0.95


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("tag", "granularity"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def load_arm(results_dir, tag: str, seed: int) -> pd.DataFrame:
    """One arm's sweep rows. ERM (tag='') lives in sweep.csv, by Phase 1's name."""
    r = Path(results_dir)
    path = r / "sweep.csv" if not tag else r / f"sweep_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (tag={tag!r} seed={seed})")
    df = _read(path)
    if "tag" in df.columns:
        df = df[df["tag"] == tag]
    df = df[df["seed"] == seed].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"{path} has no rows for tag={tag!r} seed={seed}")
    return df


def is_collapsed(df: pd.DataFrame) -> pd.Series:
    return (df["pred_mean"] < COLLAPSE_LO) | (df["pred_mean"] > COLLAPSE_HI)


def dense_row(df: pd.DataFrame) -> pd.Series:
    """The uncompressed row. Prefer magnitude=0 (a true no-op by construction)."""
    for method, value in (("magnitude", 0.0), ("heads", 0.0), ("ffn", 0.0)):
        hit = df[(df["method"] == method) & (df["param_value"].astype(float) == value)]
        if len(hit):
            return hit.iloc[0]
    raise SystemExit("no uncompressed row in this arm; cannot anchor D(0)")


def retention(
    erm: pd.DataFrame, pruned: pd.DataFrame, min_gain: float = 0.01
) -> tuple[pd.DataFrame, dict]:
    d0 = float(dense_row(pruned)["acc_wg"]) - float(dense_row(erm)["acc_wg"])
    gate = {
        "dense_gain": round(d0, 6),
        "min_gain": min_gain,
        "intervention_works": bool(d0 >= min_gain),
        "erm_dense_wg": round(float(dense_row(erm)["acc_wg"]), 6),
        "pruned_dense_wg": round(float(dense_row(pruned)["acc_wg"]), 6),
    }

    m = erm.merge(pruned, on=JOIN_KEYS, suffixes=("_erm", "_pruned"))
    assert len(m) == len(erm) == len(pruned), (
        f"arms do not cover the same grid: erm={len(erm)} pruned={len(pruned)} "
        f"joined={len(m)}")

    m["delta_wg"] = m["acc_wg_pruned"] - m["acc_wg_erm"]
    m["delta_adj"] = m["acc_adj_avg_pruned"] - m["acc_adj_avg_erm"]
    m["collapsed"] = is_collapsed(
        m.rename(columns={"pred_mean_erm": "pred_mean"})) | is_collapsed(
        m.rename(columns={"pred_mean_pruned": "pred_mean"}))
    # Same targeted knob; achieved sparsity can differ slightly between arms
    # for quantization, whose zero count is weight-dependent. Report the drift
    # rather than pretending the x-axis is identical.
    m["sparsity_drift"] = (m["achieved_sparsity_pruned"]
                           - m["achieved_sparsity_erm"]).abs()

    m["retention"] = (m["delta_wg"] / d0) if gate["intervention_works"] else pd.NA
    m.loc[m["collapsed"], "retention"] = pd.NA

    cols = [*JOIN_KEYS, "achieved_sparsity_erm", "achieved_sparsity_pruned",
            "sparsity_drift", "acc_wg_erm", "acc_wg_pruned", "delta_wg",
            "acc_adj_avg_erm", "acc_adj_avg_pruned", "delta_adj",
            "collapsed", "retention"]
    out = m[cols].sort_values(["method", "param_value"]).reset_index(drop=True)
    return out, gate


def by_family(table: pd.DataFrame) -> pd.DataFrame:
    """Mean retention per compression family, collapsed rows already excluded."""
    ok = table[~table["collapsed"] & table["retention"].notna()]
    if ok.empty:
        return pd.DataFrame(columns=["method", "n", "mean_retention",
                                     "min_retention", "max_retention"])
    g = ok.groupby("method")["retention"]
    return pd.DataFrame({
        "n": g.size(), "mean_retention": g.mean(),
        "min_retention": g.min(), "max_retention": g.max(),
    }).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True, help="pruned arm, e.g. prune05")
    ap.add_argument("--erm-tag", default="", help="baseline arm (default: ERM)")
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="dense worst-group gain below which R is undefined")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    erm = load_arm(args.results_dir, args.erm_tag, args.seed)
    pruned = load_arm(args.results_dir, args.tag, args.seed)
    table, gate = retention(erm, pruned, min_gain=args.min_gain)

    out = Path(args.out or Path(args.results_dir) /
               f"retention_{args.tag}_seed{args.seed}.csv")
    table.to_csv(out, index=False)
    gate_path = out.with_suffix(".gate.json")
    gate_path.write_text(json.dumps(gate, indent=2))

    print(f"dense worst-group: ERM {gate['erm_dense_wg']:.4f} -> "
          f"pruned {gate['pruned_dense_wg']:.4f}  (gain {gate['dense_gain']:+.4f})")
    if not gate["intervention_works"]:
        print(f"GATE FAILED: dense gain < {args.min_gain}. Data pruning bought "
              "nothing here, so there is nothing for compression to undo. "
              "Report the null result; R(s) is left undefined.")
    else:
        fam = by_family(table)
        print("\nretention by compression family:")
        print(fam.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out} and {gate_path}")


if __name__ == "__main__":
    main()
