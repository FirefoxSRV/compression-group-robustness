"""Per-group decomposition of the retention ratio.

The headline R(s) is built on worst-group accuracy, which is a minimum over
four groups and therefore says nothing about WHICH group is binding. When
compression pushes a different group to the bottom, R changes for a reason
that has nothing to do with the intervention.

This script reads the sweep CSVs that already exist (acc_g0..acc_g3 are
columns there, written by src/evaluate.py) and reports, per compression
level, the pruned-minus-ERM gain for each group next to the worst-group
gain. No retraining, no new evaluation.

Writes a LaTeX table fragment to report/tables/per_group.tex and a figure to
report/figures/fig4_per_group_retention.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_TAB = ROOT / "report" / "tables"
OUT_FIG = ROOT / "report" / "figures"

# g = 2*y + background, from src/metrics.py:31-39.
GROUP_NAME = {
    0: "landbird on land",
    1: "landbird on water",
    2: "waterbird on land",
    3: "waterbird on water",
}
# The two groups that conflict with the background shortcut.
COUNTER = (1, 2)

JOIN = ["method", "param_value", "granularity", "seed"]


def load(tag: str) -> pd.DataFrame:
    path = RESULTS / ("sweep.csv" if not tag else f"sweep_{tag}.csv")
    df = pd.read_csv(path)
    df["granularity"] = df["granularity"].fillna("")
    return df[df["tag"].fillna("") == tag]


def paired(tag: str) -> pd.DataFrame:
    """ERM and pruned rows for the same (method, setting, seed)."""
    m = load("").merge(load(tag), on=JOIN, suffixes=("_e", "_p"))
    for g in range(4):
        m[f"gain_g{g}"] = m[f"acc_g{g}_p"] - m[f"acc_g{g}_e"]
    m["gain_wg"] = m["acc_wg_p"] - m["acc_wg_e"]
    m["tag"] = tag
    return m


def main() -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    allarms = pd.concat([paired(t) for t in ("spur25", "spur50", "spur75")])

    # The table reports magnitude on the delete-50% arm: the one family and
    # budget the report walks through step by step, so the reader can line
    # this up against the headline R values.
    mag = allarms[(allarms["tag"] == "spur50") & (allarms["method"] == "magnitude")]
    agg = mag.groupby("param_value")[
        [f"gain_g{g}" for g in range(4)] + ["gain_wg"]
    ].mean()
    # Which group is binding, per arm, averaged over seeds is meaningless --
    # report the modal worst group instead.
    mode_e = mag.groupby("param_value")["worst_group_id_e"].agg(
        lambda s: s.value_counts().index[0])
    mode_p = mag.groupby("param_value")["worst_group_id_p"].agg(
        lambda s: s.value_counts().index[0])

    rows = []
    for s in sorted(agg.index):
        if s >= 0.8:          # collapsed everywhere; excluded from R as well
            continue
        r = agg.loc[s]
        rows.append(
            f"{int(round(s * 100))}\\% & "
            f"{100 * r['gain_g0']:+.1f} & {100 * r['gain_g1']:+.1f} & "
            f"\\textbf{{{100 * r['gain_g2']:+.1f}}} & {100 * r['gain_g3']:+.1f} & "
            f"{100 * r['gain_wg']:+.1f} & "
            f"g{int(mode_e.loc[s])} $\\rightarrow$ g{int(mode_p.loc[s])} \\\\"
        )
    (OUT_TAB / "per_group.tex").write_text("\n".join(rows) + "\n")

    # Figure: counter-shortcut gain holding up while worst-group gain dies.
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    x = [s for s in sorted(agg.index) if s < 0.8]
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.plot(x, [100 * agg.loc[s, "gain_g2"] for s in x], "o-",
            color="#2E6E7E", lw=2, label="waterbird on land (g2)")
    ax.plot(x, [100 * agg.loc[s, "gain_wg"] for s in x], "s--",
            color="#CC0000", lw=2, label="worst group")
    ax.set_xlabel("fraction of transformer weights pruned")
    ax.set_ylabel("accuracy gain over ERM (points)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(s * 100))}%" for s in x])
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig4_per_group_retention.png", dpi=200)

    # What goes in the prose.
    print("magnitude, delete-50% arm, mean of 3 seeds (points):")
    print((100 * agg).round(1).to_string())
    print("\nmodal worst group, ERM -> pruned:")
    print(pd.DataFrame({"erm": mode_e, "pruned": mode_p}).to_string())

    # The share of counter-shortcut gain surviving where worst-group says zero.
    hi = allarms[(allarms["param_value"] == 0.7) & (allarms["method"] == "magnitude")]
    print(f"\nat 70% sparsity, all arms and seeds (n={len(hi)}):")
    print(f"  mean g2 gain   {100 * hi['gain_g2'].mean():+.1f} pts")
    print(f"  mean worst-grp {100 * hi['gain_wg'].mean():+.1f} pts")
    print(f"  g2 gain positive in {(hi['gain_g2'] > 0).sum()}/{len(hi)} runs")
    print(f"  worst-grp positive in {(hi['gain_wg'] > 0).sum()}/{len(hi)} runs")

    # The claim the report rests on: split the 168 settings where R is
    # defined by the SIGN of R, and compare what happened to g2 in each half.
    # If g2's gain is the same on both sides, R's sign is not reporting
    # whether the intervention survived.
    ret = []
    for t in ("spur25", "spur50", "spur75"):
        for s in (0, 1, 2):
            r = pd.read_csv(RESULTS / f"retention_{t}_seed{s}.csv")
            r["granularity"] = r["granularity"].fillna("")
            ret.append(r.assign(tag=t, seed=s))
    ret = pd.concat(ret)
    j = ret.merge(allarms, on=["tag", "seed", "method", "param_value",
                               "granularity"])
    d = j[j["retention"].notna() & ~j["collapsed"] & (j["param_value"] != 0)]

    print(f"\nR is defined at {len(d)} compressed settings. Split by sign:")
    for name, half in (("R < 0 ", d[d["retention"] < 0]),
                       ("R >= 0", d[d["retention"] >= 0])):
        off = ~half["worst_group_id_p"].isin(COUNTER)
        print(f"  {name}  n={len(half):3d} | "
              f"worst-group gain {100 * half['gain_wg'].mean():+5.1f} pts | "
              f"g2 gain {100 * half['gain_g2'].mean():+5.1f} pts "
              f"(positive in {(half['gain_g2'] > 0).sum()}/{len(half)}) | "
              f"binding group off g1/g2 in {100 * off.mean():.0f}%")


if __name__ == "__main__":
    main()
