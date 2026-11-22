"""Figure 1: average accuracy hides group damage.

Table 1 reports one sparsity. The gap it describes is a curve, and the curve
is the argument: adjusted average stays near its dense value while worst-group
accuracy falls off. Drawn from 3-seed means so it cannot contradict Table 1,
which is why the earlier seed-0 version of this plot was dropped.

Writes report/figures/fig1_damage.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures"

erm = pd.read_csv(ROOT / "results" / "sweep.csv")
erm["granularity"] = erm["granularity"].fillna("")
erm = erm[erm["tag"].fillna("") == ""]


def curve(method: str) -> pd.DataFrame:
    d = erm[erm["method"] == method]
    return d.groupby("param_value")[["acc_adj_avg", "acc_wg"]].mean() * 100


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 1.85))
    c = curve("magnitude")
    ax.plot(c.index, c["acc_adj_avg"], "o-", color="#2E6E7E", lw=1.8, ms=3.5,
            label="adjusted average")
    ax.plot(c.index, c["acc_wg"], "s--", color="#CC0000", lw=1.8, ms=3.5,
            label="worst group")
    ax.set_xlabel("fraction of transformer weights zeroed", fontsize=8)
    ax.set_ylabel("accuracy (%)", fontsize=8)
    ax.axvspan(0.78, 0.97, color="#000000", alpha=0.055, lw=0, zorder=0)
    ax.text(0.875, 97, "collapsed", fontsize=7, ha="center", color="#666666")
    ax.set_xlim(-0.03, 0.98)
    ax.set_ylim(0, 104)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    # The claim the report makes in words, marked where it is measured.
    ax.annotate("$-0.5$ vs $-14.9$ at 30%", xy=(0.3, 58.4), xytext=(0.40, 40),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7,
                                              color="#444444"))
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_damage.png", dpi=220)
    print(curve("magnitude").round(1).to_string())


if __name__ == "__main__":
    main()
