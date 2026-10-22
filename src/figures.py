"""Figures for the compression-vs-group-robustness sweep.

Reads results/sweep.csv + results/baseline.json, writes PNGs to figures/.

Two conventions that carry meaning, not decoration:

  * Every accuracy panel is drawn on a SHARED y-axis from 0 to 1. Per-panel
    autoscaling would rescale each method to fill its own frame and make a
    2-point drop look exactly like a 70-point collapse.
  * Configurations where the model has collapsed to a single predicted class
    (`pred_mean` pinned near 0 or 1 -- gate V7b) are drawn hollow. Past that
    point the gap NARROWS again, purely because a model that predicts one class
    everywhere has no group disparity left to measure. Reading that narrowing as
    "robustness recovers" is the single most available misreading of this data.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import GROUP_NAMES

# Categorical slots 1-5 in the palette's fixed order -- blue, orange, aqua,
# yellow, magenta -- taken in order and never cycled. Validated as a set on the
# light surface: worst adjacent CVD dE 9.1, normal-vision 19.6, both clear.
# Aqua, yellow and magenta fall under 3:1 contrast against the surface, so the
# required relief is shipped everywhere they appear: every series is directly
# labelled or line-style-coded, and summary_table() writes the table view.
# Identity is never carried by color alone in these figures.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA = "#e87ba4"  # slot 5, taken in order -- hues are never cycled or skipped
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"

FIGURES = Path("figures")

# Panel order and the knob each one sweeps. Quantization's two granularities are
# separate configurations, so they get separate panels rather than being averaged
# into one misleading curve.
PANELS = [
    ("magnitude", "", "Magnitude pruning", "sparsity", False),
    ("heads", "", "Attention-head removal", "heads removed (of 72)", False),
    ("ffn", "", "FFN neuron removal", "neuron fraction", False),
    ("quantize", "per_tensor", "Quantization (per-tensor)", "bits", True),
    ("quantize", "per_channel", "Quantization (per-channel)", "bits", True),
]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 11, "axes.labelsize": 9.5,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "font.size": 10, "legend.frameon": False,
        "figure.dpi": 150,
    })


def load(results_dir="results", seed: int = 0, tag: str = "",
         csv_name: str | None = None, baseline_name: str | None = None
         ) -> tuple[pd.DataFrame, dict]:
    """One arm's rows. The figures draw one curve per method, so handing them a
    multi-seed CSV would silently overplot two seeds as one series -- hence the
    filter and the assertion rather than a bare read_csv."""
    r = Path(results_dir)
    df = pd.read_csv(csv_name or r / "sweep.csv")
    df["granularity"] = df["granularity"].fillna("")
    if "tag" in df.columns:
        df["tag"] = df["tag"].fillna("")
        df = df[df["tag"] == tag]
    if "seed" in df.columns:
        df = df[df["seed"] == seed]
    df = df.reset_index(drop=True)
    assert len(df), f"no rows for seed={seed} tag={tag!r} in {csv_name or 'sweep.csv'}"
    dup = df.duplicated(subset=["method", "param_value", "granularity"])
    assert not dup.any(), f"duplicate configs after filtering: {df[dup]}"

    stem = f"{tag}_seed{seed}" if tag else f"seed{seed}"
    bpath = Path(baseline_name) if baseline_name else (
        r / "baseline.json" if (not tag and seed == 0)
        else r / f"baseline_{stem}.json")
    baseline = json.loads(bpath.read_text())
    return df, baseline


def panel_data(df: pd.DataFrame, method: str, granularity: str) -> pd.DataFrame:
    sub = df[(df["method"] == method) & (df["granularity"] == granularity)]
    return sub.sort_values("param_value").reset_index(drop=True)


def collapsed(sub: pd.DataFrame) -> pd.Series:
    """V7b: one predicted class everywhere. Accuracy is still defined; the gap
    is not meaningfully a bias measurement any more."""
    return (sub["pred_mean"] < 0.05) | (sub["pred_mean"] > 0.95)


def _axes_common(ax, xlabel: str, invert: bool) -> None:
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if invert:  # fewer bits = more compression, so severity still runs left->right
        ax.invert_xaxis()


def _plot_series(ax, x, y, mask_collapsed, color, label):
    """2px line, >=6pt markers, hollow where the model has collapsed."""
    ax.plot(x, y, color=color, lw=2, zorder=3, label=label)
    ok = ~mask_collapsed
    ax.plot(x[ok], y[ok], "o", color=color, ms=6, mec=SURFACE, mew=1.5, zorder=4)
    if mask_collapsed.any():
        ax.plot(x[mask_collapsed], y[mask_collapsed], "o", mfc=SURFACE,
                mec=color, mew=2, ms=6, zorder=4)


def fig_avg_vs_worst(df, baseline, out=None) -> Path:
    """THE result: average accuracy holds while worst-group accuracy falls away.

    The shaded band between the two lines is `gap_adj` -- drawing it as an area
    makes "the gap widens" something you see rather than something you infer
    from two separate curves.
    """
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharey=True)
    flat = axes.ravel()

    for ax, (method, gran, title, xlabel, invert) in zip(flat, PANELS):
        sub = panel_data(df, method, gran)
        x, coll = sub["param_value"], collapsed(sub)
        avg, wg = sub["acc_adj_avg"], sub["acc_wg"]

        ax.fill_between(x, wg, avg, color=C_ORANGE, alpha=0.13, lw=0, zorder=2)
        _plot_series(ax, x, avg, coll, C_BLUE, "adjusted average")
        _plot_series(ax, x, wg, coll, C_ORANGE, "worst group")

        ax.axhline(baseline["acc_wg"], color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_title(title, color=INK, pad=8, loc="left")
        _axes_common(ax, xlabel, invert)

        # Direct labels on the first panel; the shared legend covers the rest.
        # Anchored mid-curve, where the two lines have separated -- at the
        # left edge they sit on top of each other and of the panel title.
        # Each label hangs on the far side of its own line, so neither can
        # drift into the other's.
        if ax is flat[0]:
            j = len(x) // 2
            ax.annotate("adjusted average", (x.iloc[j], avg.iloc[j]),
                        xytext=(4, -16), textcoords="offset points",
                        color=C_BLUE, fontsize=9.5, fontweight="bold")
            ax.annotate("worst group", (x.iloc[j], wg.iloc[j]),
                        xytext=(4, 10), textcoords="offset points",
                        color=C_ORANGE, fontsize=9.5, fontweight="bold")

    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("test accuracy")

    legend_ax = flat[5]
    legend_ax.axis("off")
    legend_ax.plot([], [], color=C_BLUE, lw=2, marker="o", ms=6, label="adjusted average accuracy")
    legend_ax.plot([], [], color=C_ORANGE, lw=2, marker="o", ms=6, label="worst-group accuracy")
    legend_ax.plot([], [], color=INK_MUTED, lw=1, ls=(0, (4, 3)), label="uncompressed worst group")
    legend_ax.plot([], [], color=INK_MUTED, lw=0, marker="o", mfc=SURFACE,
                   mec=INK_MUTED, mew=2, ms=6, label="collapsed to one class (V7b)")
    legend_ax.legend(loc="center left", fontsize=9.5, labelcolor=INK_2,
                     handlelength=2.4, borderaxespad=0)
    legend_ax.text(0, 0.86, "Shaded band = worst-group gap", transform=legend_ax.transAxes,
                   fontsize=9.5, color=INK_2, fontweight="bold")

    fig.suptitle(
        "Compression leaves average accuracy nearly intact while worst-group accuracy collapses",
        x=0.008, ha="left", fontsize=13.5, color=INK, fontweight="bold",
    )
    fig.text(0.008, 0.935,
             f"ViT-S/16 fine-tuned on Waterbirds (ERM, seed {baseline['seed']}); "
             f"compressed post-hoc, never retrained. Uncompressed: "
             f"{baseline['acc_adj_avg']:.1%} adjusted average, "
             f"{baseline['acc_wg']:.1%} worst group.",
             ha="left", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.915))

    out = Path(out or FIGURES / "fig1_avg_vs_worst_group.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_tradeoff(df, baseline, out=None) -> Path:
    """What an average-accuracy-only evaluation would miss.

    Both axes are accuracies, which is what lets all five configurations share
    one plot -- their native knobs (sparsity, head count, bits) are not
    commensurable, but their outcomes are. Read it vertically: at a fixed
    average accuracy, how far has the worst group fallen?
    """
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))

    colors = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA]
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    # Parked high on the diagonal: the lower stretch is where the method curves
    # and their direct labels live.
    ax.annotate("no gap\n(worst group = average)", (0.90, 0.90), xytext=(-10, -4),
                textcoords="offset points", ha="right", va="top",
                fontsize=9, color=INK_MUTED)

    for (method, gran, title, _xl, _inv), color in zip(PANELS, colors):
        sub = panel_data(df, method, gran)
        sub = sub[~collapsed(sub)]  # a collapsed model's "gap" is an artifact
        if sub.empty:
            continue
        sub = sub.sort_values("acc_adj_avg", ascending=False)
        ax.plot(sub["acc_adj_avg"], sub["acc_wg"], color=color, lw=2, zorder=3, label=title)
        ax.plot(sub["acc_adj_avg"], sub["acc_wg"], "o", color=color, ms=6,
                mec=SURFACE, mew=1.5, zorder=4)
        # Direct label at the MOST-compressed end. Every method starts from the
        # same uncompressed checkpoint, so the other end is one shared point and
        # five labels would land on top of each other there; here the curves have
        # separated. These labels are also the required relief for the three
        # slots under 3:1 contrast.
        tip = sub.iloc[-1]
        ax.annotate(title, (tip["acc_adj_avg"], tip["acc_wg"]),
                    xytext=(0, 13), textcoords="offset points", ha="center",
                    color=color, fontsize=9, fontweight="bold")

    ax.plot([baseline["acc_adj_avg"]], [baseline["acc_wg"]], marker="*", ms=18,
            color=INK, zorder=5)
    ax.annotate("uncompressed", (baseline["acc_adj_avg"], baseline["acc_wg"]),
                xytext=(-16, 12), textcoords="offset points", ha="right",
                fontsize=9.5, color=INK, fontweight="bold")

    ax.set_xlabel("adjusted average accuracy  (what a standard evaluation reports)")
    ax.set_ylabel("worst-group accuracy  (what it misses)")
    ax.set_xlim(0.32, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_2, handlelength=2.4)

    fig.suptitle("Equal average accuracy, unequal worst-group accuracy",
                 x=0.008, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.008, 0.93,
             "Collapsed configurations excluded. Vertical distance to the dashed "
             "line is the worst-group gap.",
             ha="left", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out = Path(out or FIGURES / "fig2_tradeoff.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_group(df, baseline, out=None) -> Path:
    """Which groups actually pay. The two minority (shortcut-conflicting) groups
    are drawn solid, the two majority (shortcut-aligned) groups dashed."""
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharey=True)
    flat = axes.ravel()
    colors = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]
    minority = {1, 2}  # landbird_on_water, waterbird_on_land

    for ax, (method, gran, title, xlabel, invert) in zip(flat, PANELS):
        sub = panel_data(df, method, gran)
        x, coll = sub["param_value"], collapsed(sub)
        for g in range(4):
            y = sub[f"acc_g{g}"]
            ax.plot(x, y, color=colors[g], lw=2,
                    ls="-" if g in minority else (0, (4, 3)), zorder=3,
                    label=GROUP_NAMES[g].replace("_", " "))
            ax.plot(x[~coll], y[~coll], "o", color=colors[g], ms=5,
                    mec=SURFACE, mew=1.2, zorder=4)
            # Hollow where the model predicts one class everywhere. Those are
            # the points where the per-group curves cross violently -- a group
            # whose label happens to be the surviving prediction jumps to 1.0.
            # That is the collapse, not a group becoming easier.
            if coll.any():
                ax.plot(x[coll], y[coll], "o", mfc=SURFACE, mec=colors[g],
                        mew=1.8, ms=5, zorder=4)
        ax.set_title(title, color=INK, pad=8, loc="left")
        _axes_common(ax, xlabel, invert)

    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("test accuracy")

    legend_ax = flat[5]
    legend_ax.axis("off")
    for g in range(4):
        legend_ax.plot([], [], color=colors[g], lw=2,
                       ls="-" if g in minority else (0, (4, 3)), marker="o", ms=5,
                       label=f"{GROUP_NAMES[g].replace('_', ' ')}"
                             f"{'  (minority)' if g in minority else '  (majority)'}")
    legend_ax.plot([], [], color=INK_MUTED, lw=0, marker="o", mfc=SURFACE,
                   mec=INK_MUTED, mew=1.8, ms=5, label="collapsed to one class (V7b)")
    legend_ax.legend(loc="center left", fontsize=9.5, labelcolor=INK_2,
                     handlelength=2.6, borderaxespad=0)
    legend_ax.text(0, 0.80,
                   "Solid = shortcut-conflicting groups.\nDashed = shortcut-aligned groups.",
                   transform=legend_ax.transAxes, fontsize=9.5, color=INK_2)

    fig.suptitle("The cost falls on the groups that conflict with the background shortcut",
                 x=0.008, ha="left", fontsize=13.5, color=INK, fontweight="bold")
    fig.text(0.008, 0.935,
             "Per-group test accuracy under each compression method. Line style "
             "encodes minority/majority, so group identity is never color-alone.",
             ha="left", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.915))

    out = Path(out or FIGURES / "fig3_per_group.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def summary_table(df, baseline, out=None) -> Path:
    """The numbers behind the figures, as the required table view."""
    rows = []
    for method, gran, title, _xl, _inv in PANELS:
        sub = panel_data(df, method, gran)
        for _, r in sub.iterrows():
            rows.append({
                "configuration": title,
                "param": f"{r['param_name']}={r['param_value']:g}",
                "achieved_sparsity": round(r["achieved_sparsity"], 4),
                "acc_adj_avg": round(r["acc_adj_avg"], 4),
                "acc_wg": round(r["acc_wg"], 4),
                "gap_adj": round(r["gap_adj"], 4),
                "d_avg_vs_baseline": round(r["acc_adj_avg"] - baseline["acc_adj_avg"], 4),
                "d_wg_vs_baseline": round(r["acc_wg"] - baseline["acc_wg"], 4),
                "worst_group": GROUP_NAMES[int(r["worst_group_id"])],
                "collapsed": bool(r["pred_mean"] < 0.05 or r["pred_mean"] > 0.95),
            })
    table = pd.DataFrame(rows)
    out = Path(out or Path("results") / "summary_table.csv")
    table.to_csv(out, index=False)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="training arm; '' is ERM")
    ap.add_argument("--suffix", default=None,
                    help="appended to each output filename (default: derived "
                         "from --tag/--seed; empty for ERM seed 0)")
    args = ap.parse_args()

    df, baseline = load(seed=args.seed, tag=args.tag)
    suffix = args.suffix
    if suffix is None:
        suffix = "" if (not args.tag and args.seed == 0) else (
            f"_{args.tag}_seed{args.seed}" if args.tag else f"_seed{args.seed}")

    fig_dir, res_dir = Path("figures"), Path("results")
    outs = (
        fig_avg_vs_worst(df, baseline, fig_dir / f"fig1_avg_vs_worst_group{suffix}.png"),
        fig_tradeoff(df, baseline, fig_dir / f"fig2_tradeoff{suffix}.png"),
        fig_per_group(df, baseline, fig_dir / f"fig3_per_group{suffix}.png"),
        summary_table(df, baseline, res_dir / f"summary_table{suffix}.csv"),
    )
    for path in outs:
        print("wrote", path)


if __name__ == "__main__":
    main()
