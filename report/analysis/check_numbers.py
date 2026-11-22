"""Re-derive every number quoted in report.tex from results/*.csv.

Fails loudly on a mismatch. The report must not be able to inherit a stale
value from README.md, which reports Phase 1 on seed 0 only.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
fails: list[str] = []


def check(label: str, got: float, want: float, tol: float = 0.05) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {label:48s} report {want:>8.2f}  data {got:>8.2f}")
    if not ok:
        fails.append(label)


def arm(tag: str) -> pd.DataFrame:
    path = R / ("sweep.csv" if not tag else f"sweep_{tag}.csv")
    d = pd.read_csv(path)
    d["granularity"] = d["granularity"].fillna("")
    return d[d["tag"].fillna("") == tag]


erm = arm("")
print("Table 1, Phase 1, 3-seed means")
b_adj = erm[(erm.method == "magnitude") & (erm.param_value == 0)].acc_adj_avg.mean()
b_wg = erm[(erm.method == "magnitude") & (erm.param_value == 0)].acc_wg.mean()
check("dense adjusted average", 100 * b_adj, 98.3)
check("dense worst group", 100 * b_wg, 73.3)

PH1 = [
    ("magnitude", 0.3, "", 97.8, 58.4),
    ("heads", 6, "", 96.6, 55.1),
    ("ffn", 0.3, "", 81.3, 22.6),
    ("quantize", 8, "per_channel", 98.3, 72.9),
    ("quantize", 6, "per_channel", 98.3, 74.2),
    ("quantize", 4, "per_channel", 96.7, 66.7),
]
for method, val, gran, adj, wg in PH1:
    s = erm[(erm.method == method) & (erm.param_value == val)
            & (erm.granularity == gran)]
    check(f"{method} {val} {gran} adj", 100 * s.acc_adj_avg.mean(), adj)
    check(f"{method} {val} {gran} worst", 100 * s.acc_wg.mean(), wg)

s30 = erm[(erm.method == "magnitude") & (erm.param_value == 0.3)]
d_adj = 100 * (b_adj - s30.acc_adj_avg.mean())
d_wg = 100 * (b_wg - s30.acc_wg.mean())
check("magnitude 30% adjusted-average drop", d_adj, 0.5)
check("magnitude 30% worst-group drop", d_wg, 14.9)
check("ratio quoted as 29x", d_wg / d_adj, 29.0, tol=1.0)
h6 = erm[(erm.method == "heads") & (erm.param_value == 6)]
check("6 heads worst-group drop", 100 * (b_wg - h6.acc_wg.mean()), 18.2)
qt4 = erm[(erm.method == "quantize") & (erm.param_value == 4)
          & (erm.granularity == "per_tensor")]
check("per-tensor 4-bit worst group", 100 * qt4.acc_wg.mean(), 8.0)

print("\nTable 2, Phase 2 dense")
for tag, wg, mean in (("spur25", 86.40, 94.69), ("spur50", 87.90, 94.35),
                      ("spur75", 87.28, 92.10)):
    d = arm(tag)
    d0 = d[(d.method == "magnitude") & (d.param_value == 0)]
    check(f"{tag} worst group", 100 * d0.acc_wg.mean(), wg, tol=0.02)
    check(f"{tag} adjusted average", 100 * d0.acc_adj_avg.mean(), mean, tol=0.02)
check("ERM worst group", 100 * b_wg, 73.31, tol=0.02)
check("ERM adjusted average", 100 * b_adj, 98.32, tol=0.02)

gains = []
for tag in ("spur25", "spur50", "spur75"):
    d = arm(tag)
    for seed in (0, 1, 2):
        p = d[(d.method == "magnitude") & (d.param_value == 0) & (d.seed == seed)]
        e = erm[(erm.method == "magnitude") & (erm.param_value == 0)
                & (erm.seed == seed)]
        gains.append(float(p.acc_wg.iloc[0]) - float(e.acc_wg.iloc[0]))
g = pd.Series(gains)
check("mean gain over ERM (pp)", 100 * g.mean(), 13.9, tol=0.2)
check("sd of that gain", 100 * g.std(), 3.1, tol=0.2)
print(f"  {'ok ' if (g > 0).all() else 'FAIL'} positive in {(g > 0).sum()}/9 arms")
if not (g > 0).all():
    fails.append("9/9 arms positive")
check("delete-50% gain quoted as 14.6", 100 * (0.8790 - b_wg), 14.6, tol=0.1)
check("all-arm mean gain quoted as 13.9", 100 * g.mean(), 13.9, tol=0.1)

print("\nSection 5.1, the collapse claim")
# The report says seeds *begin* collapsing past 80%, not that the model has
# collapsed: at 80% and 90% only one of three seeds does.
mag = erm[erm.method == "magnitude"]
for sp, want in ((0.7, 0), (0.8, 1), (0.9, 1), (0.95, 2)):
    row = mag[mag.param_value == sp]
    got = int(((row.pred_mean < 0.05) | (row.pred_mean > 0.95)).sum())
    check(f"seeds collapsed at {int(sp * 100)}% sparsity", got, want, tol=0)

print("\nTable 3, retention by usability regime")
ret = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "retention_spur*_seed*.csv")))])
live = ret[ret.retention.notna() & ~ret.collapsed]
check("non-collapsed configurations", len(live), 195, tol=0)
zero = live[live.param_value == 0]
check("uncompressed rows inside that 195", len(zero), 27, tol=0)
assert (zero.retention == 1.0).all(), "R(0) must be exactly 1 by construction"
live = live[live.param_value != 0]
check("compressed non-collapsed configurations", len(live), 168, tol=0)
for lo, hi, n, mean_r, share in ((0.40, 1.01, 87, 1.39, 95),
                                 (0.25, 0.40, 14, 1.30, 86),
                                 (-0.01, 0.25, 67, -0.20, 13)):
    s = live[(live.acc_wg_pruned >= lo) & (live.acc_wg_pruned < hi)]
    check(f"regime [{lo},{hi}) n", len(s), n, tol=0)
    check(f"regime [{lo},{hi}) mean R", s.retention.mean(), mean_r, tol=0.01)
    check(f"regime [{lo},{hi}) share R>=0.75", 100 * (s.retention >= 0.75).mean(),
          share, tol=0.6)
usable = live[live.acc_wg_pruned >= 0.40]
check("usable R minimum", usable.retention.min(), 0.29, tol=0.01)
check("usable R maximum", usable.retention.max(), 3.58, tol=0.01)
neg = live[live.retention < 0]
check("highest worst-group with negative R", neg.acc_wg_pruned.max(), 0.269, tol=0.002)

gate = pd.concat([pd.read_json(f, typ="series").to_frame().T
                  for f in sorted(glob.glob(str(R / "retention_spur*_seed*.gate.json")))])
check("smallest dense gain (pp)", 100 * gate.dense_gain.min(), 8.4, tol=0.1)
check("largest dense gain (pp)", 100 * gate.dense_gain.max(), 18.5, tol=0.1)

print("\nSection 5.3, magnitude on the delete-50% arm")
p50 = arm("spur50")
for val, want in ((0.5, 49.6),):
    e = erm[(erm.method == "magnitude") & (erm.param_value.isin([0, val]))]
    drop_e = 100 * (erm[(erm.method == "magnitude") & (erm.param_value == 0)].acc_wg.mean()
                    - erm[(erm.method == "magnitude") & (erm.param_value == val)].acc_wg.mean())
    drop_p = 100 * (p50[(p50.method == "magnitude") & (p50.param_value == 0)].acc_wg.mean()
                    - p50[(p50.method == "magnitude") & (p50.param_value == val)].acc_wg.mean())
    check("plain loses at 50% sparsity", drop_e, 49.6, tol=0.2)
    check("repaired loses at 50% sparsity", drop_p, 34.9, tol=0.2)

print("\nTable 4, per-group decomposition")
m = erm.merge(p50, on=["method", "param_value", "granularity", "seed"],
              suffixes=("_e", "_p"))
m = m[m.method == "magnitude"]
for g in range(4):
    m[f"gain_g{g}"] = m[f"acc_g{g}_p"] - m[f"acc_g{g}_e"]
m["gain_wg"] = m.acc_wg_p - m.acc_wg_e
TAB4 = {0.3: (-3.8, 3.7, 23.5, -7.1, 23.5),
        0.5: (-17.2, 5.0, 33.5, -1.0, 29.4),
        0.7: (-19.1, -11.4, 21.0, 11.0, -0.4)}
for sp, want in TAB4.items():
    row = m[m.param_value == sp]
    for i, g in enumerate(["g0", "g1", "g2", "g3"]):
        check(f"{int(sp*100)}% {g}", 100 * row[f"gain_{g}"].mean(), want[i], tol=0.1)
    check(f"{int(sp*100)}% worst group", 100 * row.gain_wg.mean(), want[4], tol=0.1)

print("\nSection 5.4, split by the sign of R")
ret2 = []
for t in ("spur25", "spur50", "spur75"):
    for sd in (0, 1, 2):
        r = pd.read_csv(R / f"retention_{t}_seed{sd}.csv")
        r["granularity"] = r["granularity"].fillna("")
        ret2.append(r.assign(tag=t, seed=sd))
ret2 = pd.concat(ret2)
pairs = []
for t in ("spur25", "spur50", "spur75"):
    d = arm(t)
    mm = erm.merge(d, on=["method", "param_value", "granularity", "seed"],
                   suffixes=("_e", "_p")).assign(tag=t)
    pairs.append(mm)
pairs = pd.concat(pairs)
j = ret2.merge(pairs, on=["tag", "seed", "method", "param_value", "granularity"])
dd = j[j.retention.notna() & ~j.collapsed & (j.param_value != 0)]
check("settings where R is defined", len(dd), 168, tol=0)
for name, half, n, wg, g2, pos in (
        ("R<0", dd[dd.retention < 0], 43, -9.0, 21.9, 34),
        ("R>=0", dd[dd.retention >= 0], 125, 17.6, 20.8, 119)):
    check(f"{name} n", len(half), n, tol=0)
    check(f"{name} mean worst-group gain",
          100 * (half.acc_wg_p - half.acc_wg_e).mean(), wg, tol=0.1)
    check(f"{name} mean g2 gain",
          100 * (half.acc_g2_p - half.acc_g2_e).mean(), g2, tol=0.1)
    check(f"{name} g2 gain positive count",
          ((half.acc_g2_p - half.acc_g2_e) > 0).sum(), pos, tol=0)
for name, half, share in (("R<0", dd[dd.retention < 0], 35),
                          ("R>=0", dd[dd.retention >= 0], 18)):
    off = ~half.worst_group_id_p.isin([1, 2])
    check(f"{name} binding group off g1/g2 (%)", 100 * off.mean(), share, tol=0.6)

print("\nSection 4.2, what the deletion rule actually removes")
import json
manifest = R / "prune_indices" / "manifest_spurious_seed0.json"
if not manifest.exists():
    print("  -- skipped: results/prune_indices/ is gitignored; run run_phase2.sh")
    print()
    if fails:
        print(f"{len(fails)} MISMATCH(ES): " + "; ".join(fails))
        sys.exit(1)
    print("every number checked reproduces from results/ (deletion checks skipped)")
    sys.exit(0)
man = json.loads(manifest.read_text())
by_tag = {e["tag"]: e for e in man}
for tag, achieved, keep in (("spur25", 64.6, 1698),
                            ("spur50", 75.6, 1170),
                            ("spur75", 86.7, 640)):
    e = by_tag[tag]
    check(f"{tag} achieved fraction (%)", 100 * e["achieved_fraction"], achieved, tol=0.1)
    check(f"{tag} images kept", e["n_keep"], keep, tol=0)
    # The report claims no counter-shortcut image is ever deleted.
    pools = e["pool_sizes"]
    assert pools == [3498, 1057], f"{tag} drew from unexpected pools {pools}"
print("  ok  deletion pools are g0 and g3 only, so g1 and g2 are untouched")

print()
if fails:
    print(f"{len(fails)} MISMATCH(ES): " + "; ".join(fails))
    sys.exit(1)
print("every number in the report reproduces from results/")
