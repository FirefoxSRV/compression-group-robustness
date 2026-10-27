"""Did everything actually run? Answered from artifacts, not from logs.

    python -m src.verify                  # audit every seed in the config
    python -m src.verify --seeds 0 1 2
    python -m src.verify --json            # machine-readable, for a repair pass

Exit status is 0 only when every expected artifact is present and consistent.

Why this exists. The overnight run of 2026-10-26 recorded **12 failed steps**,
of which exactly one was a real loss. The rest were steps that completed their
work and then died on teardown, because macOS swap filled the volume and the
process returned non-zero on the way out -- `tee: stdout: No space left on
device` after `wrote results/sweep_prune05.csv`. Reading the log would have
said twelve arms needed redoing; reading the artifacts said one did.

So the log records what the *runner* saw, and only the artifacts record what
exists. For each arm this checks:

  * the checkpoint exists AND has no surviving `*_last.pt` (an interrupted run
    leaves a plausible best-so-far checkpoint behind -- see run_phase2.sh)
  * the checkpoint's metadata agrees with its own training summary
  * the arm has all 29 sweep rows in the right CSV, with no duplicate configs
  * the retention CSV exists and is non-empty

A missing retention file for an arm whose sweep is complete is cheap to repair;
a missing checkpoint is a retrain. The report says which.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SWEEP_CONFIGS = 29


def _trained(ckpt: Path) -> bool:
    """One definition, in src/train.py; see is_run_complete."""
    from .train import is_run_complete
    return is_run_complete(ckpt)


def _sweep_rows(path: Path, seed: int, tag: str) -> tuple[int, int]:
    """(rows for this arm, duplicate configs within it)."""
    if not path.exists():
        return 0, 0
    seen, dups, n = set(), 0, 0
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if str(r.get("seed")) != str(seed):
                    continue
                if (r.get("tag") or "") != tag:
                    continue
                n += 1
                key = (r.get("method"), r.get("param_value"),
                       r.get("granularity") or "")
                if key in seen:
                    dups += 1
                seen.add(key)
    except Exception:
        return -1, 0
    return n, dups


def check_arm(cfg: dict, seed: int, tag: str) -> dict:
    """One arm: ERM when tag is '', otherwise a pruned arm."""
    paths = cfg["paths"]
    res, ck = Path(paths["results_dir"]), Path(paths["ckpt_dir"])
    name = f"vit_s16_waterbirds_{tag}_seed{seed}.pt" if tag \
        else f"vit_s16_waterbirds_seed{seed}.pt"
    ckpt = ck / name
    suffix = f"_{tag}" if tag else ""
    summary = res / f"train_summary{suffix}_seed{seed}.json"
    sweep_csv = res / "sweep.csv" if not tag else res / f"sweep_{tag}.csv"

    problems: list[str] = []
    repair = None

    if not ckpt.exists():
        problems.append("no checkpoint")
        repair = "retrain"
    elif not _trained(ckpt):
        problems.append("interrupted (a *_last.pt survives)")
        repair = "retrain"
    else:
        # Metadata must agree with the run that produced it. Phase 1's seed-0
        # checkpoint was once silently replaced by epoch-0 weights; preflight C3
        # is the deep version of this, and this is the cheap one.
        if summary.exists():
            try:
                import torch
                blob = torch.load(ckpt, map_location="cpu", weights_only=False)
                rec = json.loads(summary.read_text())
                if blob.get("epoch") != rec.get("best_epoch"):
                    problems.append(
                        f"checkpoint holds epoch {blob.get('epoch')} but "
                        f"{summary.name} selected {rec.get('best_epoch')}")
                    repair = "retrain"
            except Exception as e:
                problems.append(f"checkpoint unreadable ({type(e).__name__})")
                repair = "retrain"
        else:
            problems.append(f"no {summary.name}")

    n, dups = _sweep_rows(sweep_csv, seed, tag)
    if n == -1:
        problems.append(f"{sweep_csv.name} unreadable")
        repair = repair or "sweep"
    elif n < SWEEP_CONFIGS:
        problems.append(f"{n}/{SWEEP_CONFIGS} sweep rows in {sweep_csv.name}")
        repair = repair or "sweep"
    elif n > SWEEP_CONFIGS:
        problems.append(f"{n} sweep rows, expected {SWEEP_CONFIGS}")
    if dups:
        problems.append(f"{dups} duplicate configs in {sweep_csv.name}")

    if tag:   # retention compares a pruned arm against its ERM arm
        ret = res / f"retention_{tag}_seed{seed}.csv"
        if not ret.exists():
            problems.append("no retention csv")
            repair = repair or "retention"
        elif ret.stat().st_size == 0:
            problems.append("retention csv is empty")
            repair = repair or "retention"

    return {"seed": seed, "tag": tag or "(ERM)", "sweep_rows": n,
            "ok": not problems, "problems": problems, "repair": repair}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fractions", type=float, nargs="*", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    seeds = args.seeds if args.seeds is not None else list(cfg["train"]["seeds"])
    fractions = (args.fractions if args.fractions is not None
                 else list(cfg["prune"]["fractions"]))
    tags = [""] + [f"prune{int(round(f * 100)):02d}" for f in fractions]

    results = [check_arm(cfg, s, t) for s in seeds for t in tags]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'arm':<20}{'sweep':<10}{'status'}")
        for r in results:
            arm = f"{r['tag']}/seed{r['seed']}"
            sweep = f"{r['sweep_rows']}/{SWEEP_CONFIGS}"
            if r["ok"]:
                print(f"{arm:<20}{sweep:<10}ok")
            else:
                print(f"{arm:<20}{sweep:<10}INCOMPLETE -> {r['repair']}")
                for p in r["problems"]:
                    print(f"{'':<30}  - {p}")

    bad = [r for r in results if not r["ok"]]
    print()
    if not bad:
        print(f"all {len(results)} arms complete "
              f"({len(seeds)} seeds x {len(tags)} arms)")
        return 0
    print(f"{len(bad)} of {len(results)} arms incomplete: "
          + ", ".join(f"{r['tag']}/seed{r['seed']}" for r in bad))
    retrains = [r for r in bad if r["repair"] == "retrain"]
    if retrains:
        print(f"  {len(retrains)} need a retrain (~25 min each): "
              + ", ".join(f"{r['tag']}/seed{r['seed']}" for r in retrains))
    cheap = [r for r in bad if r["repair"] in ("sweep", "retention")]
    if cheap:
        print(f"  {len(cheap)} need only a sweep/retention re-run: "
              + ", ".join(f"{r['tag']}/seed{r['seed']}" for r in cheap))
    print("\nRe-run ./run_phase2.sh -- completed arms are skipped.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
