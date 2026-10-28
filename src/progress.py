"""Live progress for a Phase 2 run.

    python -m src.progress            # render once
    python -m src.progress --watch    # redraw every 60s until everything is done

Progress is measured in *estimated seconds of work*, not in units, because a
training run costs 25 minutes and a keep-index file costs one second; counting
them equally would show 50% while 95% of the night remained.

Two sources give sub-unit resolution, so a run in flight is visible rather than
simply "not finished":

  * `results/train_log*_seed*.csv` -- src/train.py appends and flushes one row
    per epoch, so the row count is the epoch count.
  * `results/sweep*.csv` -- src/sweep.py appends and flushes one row per
    configuration, so rows for a (tag, seed) out of 29 is that sweep's share.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

# Cost model, in seconds. Measured, not guessed: results/train_log_seed0.csv for
# the epoch time, results/sweep_run.log for the 29-config sweep.
EPOCHS = 10
SWEEP_CONFIGS = 29
T_EPOCH = 150
T_SWEEP_CONFIG = 1200 / SWEEP_CONFIGS
T_CACHE = 300
T_INDEX = 10
T_RETENTION = 5

BAR_W = 44


def _rows(path: Path, where=None) -> int:
    """Row count of a CSV, optionally filtered. Tolerates a partial last line."""
    if not path.exists():
        return 0
    try:
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            if where is None:
                return sum(1 for _ in rdr)
            return sum(1 for r in rdr if where(r))
    except Exception:
        return 0


def _trained(ckpt: Path) -> bool:
    """One definition, in src/train.py; see is_run_complete."""
    from .train import is_run_complete
    return is_run_complete(ckpt)


N_TRAIN_FULL = 4795


def _epoch_cost(idx_path: Path) -> float:
    """Seconds per epoch for an arm, scaled by how much data it actually trains on.

    T_EPOCH is calibrated on the full 4795-sample split. The identifiable-setting
    arms train on 640-1698 samples, so charging them a full-split epoch
    overstates the remaining work roughly threefold -- it read "6.6 h remaining"
    against a measured ~3.8 h. The keep-index file records n_keep, so use it.
    """
    if not idx_path.exists():
        return T_EPOCH
    try:
        import torch
        blob = torch.load(idx_path, map_location="cpu", weights_only=False)
        n_keep = int(blob.get("n_keep") or blob["keep"].numel())
        return T_EPOCH * max(n_keep, 1) / N_TRAIN_FULL
    except Exception:
        return T_EPOCH


class Unit:
    """One piece of work: its cost, how much is done, and how to say so."""

    def __init__(self, group: str, label: str, cost: float,
                 done: float, note: str = ""):
        self.group, self.label, self.cost = group, label, cost
        self.frac = 0.0 if cost <= 0 else max(0.0, min(1.0, done / cost))
        self.note = note

    @property
    def mark(self) -> str:
        if self.frac >= 1.0:
            return "x"
        return "~" if self.frac > 0 else " "


def collect(cfg: dict, seeds: list[int], fractions: list[float]) -> list[Unit]:
    paths = cfg["paths"]
    res = Path(paths["results_dir"])
    ck = Path(paths["ckpt_dir"])
    score_epoch = cfg["prune"]["score_epoch"]
    from .verify import arm_tags
    tags = arm_tags(cfg)
    units: list[Unit] = []

    # ---- stage 1: the train cache and seed 0's difficulty scores
    cache = Path(paths["train_cache"])
    units.append(Unit("stage 1  cache + scores", "train cache", T_CACHE,
                      T_CACHE if cache.exists() else 0,
                      f"{cache.stat().st_size // (1024*1024)} MB" if cache.exists() else ""))

    # ---- ERM arms (Phase 1 completion for seeds 1-2)
    for s in seeds:
        ckpt = ck / f"vit_s16_waterbirds_seed{s}.pt"
        log = res / f"train_log_seed{s}.csv"
        epochs = _rows(log)
        done = EPOCHS * T_EPOCH if _trained(ckpt) else epochs * T_EPOCH
        note = ""
        if _trained(ckpt):
            try:
                import torch
                b = torch.load(ckpt, map_location="cpu", weights_only=False)
                note = f"epoch {b['epoch']}  val {b['val_acc_avg']:.4f}"
            except Exception:
                note = "present"
        elif epochs:
            note = f"epoch {epochs}/{EPOCHS}"
        units.append(Unit("Phase 1  ERM arms", f"seed {s} train", EPOCHS * T_EPOCH,
                          done, note))

        n = _rows(Path(paths["sweep_csv"]),
                  lambda r: str(r.get("seed")) == str(s) and not r.get("tag"))
        units.append(Unit("Phase 1  ERM arms", f"seed {s} sweep",
                          SWEEP_CONFIGS * T_SWEEP_CONFIG, n * T_SWEEP_CONFIG,
                          f"{n}/{SWEEP_CONFIGS} configs"))

        sc = res / "difficulty" / f"difficulty_seed{s}_ep{score_epoch}.pt"
        units.append(Unit("stage 1  cache + scores", f"difficulty seed {s}",
                          T_INDEX, T_INDEX if sc.exists() else 0,
                          "" if sc.exists() else f"(arrives with ERM seed {s})"))

    # ---- Phase 2 arms
    for s in seeds:
        for tag in tags:
            idx = res / "prune_indices" / f"{tag}_seed{s}.pt"
            units.append(Unit("Phase 2  keep-index sets", f"{tag} seed {s}",
                              T_INDEX, T_INDEX if idx.exists() else 0))
    for s in seeds:
        for tag in tags:
            ckpt = ck / f"vit_s16_waterbirds_{tag}_seed{s}.pt"
            log = res / f"train_log_{tag}_seed{s}.csv"
            epochs = _rows(log)
            t_ep = _epoch_cost(res / "prune_indices" / f"{tag}_seed{s}.pt")
            done = EPOCHS * t_ep if _trained(ckpt) else epochs * t_ep
            note = "done" if _trained(ckpt) else (
                f"epoch {epochs}/{EPOCHS}" if epochs else "")
            units.append(Unit("Phase 2  pruned retrains", f"{tag} seed {s}",
                              EPOCHS * t_ep, done, note))

            n = _rows(res / f"sweep_{tag}.csv",
                      lambda r, s=s, tag=tag: str(r.get("seed")) == str(s)
                      and r.get("tag") == tag)
            units.append(Unit("Phase 2  pruned sweeps", f"{tag} seed {s}",
                              SWEEP_CONFIGS * T_SWEEP_CONFIG,
                              n * T_SWEEP_CONFIG, f"{n}/{SWEEP_CONFIGS}"))

            ret = res / f"retention_{tag}_seed{s}.csv"
            units.append(Unit("Phase 2  retention", f"{tag} seed {s}",
                              T_RETENTION, T_RETENTION if ret.exists() else 0))
    return units


def bar(frac: float, width: int = BAR_W) -> str:
    full = int(frac * width)
    return "#" * full + "-" * (width - full)


def latest_activity(res: Path) -> str:
    """The most recently touched log, and its last meaningful line."""
    # Exclude this watcher's own log: writing a frame updates its mtime, so it
    # would always be the newest file and "now:" would report itself.
    SELF = {"progress_watch.log"}
    logs = sorted((q for q in Path(res / "logs").glob("*.log")
                   if q.name not in SELF),
                  key=lambda q: q.stat().st_mtime, reverse=True) \
        if (res / "logs").exists() else []
    if not logs:
        return "no logs yet"
    newest = logs[0]
    age = time.time() - newest.stat().st_mtime
    line = ""
    try:
        for cand in reversed(newest.read_text(errors="replace").splitlines()[-40:]):
            if cand.strip():
                line = cand.strip()
                break
    except Exception:
        pass
    return f"{newest.name} ({age:.0f}s ago): {line[:110]}"


def render(cfg: dict, seeds: list[int], fractions: list[float],
           started: float | None) -> tuple[str, bool]:
    units = collect(cfg, seeds, fractions)
    total = sum(u.cost for u in units)
    done = sum(u.cost * u.frac for u in units)
    frac = done / total if total else 1.0
    remaining_h = (total - done) / 3600

    res = Path(cfg["paths"]["results_dir"])
    out = []
    out.append(f"Phase 1 + Phase 2 progress   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if started:
        el = time.time() - started
        out.append(f"elapsed {el/3600:.2f} h    work remaining ~{remaining_h:.1f} h"
                   f"    (estimate: {T_EPOCH}s/epoch at full split, scaled by "
                   f"each arm's train size; "
                   f"{int(T_SWEEP_CONFIG * SWEEP_CONFIGS // 60)}min/sweep)")
    else:
        out.append(f"work remaining ~{remaining_h:.1f} h")
    out.append("")
    out.append(f"[{bar(frac)}] {frac*100:5.1f}%   "
               f"{done/3600:.1f}/{total/3600:.1f} h of work")
    out.append("")

    # Bucket by group in first-seen order: units are generated interleaved
    # (retrain, sweep, retention per arm), so printing them in that order
    # repeats every heading.
    buckets: dict[str, list[Unit]] = {}
    for u in units:
        buckets.setdefault(u.group, []).append(u)

    # The trivial-cost groups are 24 lines that never tell you anything except
    # a count, so they collapse to one line each.
    COMPACT = {"Phase 2  keep-index sets", "Phase 2  retention"}
    for group, members in buckets.items():
        if group in COMPACT:
            n = sum(1 for m in members if m.frac >= 1.0)
            mark = "x" if n == len(members) else ("~" if n else " ")
            out.append(f"  [{mark}] {group:<28} {n}/{len(members)}")
            continue
        out.append(f"  {group}")
        for u in members:
            pct = "" if u.frac in (0.0, 1.0) else f" {u.frac*100:.0f}%"
            out.append(f"    [{u.mark}] {u.label:<22}{pct:<6} {u.note}")
    out.append("")
    out.append(f"  now: {latest_activity(res)}")

    # Anything the runner recorded as failed is the first thing to see.
    fails = []
    for log in sorted((res / "logs").glob("*.log")) if (res / "logs").exists() else []:
        try:
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith("*** FAILED:"):
                    fails.append(line)
        except Exception:
            pass
    if fails:
        # A logged failure is NOT evidence of missing work. On 2026-10-26, 12
        # steps were recorded failed and 11 of them had finished their work and
        # merely died on teardown when macOS swap filled the volume. So
        # reconcile against the artifacts before presenting any of it, or this
        # panel sends the reader off to redo eleven arms that are already done.
        from .verify import check_arm
        incomplete = []
        try:
            import yaml
            cfg = yaml.safe_load(open("configs/base.yaml"))
            tags = [""] + [f"prune{int(round(f * 100)):02d}" for f in fractions]
            incomplete = [f"{t or '(ERM)'}/seed{sd}"
                          for sd in seeds for t in tags
                          if not check_arm(cfg, sd, t)["ok"]]
        except Exception as e:
            incomplete = [f"(could not verify: {type(e).__name__})"]
        out.append("")
        if incomplete:
            out.append(f"  INCOMPLETE ARMS ({len(incomplete)}): "
                       + ", ".join(incomplete))
            out.append("  run `python -m src.verify` for the per-arm detail")
        else:
            out.append(f"  {len(set(fails))} step(s) were logged as failed, but every "
                       "arm verifies complete.")
            out.append("  Steps that finish their work and then fail on teardown "
                       "(a full disk, say) log")
            out.append("  as failures while losing nothing. The artifacts are "
                       "the authority, not the log.")
        for f in dict.fromkeys(fails):
            out.append(f"    logged: {f}")
    return "\n".join(out), frac >= 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fractions", type=float, nargs="*", default=None)
    ap.add_argument("--watch", action="store_true", help="redraw until complete")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--out", default=None,
                    help="also write each render to this file (atomically)")
    ap.add_argument("--started", type=float, default=None,
                    help="unix timestamp the run began, for the elapsed clock")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    seeds = args.seeds if args.seeds is not None else list(cfg["train"]["seeds"])
    fractions = (args.fractions if args.fractions is not None
                 else list(cfg["prune"]["fractions"]))

    while True:
        text, complete = render(cfg, seeds, fractions, args.started)
        print(text, flush=True)
        if args.out:
            p = Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(text + "\n")
            tmp.replace(p)            # atomic: never read a half-written frame
        if not args.watch or complete:
            return 0
        time.sleep(args.interval)
        print("\n" + "=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
