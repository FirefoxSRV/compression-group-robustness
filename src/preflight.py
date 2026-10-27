"""Fail-fast checks before a long unattended run.

Phase 2 is ~9 hours of MPS time across 14 training runs and 14 sweeps. Every
way it can fail is cheap to detect up front and expensive to discover at hour
six, so this module checks the lot in about a minute and refuses to let
run_phase2.sh start if anything is wrong.

    python -m src.preflight                 # everything, including the deep check
    python -m src.preflight --quick         # skip the 30s checkpoint evaluation
    python -m src.preflight --seeds 0 1 2   # plan for these seeds

Checks, in the order a failure would hurt:

  C1  device, and that MPS is really there (a silent CPU fallback is a 10x
      slowdown that turns 9 hours into days)
  C2  the dataset and all three uint8 caches, with their row and group counts
  C3  checkpoint integrity -- the one that matters most. An existing checkpoint
      is compared against the records that describe it, so a checkpoint that
      has been overwritten by a different epoch is caught BEFORE 348 sweep
      configurations are computed against it. This is not hypothetical: the
      Phase 1 seed-0 checkpoint was once replaced by epoch-0 weights, and
      nothing in the pipeline noticed.
  C4  the sweep CSV header, because src.sweep refuses to resume onto an old one
  C5  disk, against what this particular run will actually write
  C6  the work manifest: what exists, what will run, and the time it implies
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

# Measured on this project, not guessed: see results/train_log_seed0.csv for
# the per-epoch time and results/sweep_run.log for the 29-config sweep.
SECONDS_PER_EPOCH = 150          # clean epochs are 114-118s; 150 is a fair mean
EPOCHS = 10
SECONDS_PER_SWEEP = 1200         # 958s + 237s measured
MB = 1024 * 1024
CKPT_MB = 87                     # one selection checkpoint
LAST_MB = 260                    # one resume file, transient (deleted on finish)
TRAIN_CACHE_MB = 721
SWAP_HEADROOM_MB = 5 * 1024      # macOS needs room on a volume this full


class Failed(Exception):
    """A check that must stop the run."""


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def _warn(msg: str) -> None:
    print(f"  warn  {msg}")


# --------------------------------------------------------------- C1: device
def check_device() -> None:
    print("C1 device")
    if not torch.backends.mps.is_available():
        raise Failed(
            "MPS is not available, so training would fall back to CPU. That is "
            "roughly a 10x slowdown -- this schedule would take days, not "
            "hours. Fix the torch install before starting.")
    _ok(f"MPS available (torch {torch.__version__})")


# ---------------------------------------------------------- C2: data + caches
def check_data(cfg: dict, results_dir: Path, verify_train_cache: bool) -> None:
    print("C2 dataset and caches")
    data_dir = Path(cfg["paths"]["data_dir"])
    if not (data_dir / "metadata.csv").exists():
        raise Failed(f"{data_dir}/metadata.csv missing -- run python -m src.data")
    _ok(f"{data_dir}/metadata.csv")

    expected_train_groups = list(cfg["data"]["train_group_counts"])
    wanted = {"val": None, "test": None}
    if verify_train_cache:
        wanted["train"] = expected_train_groups

    for split, expect_groups in wanted.items():
        # From the config, not rebuilt from results_dir -- these keys are
        # already project-relative ("results/test_cache.pt"), so joining them
        # onto results_dir yields results/results/... and every check that
        # follows degrades to a silent skip.
        path = Path(cfg["paths"][f"{split}_cache"])
        if not path.exists():
            how = ("python -m src.data --splits train" if split == "train"
                   else "python -m src.data")
            raise Failed(f"{path} missing -- run `{how}`")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        n = len(cache["y"])
        counts = [int((cache["group"] == g).sum()) for g in range(4)]
        if min(counts) == 0:
            # An empty group makes worst-group accuracy silently NaN (gate V2).
            raise Failed(f"{path} has an empty group: {counts}")
        if expect_groups is not None and counts != expect_groups:
            raise Failed(
                f"{path} group counts {counts} != configured "
                f"{expect_groups}. Group indexing is wrong; stop everything.")
        _ok(f"{split}_cache.pt  n={n}  groups={counts}")


# ------------------------------------------------- C3: checkpoint integrity
def check_checkpoints(cfg: dict, ckpt_dir: Path, results_dir: Path,
                      seeds: list[int], deep: bool) -> None:
    """Compare every existing checkpoint against the records that describe it.

    Two independent records exist for an ERM checkpoint:
      * `train_summary_seed{s}.json` -- the epoch and val accuracy selected
      * `baseline.json`              -- the dense test metrics computed FROM it
    Metadata is free to check. The deep check reloads the weights and
    re-evaluates them, which is the only way to catch weights that were
    replaced while the metadata still looks plausible.
    """
    print("C3 checkpoint integrity")
    from .model import TIMM_NAME

    checked_any = False
    for seed in seeds:
        ckpt = ckpt_dir / f"vit_s16_waterbirds_seed{seed}.pt"
        summary = results_dir / f"train_summary_seed{seed}.json"
        if not ckpt.exists():
            _ok(f"seed {seed}: no checkpoint yet (will be trained)")
            continue
        checked_any = True
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if blob.get("timm_model") != TIMM_NAME:
            raise Failed(f"{ckpt} is a {blob.get('timm_model')}, not {TIMM_NAME}")
        if blob.get("seed") != seed:
            raise Failed(f"{ckpt} records seed {blob.get('seed')}, not {seed}")

        if not summary.exists():
            _warn(f"seed {seed}: {ckpt.name} present but no {summary.name} to "
                  f"check it against ({blob.get('epoch')=}, "
                  f"{blob.get('val_acc_avg')=})")
            continue

        rec = json.loads(summary.read_text())
        if rec.get("truncated"):
            raise Failed(
                f"{summary} describes a TRUNCATED run, which must never have "
                f"produced {ckpt}. Something wrote a checkpoint it should not "
                f"have.")
        if blob.get("epoch") != rec.get("best_epoch"):
            raise Failed(
                f"{ckpt} holds epoch {blob.get('epoch')} but {summary.name} "
                f"selected epoch {rec.get('best_epoch')}. The checkpoint has "
                f"been overwritten by a different epoch -- every Phase 1 number "
                f"computed from it is now decoupled from these weights. "
                f"Retrain seed {seed} before going further.")
        if blob.get("val_acc_avg") != rec.get("best_val_acc_avg"):
            raise Failed(
                f"{ckpt} val_acc_avg {blob.get('val_acc_avg')} != "
                f"{summary.name} best_val_acc_avg {rec.get('best_val_acc_avg')}")
        _ok(f"seed {seed}: epoch {blob['epoch']}, val_acc_avg "
            f"{blob['val_acc_avg']:.10f} -- matches {summary.name}")

    baseline = Path(cfg["paths"]["baseline_json"])
    if not (deep and checked_any and baseline.exists()):
        if deep and checked_any:
            _warn("no baseline.json; skipping the deep check")
        return

    rec = json.loads(baseline.read_text())
    ckpt = Path(rec["checkpoint"])
    if not ckpt.exists():
        raise Failed(
            f"{baseline} describes {ckpt}, which does not exist. Either the "
            f"baseline is stale or a checkpoint is missing; both mean the "
            f"published Phase 1 numbers have nothing backing them.")

    # The real test: re-evaluate the weights and see if they still produce the
    # numbers the project published. Metadata can be right while weights are
    # wrong -- any rerun writes consistent-looking metadata.
    from .data import get_device, load_eval_cache
    from .evaluate import evaluate_cached
    from .model import fresh_model

    device = get_device()
    cache = load_eval_cache(Path(cfg["paths"]["test_cache"]))
    model = fresh_model(ckpt, device)
    got = evaluate_cached(model, cache, device,
                          batch_size=cfg["eval"]["batch_size"])
    for key in ("acc_adj_avg", "acc_wg", "acc_avg_plain", "pred_mean"):
        want = rec.get(key)
        if want is None:
            continue
        if abs(got[key] - want) > 1e-9:
            raise Failed(
                f"deep check FAILED on {ckpt}: {key} is {got[key]!r} but "
                f"baseline.json recorded {want!r}. These are not the weights "
                f"that produced the published Phase 1 results.")
    _ok(f"deep check: {ckpt.name} reproduces baseline.json "
        f"(acc_wg {got['acc_wg']:.10f})")


# ------------------------------------------------------ C4: sweep CSV header
def check_sweep_csv(path: Path) -> None:
    print("C4 sweep CSV")
    from .sweep import COLUMNS

    if not path.exists():
        _ok("no sweep.csv yet (will be created)")
        return
    with open(path) as f:
        header = next(csv.reader(f), [])
    if header != list(COLUMNS):
        raise Failed(
            f"{path} has an old header, so src.sweep will refuse to resume "
            f"onto it. Run `python -m src.migrate_csv {path}` first.\n"
            f"  got:    {header}\n  expect: {list(COLUMNS)}")
    rows = sum(1 for _ in open(path)) - 1
    _ok(f"sweep.csv header current, {rows} existing rows")


# ------------------------------------------------------------- C5/C6: plan
def plan(cfg: dict, ckpt_dir: Path, results_dir: Path, seeds: list[int],
         fractions: list[float]) -> dict:
    """What this run still has to do. Drives both the disk and time estimates."""
    tags = [f"prune{int(round(f * 100)):02d}" for f in fractions]

    def done(path: Path) -> bool:
        # Mirrors run_phase2.sh's `trained()`: a checkpoint with a surviving
        # *_last.pt is an interrupted run, not a finished one.
        return path.exists() and not Path(
            str(path).replace(".pt", "_last.pt")).exists()

    erm_todo = [s for s in seeds
                if not done(ckpt_dir / f"vit_s16_waterbirds_seed{s}.pt")]
    prune_todo = [(s, t) for s in seeds for t in tags
                  if not done(ckpt_dir / f"vit_s16_waterbirds_{t}_seed{s}.pt")]
    need_train_cache = not Path(cfg["paths"]["train_cache"]).exists()

    # One sweep per arm per seed: the ERM arm plus one per prune fraction.
    sweeps = len(erm_todo) + len(prune_todo)
    train_runs = len(erm_todo) + len(prune_todo)

    disk_mb = (TRAIN_CACHE_MB if need_train_cache else 0)
    disk_mb += len(erm_todo) * CKPT_MB * 2      # selection + the epoch-e scoring copy
    disk_mb += len(prune_todo) * CKPT_MB
    disk_mb += LAST_MB                          # only one run is live at a time
    disk_mb += SWAP_HEADROOM_MB

    seconds = train_runs * EPOCHS * SECONDS_PER_EPOCH + sweeps * SECONDS_PER_SWEEP
    return {"erm_todo": erm_todo, "prune_todo": prune_todo, "tags": tags,
            "need_train_cache": need_train_cache, "train_runs": train_runs,
            "sweeps": sweeps, "disk_mb": disk_mb, "seconds": seconds}


def check_disk(results_dir: Path, need_mb: int) -> None:
    print("C5 disk")
    free_mb = shutil.disk_usage(results_dir).free // MB
    if free_mb < need_mb:
        raise Failed(
            f"{free_mb // 1024} GiB free, but this run needs about "
            f"{need_mb // 1024} GiB (artifacts plus "
            f"{SWAP_HEADROOM_MB // 1024} GiB of swap headroom). A volume with "
            f"no room to swap is what turned a 115s epoch into a 3338s one in "
            f"the Phase 1 log. Free space before starting.")
    _ok(f"{free_mb // 1024} GiB free, need about {need_mb // 1024} GiB")


def report_plan(p: dict, seeds: list[int]) -> None:
    print("C6 work remaining")
    if p["need_train_cache"]:
        _ok(f"train cache to build ({TRAIN_CACHE_MB} MB)")
    else:
        _ok("train cache present")
    _ok(f"ERM runs to train: {p['erm_todo'] or 'none'}")
    _ok(f"pruned runs to train: {len(p['prune_todo'])} "
        f"{[f'{t}/s{s}' for s, t in p['prune_todo']][:6]}"
        f"{' ...' if len(p['prune_todo']) > 6 else ''}")
    _ok(f"sweeps to run: {p['sweeps']} (29 configs each)")
    h = p["seconds"] / 3600
    _ok(f"estimated {h:.1f} h at {SECONDS_PER_EPOCH}s/epoch and "
        f"{SECONDS_PER_SWEEP // 60}min/sweep")
    if not p["train_runs"] and not p["sweeps"]:
        _warn("nothing left to do for these seeds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fractions", type=float, nargs="*", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="skip the deep checkpoint re-evaluation (~30s)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    results_dir = Path(cfg["paths"]["results_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    seeds = args.seeds if args.seeds is not None else list(cfg["train"]["seeds"])
    fractions = (args.fractions if args.fractions is not None
                 else list(cfg["prune"]["fractions"]))

    print(f"preflight: seeds={seeds} fractions={fractions}\n")
    try:
        p = plan(cfg, ckpt_dir, results_dir, seeds, fractions)
        check_device()
        check_data(cfg, results_dir,
                   verify_train_cache=not p["need_train_cache"])
        check_checkpoints(cfg, ckpt_dir, results_dir, seeds, deep=not args.quick)
        check_sweep_csv(Path(cfg["paths"]["sweep_csv"]))
        check_disk(results_dir, p["disk_mb"])
        report_plan(p, seeds)
    except Failed as e:
        print(f"\nPREFLIGHT FAILED\n\n{e}\n")
        return 1
    print("\npreflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
