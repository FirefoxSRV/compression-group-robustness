"""Run every compression configuration and record group-aware accuracy.

The experimental design in one line: compress a TRAINED checkpoint, evaluate
immediately, never retrain. Retraining would let the model re-fit the shortcut
and would answer a different question.

Two invariants this file exists to protect:

  1. Every configuration starts from `fresh_model(ckpt)`. Compressing an
     already-compressed model compounds the damage and yields smooth, plausible,
     completely wrong curves -- the single most dangerous bug available here,
     because nothing about the output looks wrong. `_fingerprint` re-checks the
     source model before every config rather than trusting the control flow.
  2. Rows are appended and flushed as they finish, so an interrupted run
     (this is ~30 forward passes over 5,794 test images on MPS) resumes instead
     of restarting. `--resume` skips configs already present in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml

from .compress import PARAM_NAME, REGISTRY
from .data import get_device, load_eval_cache
from .evaluate import evaluate_cached
from .metrics import N_GROUPS
from .model import EXCLUDED_DESCRIPTION, fresh_model, n_targeted_params

METRIC_COLUMNS = [
    "acc_avg_plain", "acc_adj_avg", "acc_wg", "worst_group_id", "gap_adj",
    *[f"acc_g{g}" for g in range(N_GROUPS)],
    *[f"n_g{g}" for g in range(N_GROUPS)],
    "pred_mean", "eval_seconds",
]
COLUMNS = [
    "tag", "method", "param_name", "param_value", "granularity",
    "n_params_targeted", "n_params_zeroed", "achieved_sparsity",
    *METRIC_COLUMNS,
    "seed", "checkpoint", "excluded",
]


def load_config(path="configs/base.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid(cfg: dict) -> list[dict]:
    """The full configuration list, in run order.

    Baselines first (sparsity 0.0, 0 heads, fraction 0.0): if a no-op config
    fails to reproduce the uncompressed numbers, the harness is broken and
    there is no point spending an hour on the rest of the grid.
    """
    s = cfg["sweep"]
    grid = [
        *[{"method": "magnitude", "params": {"sparsity": v}} for v in s["magnitude_sparsity"]],
        *[{"method": "heads", "params": {
            "n_heads_removed": v,
            "min_heads_per_block": cfg["compress"]["min_heads_per_block"],
        }} for v in s["heads_removed"]],
        *[{"method": "ffn", "params": {"neuron_fraction": v}} for v in s["ffn_fraction"]],
        *[{"method": "quantize", "params": {"bits": b, "granularity": g}}
          for g in s["quantize_granularity"] for b in s["quantize_bits"]],
    ]
    # No-op configs to the front, cheapest first.
    return sorted(grid, key=lambda c: _param_value(c) not in (0, 0.0))


def _param_value(config: dict):
    return config["params"][PARAM_NAME[config["method"]]]


def _key(method: str, param_value, granularity, tag, seed, checkpoint) -> tuple:
    """Identity of a row, for resume. Stringified so a CSV read-back matches.

    The arm, seed and checkpoint are part of the identity. Leaving them out --
    as this did through Phase 1, when exactly one checkpoint existed -- means a
    second seed reads the first seed's rows, calls all 29 configs done, and
    exits having computed nothing, with a CSV that looks complete.
    """
    return (str(tag or ""), str(method), str(param_value), str(granularity or ""),
            str(seed), str(checkpoint))


@torch.no_grad()
def _fingerprint(model) -> tuple[int, float]:
    """Cheap proof the source model is still pristine: zero count + weight sum.

    Guards invariant 1 from the module docstring. A compression method that
    breaks its no-mutation contract shifts both numbers immediately.
    """
    from .model import count_zeros, targeted_named_weights
    total = sum(float(p.detach().double().sum()) for _, p in targeted_named_weights(model))
    return count_zeros(model), total


def run_one(source, config, cache, device, batch_size, seed, ckpt_path, tag="") -> dict:
    method = config["method"]
    params = config["params"]
    compressed, stats = REGISTRY[method](source, **params)
    compressed = compressed.to(device).eval()

    metrics = evaluate_cached(compressed, cache, device, batch_size=batch_size)
    # Free the copy before the next config; two ViT-S on MPS is avoidable pressure.
    del compressed

    return {
        "tag": tag,
        "method": method,
        "param_name": PARAM_NAME[method],
        "param_value": _param_value(config),
        "granularity": params.get("granularity", ""),
        "n_params_targeted": stats["n_params_targeted"],
        "n_params_zeroed": stats["n_params_zeroed"],
        "achieved_sparsity": stats["achieved_sparsity"],
        **{k: metrics[k] for k in METRIC_COLUMNS},
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "excluded": EXCLUDED_DESCRIPTION,
    }


def default_paths(cfg: dict, seed: int, tag: str | None) -> dict:
    """Where this arm's checkpoint, CSV and baseline live.

    Untagged seed 0 keeps the exact Phase 1 filenames, so existing artifacts,
    figures and the README stay valid.
    """
    paths = cfg["paths"]
    stem = f"{tag}_seed{seed}" if tag else f"seed{seed}"
    legacy = (not tag) and seed == 0
    return {
        "ckpt": Path(paths["ckpt_dir"]) / (
            f"vit_s16_waterbirds_{tag}_seed{seed}.pt" if tag
            else f"vit_s16_waterbirds_seed{seed}.pt"),
        "csv": Path(paths["sweep_csv"]) if not tag
               else Path(paths["results_dir"]) / f"sweep_{tag}.csv",
        "baseline": Path(paths["baseline_json"]) if legacy
                    else Path(paths["results_dir"]) / f"baseline_{stem}.json",
    }


def run_sweep(
    seed: int = 0,
    config_path: str = "configs/base.yaml",
    resume: bool = True,
    methods: list[str] | None = None,
    checkpoint: str | Path | None = None,
    out_csv: str | Path | None = None,
    baseline_json: str | Path | None = None,
    tag: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    device = get_device()

    defaults = default_paths(cfg, seed, tag)
    ckpt_path = Path(checkpoint) if checkpoint else defaults["ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    cache = load_eval_cache(paths["test_cache"])
    print(f"device={device}  checkpoint={ckpt_path}  test n={len(cache['y'])}")

    # ONE source model, held on CPU, deep-copied per config by every `apply`.
    source = fresh_model(ckpt_path, torch.device("cpu"))
    baseline_fp = _fingerprint(source)
    print(f"targeted params: {n_targeted_params(source):,}  fingerprint={baseline_fp}")

    # The uncompressed reference every delta in the paper is measured against.
    baseline_path = Path(baseline_json) if baseline_json else defaults["baseline"]
    if not (resume and baseline_path.exists()):
        base_metrics = evaluate_cached(source.to(device), cache, device,
                                       batch_size=cfg["eval"]["batch_size"])
        source = source.to("cpu")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(
            {"seed": seed, "tag": tag or "", "checkpoint": str(ckpt_path),
             **base_metrics}, indent=2))
        print(f"baseline: adj_avg {base_metrics['acc_adj_avg']:.4f} "
              f"wg {base_metrics['acc_wg']:.4f} gap {base_metrics['gap_adj']:.4f}")

    grid = build_grid(cfg)
    if methods:
        grid = [c for c in grid if c["method"] in methods]

    out_path = Path(out_csv) if out_csv else defaults["csv"]
    done = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != COLUMNS:
                raise SystemExit(
                    f"{out_path} has an old header; run `python -m src.migrate_csv "
                    f"{out_path}` or pass --out to write a new file")
            for row in reader:
                done.add(_key(row["method"], row["param_value"], row["granularity"],
                              row["tag"], row["seed"], row["checkpoint"]))
        print(f"resume: {len(done)} rows already in {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (resume and out_path.exists())
    t_start = time.time()

    with open(out_path, "a" if not write_header else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
            f.flush()

        for i, config in enumerate(grid, 1):
            value = _param_value(config)
            gran = config["params"].get("granularity", "")
            key = _key(config["method"], value, gran, tag, seed, ckpt_path)
            label = f"{config['method']}={value}" + (f"/{gran}" if gran else "")
            if key in done:
                print(f"[{i}/{len(grid)}] {label}: skipped (done)")
                continue

            assert _fingerprint(source) == baseline_fp, (
                f"source model mutated before {label} -- a compression method "
                "broke its no-mutation contract; every row after this is suspect"
            )
            row = run_one(source, config, cache, device,
                          cfg["eval"]["batch_size"], seed, ckpt_path, tag or "")
            writer.writerow(row)
            f.flush()  # survive an interrupt with the row already on disk
            print(f"[{i}/{len(grid)}] {label}: sparsity {row['achieved_sparsity']:.3f} "
                  f"adj_avg {row['acc_adj_avg']:.4f} wg {row['acc_wg']:.4f} "
                  f"gap {row['gap_adj']:.4f} ({row['eval_seconds']:.0f}s)", flush=True)

    print(f"wrote {out_path} in {time.time() - t_start:.0f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="subset of: magnitude heads ffn quantize")
    ap.add_argument("--no-resume", action="store_true",
                    help="rewrite sweep.csv from scratch")
    ap.add_argument("--checkpoint", default=None,
                    help="sweep this checkpoint instead of the seed default")
    ap.add_argument("--tag", default=None,
                    help="training arm, e.g. prune05. Selects the default "
                         "checkpoint/CSV/baseline names and is recorded in "
                         "every row")
    ap.add_argument("--out", default=None, help="sweep CSV path")
    ap.add_argument("--baseline", default=None, help="baseline JSON path")
    args = ap.parse_args()
    run_sweep(seed=args.seed, config_path=args.config,
              resume=not args.no_resume, methods=args.methods,
              checkpoint=args.checkpoint, out_csv=args.out,
              baseline_json=args.baseline, tag=args.tag)


if __name__ == "__main__":
    main()
