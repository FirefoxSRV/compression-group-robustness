"""Plain ERM fine-tuning of ViT-S/16 on Waterbirds.

Deliberately group-blind. No group labels, no reweighting, no balanced sampler,
no GroupDRO. The model is SUPPOSED to learn the spurious background shortcut --
if training were balanced, worst-group accuracy would be high and the whole
experiment would measure nothing.

Phase 2 reuses this file unchanged in every respect that matters. Two additions:

  * `prune_indices` trains on a SUBSET of the training split and nothing else
    changes -- same recipe, same schedule, same selection metric -- so the ERM
    and pruned models differ in exactly one thing: the training set.
  * `score_epoch` captures per-sample predicted distributions over the
    un-augmented train cache at the end of one fixed early epoch, which is the
    input `src/prune_data.py` ranks. Capture happens only on an ERM run; a
    pruned run has no business producing difficulty scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import (
    WaterbirdsSplit,
    get_device,
    load_eval_cache,
    normalize_uint8,
    train_tf,
)
from .evaluate import evaluate_cached
from .model import create_model, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


def checkpoint_name(seed: int, tag: str | None = None) -> str:
    """ERM keeps the Phase 1 filename verbatim; a tagged run gets its own."""
    return (f"vit_s16_waterbirds_seed{seed}.pt" if not tag
            else f"vit_s16_waterbirds_{tag}_seed{seed}.pt")


@torch.no_grad()
def score_train_split(model, train_cache, device, batch_size: int = 64) -> dict:
    """Per-sample predicted distribution over the UN-AUGMENTED training split.

    `model.eval()` and the deterministic cache are both required: difficulty
    must be a property of the sample and the weights, not of which random crop
    happened to be drawn.

    Returns probs (N,2) float32 and the ICLR-2025 difficulty

        d_i = || p(w_e, x_i) - onehot(y_i) ||_2

    computed in the general form. (For 2 classes this collapses to
    sqrt(2) * (1 - p_correct), so the ranking is the obvious one -- but the
    general form is what the paper writes and what a 3-class port would need.)
    """
    was_training = model.training
    model.eval()
    images, y = train_cache["images"], train_cache["y"]
    n = len(y)

    probs = torch.empty((n, 2), dtype=torch.float32)
    for i in range(0, n, batch_size):
        x = normalize_uint8(images[i:i + batch_size], device)
        probs[i:i + batch_size] = model(x).softmax(dim=1).float().cpu()
    if device.type == "mps":
        torch.mps.synchronize()

    onehot = torch.zeros_like(probs)
    onehot[torch.arange(n), y.long()] = 1.0
    difficulty = (probs - onehot).norm(dim=1)

    if was_training:
        model.train()
    return {"probs": probs, "difficulty": difficulty, "y": y.clone(),
            "group": train_cache["group"].clone()}


def train_one_seed(
    seed: int,
    data_dir="data/waterbirds_v1.0",
    ckpt_dir="checkpoints",
    results_dir="results",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    eval_batch_size: int = 64,
    prune_indices=None,
    tag: str | None = None,
    score_epoch: int | None = None,
    scores_dir: str = "results/difficulty",
    stop_after_epoch: int | None = None,
) -> dict:
    """`stop_after_epoch` runs the REAL schedule (cosine T_max=epochs, so the
    learning rate at every step matches a full run) and exits early, writing no
    selection checkpoint.

    This is how seed 0 gets its difficulty scores without being retrained: the
    Phase 1 checkpoint already exists, MPS training is not bit-reproducible, and
    a rerun that overwrote it would silently decouple the published Phase 1
    numbers from the weights that produced them. Truncating also makes scoring
    cost two epochs instead of ten. Selection is skipped because `best val` over
    a truncated run is not the Phase 1 selection rule.
    """
    set_seed(seed)
    device = get_device()

    full_train = WaterbirdsSplit(data_dir, "train", transform=train_tf)
    n_full = len(full_train)

    if prune_indices is None:
        train_set = full_train
    else:
        keep = torch.as_tensor(prune_indices).long()
        # Phase 2 keep-sets are produced by src/prune_data.py; re-check the
        # contract here rather than trusting the caller (gate V11).
        assert keep.ndim == 1 and keep.numel() > 0, "empty keep-index set"
        assert int(keep.min()) >= 0 and int(keep.max()) < n_full, (
            f"keep index out of range for a {n_full}-sample train split")
        assert keep.unique().numel() == keep.numel(), "duplicate keep indices"
        assert keep.numel() < n_full, "keep set is the whole split; nothing pruned"
        train_set = Subset(full_train, keep.tolist())

    print(f"[seed {seed}{'/' + tag if tag else ''}] device={device} "
          f"epochs={epochs} lr={lr} bs={batch_size} "
          f"n_train={len(train_set)}/{n_full}")

    val_cache = load_eval_cache(Path(results_dir) / "val_cache.pt")

    # Difficulty scoring reads the un-augmented train cache. Only an ERM run
    # produces scores, and only if the cache was built.
    train_cache = None
    if score_epoch is not None and prune_indices is None:
        train_cache_path = Path(results_dir) / "train_cache.pt"
        if not train_cache_path.exists():
            raise FileNotFoundError(
                f"{train_cache_path} missing -- run `python -m src.data "
                "--splits train` before scoring")
        train_cache = load_eval_cache(train_cache_path)
        assert len(train_cache["y"]) == n_full, (
            f"train cache has {len(train_cache['y'])} rows, split has {n_full}")

    # No balanced/weighted sampler -- plain shuffling. num_workers=0 for MPS.
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=g, drop_last=False,
    )

    model = create_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()  # unweighted

    suffix = f"_{tag}" if tag else ""
    log_path = Path(results_dir) / f"train_log{suffix}_seed{seed}.csv"
    log_rows = []
    best = {"val_acc_avg": -1.0, "epoch": -1}
    ckpt_path = Path(ckpt_dir) / checkpoint_name(seed, tag)
    scores_path = None

    for epoch in range(epochs):
        model.train()
        t0, running, n_seen, n_correct = time.time(), 0.0, 0, 0
        for x, y, _metadata in loader:  # 3-tuple, not (x, y)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * y.size(0)
            n_correct += int((logits.argmax(1) == y).sum())
            n_seen += y.size(0)
        scheduler.step()

        val = evaluate_cached(model, val_cache, device, batch_size=eval_batch_size)
        # SELECTION METRIC: plain unweighted validation accuracy.
        # Not worst-group (a group-aware intervention that would inflate the ERM
        # baseline), and not adjusted-average either -- adjusted weighting needs
        # group labels, so using it here would leak group information too.
        # Phase 2 does NOT change this: the pruned run is selected the same way.
        val_acc_avg = val["acc_avg_plain"]

        row = {
            "epoch": epoch, "seed": seed,
            "train_loss": running / n_seen, "train_acc": n_correct / n_seen,
            "val_acc_avg": val_acc_avg, "val_acc_adj": val["acc_adj_avg"],
            "val_acc_wg": val["acc_wg"], "val_worst_group": val["worst_group_id"],
            "lr": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - t0,
        }
        log_rows.append(row)
        print(f"[seed {seed}] ep {epoch}: loss {row['train_loss']:.4f} "
              f"train_acc {row['train_acc']:.4f} | val_avg {val_acc_avg:.4f} "
              f"val_adj {val['acc_adj_avg']:.4f} val_wg {val['acc_wg']:.4f} "
              f"({row['epoch_seconds']:.0f}s)", flush=True)

        if train_cache is not None and epoch == score_epoch:
            scored = score_train_split(model, train_cache, device, eval_batch_size)
            scores_path = Path(scores_dir) / f"difficulty_seed{seed}_ep{epoch}.pt"
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**scored, "seed": seed, "score_epoch": epoch,
                        "n_train": n_full}, scores_path)
            # The epoch-e weights too, so scores can be recomputed (a different
            # difficulty definition, say) without paying for another retrain.
            save_checkpoint(model, Path(ckpt_dir) / f"vit_s16_waterbirds_seed{seed}_ep{epoch}.pt",
                            seed=seed, epoch=epoch, val_acc_avg=val_acc_avg)
            d = scored["difficulty"]
            print(f"[seed {seed}]   -> scored {len(d)} train samples at epoch "
                  f"{epoch}: d mean {d.mean():.4f} max {d.max():.4f} "
                  f"-> {scores_path}", flush=True)

        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            print(f"[seed {seed}] stopping after epoch {epoch} as requested; "
                  "no selection checkpoint written", flush=True)
            break

        if val_acc_avg > best["val_acc_avg"]:
            best = {"val_acc_avg": val_acc_avg, "epoch": epoch}
            save_checkpoint(model, ckpt_path, seed=seed, epoch=epoch,
                            val_acc_avg=val_acc_avg)
            print(f"[seed {seed}]   -> new best, saved {ckpt_path}", flush=True)

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    truncated = stop_after_epoch is not None
    if truncated:
        print(f"[seed {seed}] truncated at epoch {stop_after_epoch}; "
              f"scores at {scores_path}")
    else:
        print(f"[seed {seed}] best epoch {best['epoch']} "
              f"val_acc_avg {best['val_acc_avg']:.4f}")
    return {"seed": seed, "tag": tag, "best_epoch": best["epoch"],
            "truncated": truncated,
            "best_val_acc_avg": best["val_acc_avg"],
            "checkpoint": None if truncated else str(ckpt_path),
            "n_train": len(train_set), "n_train_full": n_full,
            "scores": str(scores_path) if scores_path else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--prune-indices", default=None,
                    help="a .pt keep-index file from src.prune_data; trains on "
                         "that subset and tags the checkpoint accordingly")
    ap.add_argument("--tag", default=None,
                    help="checkpoint/log suffix (default: inferred from "
                         "--prune-indices)")
    ap.add_argument("--score-epoch", type=int, default=None,
                    help="capture train-split difficulty at the end of this "
                         "0-indexed epoch (ERM runs only). Default: off; pass "
                         "--score to take the value from configs/base.yaml")
    ap.add_argument("--score", action="store_true",
                    help="enable scoring using prune.score_epoch from the config")
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="run the real schedule but exit after this 0-indexed "
                         "epoch, writing no selection checkpoint. Use with "
                         "--score to produce difficulty scores for a seed whose "
                         "Phase 1 checkpoint must not be overwritten")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    pcfg = cfg.get("prune", {})

    score_epoch = args.score_epoch
    if score_epoch is None and args.score:
        score_epoch = pcfg.get("score_epoch", 1)

    prune_indices, tag = None, args.tag
    if args.prune_indices:
        blob = torch.load(args.prune_indices, map_location="cpu", weights_only=False)
        prune_indices = blob["keep"] if isinstance(blob, dict) else blob
        if tag is None:
            tag = (blob.get("tag") if isinstance(blob, dict) else None) \
                  or Path(args.prune_indices).stem

    info = train_one_seed(
        args.seed, epochs=args.epochs, lr=args.lr,
        prune_indices=prune_indices, tag=tag, score_epoch=score_epoch,
        scores_dir=pcfg.get("scores_dir", "results/difficulty"),
        stop_after_epoch=args.stop_after_epoch,
    )
    suffix = f"_{tag}" if tag else ""
    out = Path("results") / f"train_summary{suffix}_seed{args.seed}.json"
    out.write_text(json.dumps(info, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
