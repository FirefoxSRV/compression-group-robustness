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

Phase 2 runs 14 of these back to back on a fanless laptop, which adds two
operational concerns that are not about the science:

  * `--resume` continues a killed run from a per-epoch `*_last.pt`, restoring
    the optimizer, the scheduler and every RNG stream -- see `_RunState`.
  * a truncated (`--stop-after-epoch`) run writes its log and summary under
    their own names, so scoring seed 0 cannot clobber the Phase 1 record.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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

# Fixed column order for the per-epoch log. Pinned rather than taken from
# `log_rows[0].keys()` because the log is now appended to across a resume, and a
# reordered header halfway down a CSV is not a CSV.
LOG_FIELDS = (
    "epoch", "seed", "train_loss", "train_acc", "val_acc_avg", "val_acc_adj",
    "val_acc_wg", "val_worst_group", "lr", "epoch_seconds",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


def checkpoint_name(seed: int, tag: str | None = None) -> str:
    """ERM keeps the Phase 1 filename verbatim; a tagged run gets its own."""
    return (f"vit_s16_waterbirds_seed{seed}.pt" if not tag
            else f"vit_s16_waterbirds_{tag}_seed{seed}.pt")


def run_suffix(tag: str | None = None, stop_after_epoch: int | None = None) -> str:
    """Filename suffix for a run's log and summary.

    A truncated run is a SCORING run, not a training record, so it gets its own
    filenames. Without this, `--seed 0 --score --stop-after-epoch 1` (exactly how
    run_phase2.sh scores seed 0) would open `results/train_log_seed0.csv` in "w"
    and replace the 10-epoch Phase 1 log with two rows -- the same log whose
    epoch timings README.md and explain/understand.tex both quote.
    """
    s = f"_{tag}" if tag else ""
    if stop_after_epoch is not None:
        s += f"_ep{stop_after_epoch}"
    return s


def last_checkpoint_path(ckpt_dir, seed: int, tag: str | None = None) -> Path:
    """The per-epoch resume checkpoint, beside the selection checkpoint."""
    return Path(ckpt_dir) / checkpoint_name(seed, tag).replace(".pt", "_last.pt")


def is_run_complete(ckpt_path) -> bool:
    """Did the run that owns this checkpoint finish its whole schedule?

    A checkpoint on its own proves nothing: `should_save_selection` writes
    best-so-far on every validation improvement, so a run killed at epoch 8
    leaves a healthy-looking epoch-3 checkpoint. `*_last.pt` is unlinked only
    after the final epoch, so its ABSENCE is the completion marker.

    run_phase2.sh's `trained()` is the shell twin of this, and gate V14 pins
    them together. src/progress.py and src/verify.py both call this rather than
    reimplementing it -- there were briefly three copies.
    """
    ckpt_path = Path(ckpt_path)
    last = Path(str(ckpt_path).replace(".pt", "_last.pt"))
    return ckpt_path.exists() and not last.exists()


def should_save_selection(truncated: bool, val_acc_avg: float, best_val: float) -> bool:
    """Whether this epoch should overwrite the selection checkpoint.

    A truncated run must answer False at EVERY epoch, not just at the epoch it
    stops on. The original guard was `if epoch >= stop_after_epoch: break`
    placed above the save, which left epoch 0 of a `stop_after_epoch=1` run
    falling straight through -- and that overwrote the Phase 1 seed-0
    checkpoint with epoch-0 weights (val_acc_avg 0.8616 in place of 0.9199).
    Selection over a truncated run is not the Phase 1 selection rule, so there
    is no epoch at which saving is correct. Gate V13.
    """
    return (not truncated) and val_acc_avg > best_val


def describe_checkpoint(path) -> str:
    """The metadata of a checkpoint, for guard messages. Never raises."""
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
        return (f"seed {c.get('seed')}, epoch {c.get('epoch')}, "
                f"val_acc_avg {c.get('val_acc_avg')}")
    except Exception as e:                      # truncated, missing, not a ckpt
        return f"unreadable ({type(e).__name__}: {e})"


def _keep_fingerprint(keep: torch.Tensor | None) -> str | None:
    """Identity of a keep-index set, so a resume cannot cross prune arms."""
    if keep is None:
        return None
    return hashlib.sha256(keep.to(torch.int64).numpy().tobytes()).hexdigest()[:16]


def _rng_state() -> dict:
    """Every RNG stream that affects training, so a resume is not a new run.

    MPS has no entry here on purpose: timm's `vit_small_patch16_224` is built
    with drop_rate=drop_path_rate=attn_drop=0, so there is no stochastic op on
    the device. Augmentation (`RandomResizedCrop`, flip) draws from the CPU
    torch RNG -- in the main process when num_workers=0, and in workers seeded
    from `loader.generator` when it is not, which is why `g` is saved too.
    """
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


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
    num_workers: int = 0,
    resume: bool = False,
    allow_overwrite: bool = False,
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

    `resume` continues from `*_last.pt` if one is there for THIS run. The
    identity check is deliberately strict -- `src/sweep.py:80` records what a
    resume key too narrow to tell two runs apart already cost this project once.
    """
    set_seed(seed)
    device = get_device()

    full_train = WaterbirdsSplit(data_dir, "train", transform=train_tf)
    n_full = len(full_train)

    keep = None
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
          f"epochs={epochs} lr={lr} bs={batch_size} workers={num_workers} "
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

    # num_workers defaults to 0, which reproduces Phase 1's augmentation stream
    # BIT-FOR-BIT (seed 0 epoch-0 loss 0.20122195619534278, verified).
    #
    # The decode-bound theory said workers should help: 4795 images at 117s is
    # ~41 img/s on the main thread. Measured on this machine, they did not --
    # 189s at workers=4 against 125-208s across four workers=0 runs of the same
    # epoch. The worker run sits inside the single-threaded spread, so there is
    # no effect to buy, and the 2022 "num_workers=0 for MPS" note is beside the
    # point either way. Since the only argument for workers was speed and the
    # speed is not there, 0 stays the default and Phase 1 comparability is free.
    #
    # The flag is kept because the spread itself is the finding: epoch time on a
    # fanless M2 is set by thermal and disk state, not by this knob. Worth
    # re-measuring on a machine that is not at 98% disk. Note that any nonzero
    # value shifts which augmentation draw each sample gets (workers seed from
    # `loader.generator`), so it is not bit-comparable with the numbers above --
    # at workers=4 epoch-0 loss was 0.2005, a 0.0007 shift.
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=g, drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = create_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()  # unweighted

    suffix = run_suffix(tag, stop_after_epoch)
    log_path = Path(results_dir) / f"train_log{suffix}_seed{seed}.csv"
    log_rows: list[dict] = []
    best = {"val_acc_avg": -1.0, "epoch": -1}
    ckpt_path = Path(ckpt_dir) / checkpoint_name(seed, tag)
    last_path = last_checkpoint_path(ckpt_dir, seed, tag)
    scores_path = None
    start_epoch = 0
    truncated = stop_after_epoch is not None

    # What makes two runs the same run. num_workers is in here because it moves
    # the augmentation stream: resuming across a change of it would silently
    # splice two different training distributions together.
    identity = {
        "seed": seed, "tag": tag, "epochs": epochs, "lr": lr,
        "weight_decay": weight_decay, "batch_size": batch_size,
        "score_epoch": score_epoch, "stop_after_epoch": stop_after_epoch,
        "num_workers": num_workers, "keep": _keep_fingerprint(keep),
    }

    state = None
    # Irreplaceable-artifact guard. Checkpoints are gitignored, are not backed
    # up anywhere, and cost 20-40 minutes of MPS time each to rebuild -- and the
    # Phase 1 seed-0 checkpoint was destroyed exactly once by a run that had no
    # business writing it. A FRESH run therefore refuses to start on top of an
    # existing selection checkpoint. This never fires under run_phase2.sh, which
    # skips completed runs and passes --resume to incomplete ones.
    if not truncated and not resume and ckpt_path.exists() and not allow_overwrite:
        raise SystemExit(
            f"{ckpt_path} exists and this is a fresh run, which would overwrite "
            f"it.\n  it currently holds: {describe_checkpoint(ckpt_path)}\n"
            f"  to continue an interrupted run:  --resume\n"
            f"  to deliberately retrain it:      --allow-overwrite")

    if resume and last_path.exists():
        # A corrupt last.pt must not be fatal. It means the process died inside
        # torch.save; the correct response is to retrain this one run, not to
        # abort the remaining hours of the schedule.
        try:
            state = torch.load(last_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[seed {seed}] {last_path} is unreadable ({type(e).__name__}: "
                  f"{e}); discarding it and training from epoch 0", flush=True)
            last_path.unlink(missing_ok=True)

    if state is not None:
        if state.get("identity") != identity:
            raise SystemExit(
                f"{last_path} belongs to a different run.\n"
                f"  on disk: {state.get('identity')}\n"
                f"  asked for: {identity}\n"
                "Delete it to start fresh, or fix the arguments. Resuming "
                "across a changed recipe produces a plausible model trained on "
                "a schedule that never existed.")
        model.load_state_dict({k: v.to(device) for k, v in state["model"].items()})
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        g.set_state(state["generator"])
        _set_rng_state(state["rng"])
        log_rows = state["log_rows"]
        best = state["best"]
        scores_path = Path(state["scores_path"]) if state["scores_path"] else None
        start_epoch = state["epoch"] + 1
        print(f"[seed {seed}] resuming {last_path} at epoch {start_epoch} "
              f"(best so far: epoch {best['epoch']} "
              f"val_acc_avg {best['val_acc_avg']:.4f})", flush=True)

        # Scoring fires on one fixed epoch. If that epoch is already behind us
        # and no scores file survived, a resume would run to completion and
        # leave stage 3 with nothing to rank -- fail now instead.
        if (train_cache is not None and score_epoch is not None
                and start_epoch > score_epoch
                and not (scores_path and scores_path.exists())):
            raise SystemExit(
                f"resume is past score_epoch={score_epoch} but no scores file "
                f"exists ({scores_path}). Difficulty scoring would be skipped "
                f"silently. Delete {last_path} and rerun from epoch 0.")
        if start_epoch >= epochs:
            print(f"[seed {seed}] already finished {epochs} epochs; nothing to do",
                  flush=True)
    elif resume:
        print(f"[seed {seed}] no {last_path}; starting from epoch 0", flush=True)

    # Append-and-flush per epoch, the pattern src/sweep.py:208-233 already uses:
    # an interrupt now leaves every epoch that did finish on disk. On a resume
    # the file is rewritten from `log_rows` first, because last.pt -- not a CSV
    # that may hold rows from an abandoned attempt -- is the authority.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=list(LOG_FIELDS))
        writer.writeheader()
        writer.writerows(log_rows)
        logf.flush()

        for epoch in range(start_epoch, epochs):
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
            # Release the MPS caching allocator's pool each epoch. `del` alone
            # does not: MPS keeps freed blocks reserved, and on 16 GB macOS
            # covers the growth with swap on the data volume -- which hit 33 GB
            # and filled the disk twice on 2026-10-26. Cheap, and it keeps the
            # footprint flat across ten epochs.
            if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            # SELECTION METRIC: plain unweighted validation accuracy.
            # Not worst-group (a group-aware intervention that would inflate the
            # ERM baseline), and not adjusted-average either -- adjusted
            # weighting needs group labels, so using it here would leak group
            # information too. Phase 2 does NOT change this: the pruned run is
            # selected the same way.
            val_acc_avg = val["acc_avg_plain"]

            row = {
                "epoch": epoch, "seed": seed,
                "train_loss": running / n_seen, "train_acc": n_correct / n_seen,
                "val_acc_avg": val_acc_avg, "val_acc_adj": val["acc_adj_avg"],
                "val_acc_wg": val["acc_wg"], "val_worst_group": val["worst_group_id"],
                "lr": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - t0,
            }
            log_rows.append(row)
            writer.writerow(row)
            logf.flush()
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
                # The epoch-e weights too, so scores can be recomputed (a
                # different difficulty definition, say) without paying for
                # another retrain.
                save_checkpoint(model, Path(ckpt_dir) / f"vit_s16_waterbirds_seed{seed}_ep{epoch}.pt",
                                seed=seed, epoch=epoch, val_acc_avg=val_acc_avg)
                d = scored["difficulty"]
                print(f"[seed {seed}]   -> scored {len(d)} train samples at epoch "
                      f"{epoch}: d mean {d.mean():.4f} max {d.max():.4f} "
                      f"-> {scores_path}", flush=True)
                # Scoring happens once, at one fixed epoch. Holding the 688 MB
                # train cache for the remaining epochs buys nothing.
                del scored, train_cache
                train_cache = None
                gc.collect()
                if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

            # A truncated run writes NO selection checkpoint and NO resume
            # state, at any epoch. Both guards are on `truncated`, not on the
            # break below, because the break only fires at
            # `epoch >= stop_after_epoch`: with stop_after_epoch=1, epoch 0 fell
            # straight through to the save and overwrote the very Phase 1
            # checkpoint that --stop-after-epoch exists to protect. It did
            # exactly that once; `epoch 0, val_acc_avg 0.8615` replaced
            # `epoch 3, val_acc_avg 0.9199`. Selection over a truncated run is
            # not the Phase 1 selection rule anyway, so there is nothing to
            # save -- and two epochs are not worth a 256 MB resume file.
            if should_save_selection(truncated, val_acc_avg, best["val_acc_avg"]):
                best = {"val_acc_avg": val_acc_avg, "epoch": epoch}
                save_checkpoint(model, ckpt_path, seed=seed, epoch=epoch,
                                val_acc_avg=val_acc_avg)
                print(f"[seed {seed}]   -> new best, saved {ckpt_path}", flush=True)

            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                print(f"[seed {seed}] stopping after epoch {epoch} as requested; "
                      "no selection checkpoint written", flush=True)
                break

            # Resume point. Written AFTER the selection checkpoint so the two
            # can never disagree about which epoch produced `best`. The
            # optimizer and scheduler are not optional: AdamW's moments and the
            # cosine position both change the next step.
            if truncated:
                continue
            # Atomic, like save_checkpoint: a kill partway through writing
            # 256 MB would leave a file that `--resume` cannot read, turning a
            # recoverable interrupt into a lost run.
            last_path.parent.mkdir(parents=True, exist_ok=True)
            last_tmp = last_path.with_suffix(".tmp")
            torch.save({
                "identity": identity,
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator": g.get_state(),
                "rng": _rng_state(),
                "epoch": epoch,
                "best": best,
                "log_rows": log_rows,
                "scores_path": str(scores_path) if scores_path else None,
            }, last_tmp)
            last_tmp.replace(last_path)

    if truncated:
        print(f"[seed {seed}] truncated at epoch {stop_after_epoch}; "
              f"scores at {scores_path}")
    else:
        print(f"[seed {seed}] best epoch {best['epoch']} "
              f"val_acc_avg {best['val_acc_avg']:.4f}")

    # The run finished; the resume state is now just 1.3 GB of dead weight on a
    # disk this project does not have to spare.
    last_path.unlink(missing_ok=True)

    return {"seed": seed, "tag": tag, "best_epoch": best["epoch"],
            "truncated": truncated,
            "best_val_acc_avg": best["val_acc_avg"],
            "checkpoint": None if truncated else str(ckpt_path),
            "n_train": len(train_set), "n_train_full": n_full,
            "log": str(log_path),
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
    ap.add_argument("--num-workers", type=int, default=0,
                    help="dataloader workers. 0 (the default) reproduces "
                         "the Phase 1 augmentation stream bit-for-bit; >0 "
                         "shifts it and measured no faster here")
    ap.add_argument("--resume", action="store_true",
                    help="continue from this run's *_last.pt if it exists, "
                         "restoring optimizer, scheduler and RNG state")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="permit a fresh run to overwrite an existing selection "
                         "checkpoint. Required to deliberately retrain a seed")
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
        num_workers=args.num_workers, resume=args.resume,
        allow_overwrite=args.allow_overwrite,
    )
    # Same suffix rule as the log: a truncated scoring run must not overwrite
    # the summary of the full run that produced the published checkpoint.
    suffix = run_suffix(tag, args.stop_after_epoch)
    out = Path("results") / f"train_summary{suffix}_seed{args.seed}.json"
    out.write_text(json.dumps(info, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
