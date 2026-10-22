#!/usr/bin/env bash
# Phase 2 end to end, ordered so the most informative result lands first.
#
#   ./run_phase2.sh            # everything (~9h)
#   ./run_phase2.sh stage1     # one stage (stage1 | stage2 | stage3 | stage4)
#   SEEDS="0" ./run_phase2.sh  # seed 0 only (~3h), a complete single-seed answer
#
# Every step is skipped if its output already exists, so an interrupted night
# resumes by rerunning the same command. Run it under caffeinate -- `pmset -g`
# reports sleep=1 on this machine, and a sleeping Mac is what turned a 115s
# epoch into a 3338s one in the Phase 1 log.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"
FRACTIONS="${FRACTIONS:-01 02 05 10}"
SCORE_EPOCH="${SCORE_EPOCH:-1}"
LOG_DIR=results/logs
mkdir -p "$LOG_DIR"

PY="caffeinate -i python"          # -i: no idle sleep while this runs
say() { printf '\n=== %s  [%s] ===\n' "$1" "$(date +%H:%M:%S)"; }
have() { [ -e "$1" ]; }

# -- stage 1: the train cache + difficulty scores -----------------------------
# Seed 0 is scored with --stop-after-epoch so its Phase 1 checkpoint is never
# rewritten. Seeds 1-2 do not exist yet, so they score during their real run.
stage1() {
  say "train cache"
  have results/train_cache.pt \
    && echo "results/train_cache.pt exists, skipping" \
    || $PY -m src.data --splits train 2>&1 | tee "$LOG_DIR/train_cache.log"

  say "seed 0 difficulty scores (truncated run, checkpoint untouched)"
  local s0="results/difficulty/difficulty_seed0_ep${SCORE_EPOCH}.pt"
  have "$s0" && echo "$s0 exists, skipping" || \
    $PY -m src.train --seed 0 --score --stop-after-epoch "$SCORE_EPOCH" \
      2>&1 | tee "$LOG_DIR/score_seed0.log"
}

# -- stage 2: finish Phase 1 (seeds 1 and 2) ----------------------------------
stage2() {
  for seed in $SEEDS; do
    [ "$seed" = "0" ] && continue          # seed 0 is already done
    say "ERM seed $seed: train"
    have "checkpoints/vit_s16_waterbirds_seed${seed}.pt" \
      && echo "checkpoint exists, skipping" \
      || $PY -m src.train --seed "$seed" --score 2>&1 | tee "$LOG_DIR/train_seed${seed}.log"

    say "ERM seed $seed: sweep"
    $PY -m src.sweep --seed "$seed" 2>&1 | tee "$LOG_DIR/sweep_seed${seed}.log"
  done
}

# -- stage 3: build the pruned training sets ----------------------------------
stage3() {
  for seed in $SEEDS; do
    say "prune indices seed $seed"
    $PY -m src.prune_data --seed "$seed" 2>&1 | tee "$LOG_DIR/prune_seed${seed}.log"
  done
}

# -- stage 4: retrain on each pruned set, sweep it, measure retention ---------
stage4() {
  for seed in $SEEDS; do
    for f in $FRACTIONS; do
      local tag="prune${f}"
      local ckpt="checkpoints/vit_s16_waterbirds_${tag}_seed${seed}.pt"

      say "$tag seed $seed: retrain"
      have "$ckpt" && echo "$ckpt exists, skipping" || \
        $PY -m src.train --seed "$seed" \
            --prune-indices "results/prune_indices/${tag}_seed${seed}.pt" \
            2>&1 | tee "$LOG_DIR/train_${tag}_seed${seed}.log"

      say "$tag seed $seed: sweep"
      $PY -m src.sweep --seed "$seed" --tag "$tag" \
          2>&1 | tee "$LOG_DIR/sweep_${tag}_seed${seed}.log"

      say "$tag seed $seed: retention"
      $PY -m src.retention --seed "$seed" --tag "$tag" \
          2>&1 | tee "$LOG_DIR/retention_${tag}_seed${seed}.log"
    done
  done
}

main() {
  local t0=$SECONDS
  echo "seeds: $SEEDS   fractions: $FRACTIONS   score epoch: $SCORE_EPOCH"
  df -h . | tail -1
  for stage in "${@:-stage1 stage2 stage3 stage4}"; do "$stage"; done
  say "done in $(( (SECONDS - t0) / 60 )) min"
  echo "Next: python -m src.figures --seed 0 --tag prune05"
}

main "$@"
