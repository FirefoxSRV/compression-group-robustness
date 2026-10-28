#!/usr/bin/env bash
# Phase 2 end to end, ordered so the most informative result lands first and
# built to survive one unattended night.
#
#   ./run_phase2.sh            # everything (~9h)
#   ./run_phase2.sh stage1     # one or more stages (stage0 | stage1 | ... | stage4)
#   SEEDS="0" ./run_phase2.sh  # seed 0 only (~3h), a complete single-seed answer
#   SKIP_PREFLIGHT=1 ./...     # only if you know why you are doing it
#
# Design notes, each paid for the hard way:
#
#   * stage0 is a preflight. Nine hours is far too long to discover a missing
#     cache or a damaged checkpoint at hour six, so src/preflight.py checks the
#     device, the caches, checkpoint integrity, the CSV header and the disk --
#     and refuses to start if any of it is wrong.
#
#   * The default stage ORDER is 1,3,4 then 2,3,4 -- not 1,2,3,4. Stages 3 and 4
#     skip whatever is not ready yet and do whatever is, so this order produces
#     seed 0's complete pruning answer (the gate on whether pruning buys
#     anything at all) BEFORE spending ~1.3h on the seeds 1-2 ERM baselines.
#     If the night is cut short, what survives is a result, not a prefix.
#
#   * A step that fails is recorded and skipped, not fatal. One transient
#     failure must not cost the remaining hours. Every failure is listed loudly
#     at the end and the script exits non-zero.
#
#   * Completion is `trained()`: a checkpoint AND no *_last.pt. src/train.py
#     saves best-so-far on every validation improvement, so a run killed at
#     epoch 8 leaves a healthy-looking epoch-3 checkpoint behind. Treating that
#     as done would silently seat a half-trained model in a 10-epoch slot, and
#     the sweep would just compress it without complaint.
#
#   * Run under caffeinate -- `pmset -g` reports sleep=1 on this machine, and a
#     sleeping Mac is what turned a 115s epoch into a 3338s one in the Phase 1
#     log. Every step is wrapped, and the whole script should be too.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"
SCORE_EPOCH="${SCORE_EPOCH:-1}"

# The arm list comes from configs/base.yaml, not from a second copy of the
# fractions here -- a bash list that drifts from the config silently trains the
# wrong arms. VARIANT picks the selection rule and therefore both the sparsities
# and the tag prefix ("spur25" vs "prune05"), so the two variants coexist in one
# results directory instead of overwriting each other.
VARIANT="${VARIANT:-$(python -c 'import yaml;print(yaml.safe_load(open("configs/base.yaml"))["prune"].get("variant","global"))')}"
FRACTIONS="${FRACTIONS:-$(python - "$VARIANT" <<'EOF'
import sys, yaml
c = yaml.safe_load(open("configs/base.yaml"))["prune"]
fr = c["spurious_sparsities"] if sys.argv[1] == "spurious" else c["fractions"]
print(" ".join(f"{round(f * 100):02d}" for f in fr))
EOF
)}"
TAG_PREFIX="${TAG_PREFIX:-$([ "$VARIANT" = "spurious" ] && echo spur || echo prune)}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-2048}"   # per run: 87MB ckpt + 260MB last.pt + room
DISK_WAIT_MAX_S="${DISK_WAIT_MAX_S:-3600}"   # how long to wait for swap to drain
DISK_POLL_S="${DISK_POLL_S:-120}"
LOG_DIR=results/logs
mkdir -p "$LOG_DIR"

PY="caffeinate -i python"            # -i: no idle sleep while this runs
say() { printf '\n=== %s  [%s] ===\n' "$1" "$(date +%H:%M:%S)"; }
have() { [ -e "$1" ]; }
FAILED=()

# A training run is DONE only if its checkpoint exists and its resume state is
# gone. See the header: the checkpoint alone proves nothing.
trained() { [ -e "$1" ] && [ ! -e "${1%.pt}_last.pt" ]; }

# Run one step, logging it, and survive its failure. `if` suspends set -e for
# the condition, and pipefail makes python's exit status (not tee's) decide.
step() {
  local label="$1" log="$2"; shift 2
  say "$label"
  if $PY "$@" 2>&1 | tee -a "$log"; then return 0; fi
  printf '*** FAILED: %s   (log: %s)\n' "$label" "$log" | tee -a "$log"
  FAILED+=("$label")
  return 0
}

# Running out of disk mid-write would scatter truncated checkpoints across the
# matrix, so never start a run without room. But do NOT abort on first sight of
# it: the pressure here is macOS swap, which ballooned to 33 GB during a sweep
# and was released again minutes after the process exited. Aborting on that
# transient cost 3.5 idle hours on the night of 2026-10-26 with 74% done and
# 30 GiB free by the time anyone looked. So wait for it, and only give up if it
# stays low -- a genuinely full disk will not recover while we sleep.
require_disk() {
  local free waited=0
  while :; do
    free=$(df -m . | awk 'NR==2 {print $4}')
    [ "$free" -ge "$MIN_FREE_MB" ] && return 0
    if [ "$waited" -ge "$DISK_WAIT_MAX_S" ]; then
      printf '\n*** ABORT: %s MiB free, need %s MiB, unchanged after %s min.\n' \
        "$free" "$MIN_FREE_MB" "$((DISK_WAIT_MAX_S / 60))"
      printf '    Free space and re-run; completed work is skipped.\n'
      exit 1
    fi
    printf '    waiting for disk: %s MiB free, need %s MiB (%s/%s min)\n' \
      "$free" "$MIN_FREE_MB" "$((waited / 60))" "$((DISK_WAIT_MAX_S / 60))"
    sleep "$DISK_POLL_S"
    waited=$((waited + DISK_POLL_S))
  done
}

# -- stage 0: preflight -------------------------------------------------------
stage0() {
  if [ "$SKIP_PREFLIGHT" != "0" ]; then
    say "preflight SKIPPED (SKIP_PREFLIGHT=$SKIP_PREFLIGHT)"
    return 0
  fi
  say "preflight"
  # Deliberately NOT via step(): if preflight fails, nothing else should run.
  $PY -m src.preflight --seeds $SEEDS 2>&1 | tee "$LOG_DIR/preflight.log"
}

# -- stage 1: the train cache + difficulty scores -----------------------------
# Seed 0 is scored with --stop-after-epoch so its Phase 1 checkpoint is never
# rewritten. Seeds 1-2 do not exist yet, so they score during their real run.
stage1() {
  say "train cache"
  have results/train_cache.pt \
    && echo "results/train_cache.pt exists, skipping" \
    || step "build train cache" "$LOG_DIR/train_cache.log" -m src.data --splits train

  local s0="results/difficulty/difficulty_seed0_ep${SCORE_EPOCH}.pt"
  if have "$s0"; then
    say "seed 0 difficulty scores"; echo "$s0 exists, skipping"
  else
    step "seed 0 difficulty scores (truncated run, checkpoint untouched)" \
      "$LOG_DIR/score_seed0.log" \
      -m src.train --seed 0 --score --stop-after-epoch "$SCORE_EPOCH"
  fi
}

# -- stage 2: finish Phase 1 (seeds 1 and 2) ----------------------------------
stage2() {
  for seed in $SEEDS; do
    [ "$seed" = "0" ] && continue          # seed 0 is already done
    local ckpt="checkpoints/vit_s16_waterbirds_seed${seed}.pt"
    if trained "$ckpt"; then
      say "ERM seed $seed"; echo "$ckpt complete, skipping"
    else
      require_disk
      step "ERM seed $seed: train" "$LOG_DIR/train_seed${seed}.log" \
        -m src.train --seed "$seed" --score --resume
    fi
    # The sweep resumes internally (--resume skips rows already in the CSV), so
    # re-running it costs seconds when it is already complete.
    if trained "$ckpt"; then
      require_disk
      step "ERM seed $seed: sweep" "$LOG_DIR/sweep_seed${seed}.log" \
        -m src.sweep --seed "$seed"
    else
      echo "$ckpt did not finish; not sweeping it"
    fi
  done
}

# -- stage 3: build the pruned training sets ----------------------------------
# Skips a seed whose difficulty scores do not exist yet, so this stage can run
# before AND after stage 2 and simply do more the second time.
stage3() {
  for seed in $SEEDS; do
    local scores="results/difficulty/difficulty_seed${seed}_ep${SCORE_EPOCH}.pt"
    say "prune indices seed $seed"
    if ! have "$scores"; then
      echo "$scores not present yet, skipping (stage2 produces it)"
      continue
    fi
    # select_keep_indices is a pure function of (difficulty, y, fraction) --
    # gate V10b -- so regenerating is harmless, but skipping keeps it obvious
    # that a set stage 4 already trained on has not moved underneath it.
    local all=1
    for f in $FRACTIONS; do
      have "results/prune_indices/${TAG_PREFIX}${f}_seed${seed}.pt" || all=0
    done
    if [ "$all" = "1" ]; then
      echo "all fractions present, skipping"
      continue
    fi
    step "prune indices seed $seed ($VARIANT)" \
      "$LOG_DIR/prune_${VARIANT}_seed${seed}.log" \
      -m src.prune_data --seed "$seed" --variant "$VARIANT"
  done
}

# -- stage 4: retrain on each pruned set, sweep it, measure retention ---------
# Skips a (seed, fraction) whose keep-index file is not there yet, for the same
# reason stage 3 does: this stage runs twice and does more the second time.
stage4() {
  for seed in $SEEDS; do
    for f in $FRACTIONS; do
      local tag="${TAG_PREFIX}${f}"
      local ckpt="checkpoints/vit_s16_waterbirds_${tag}_seed${seed}.pt"
      local idx="results/prune_indices/${tag}_seed${seed}.pt"

      if ! have "$idx"; then
        say "$tag seed $seed"; echo "$idx not present yet, skipping"
        continue
      fi

      if trained "$ckpt"; then
        say "$tag seed $seed: retrain"; echo "$ckpt complete, skipping"
      else
        require_disk
        step "$tag seed $seed: retrain" "$LOG_DIR/train_${tag}_seed${seed}.log" \
          -m src.train --seed "$seed" --prune-indices "$idx" --resume
      fi

      # Only sweep and measure a model that actually finished training.
      if trained "$ckpt"; then
        require_disk
        step "$tag seed $seed: sweep" "$LOG_DIR/sweep_${tag}_seed${seed}.log" \
          -m src.sweep --seed "$seed" --tag "$tag"
        require_disk
        step "$tag seed $seed: retention" "$LOG_DIR/retention_${tag}_seed${seed}.log" \
          -m src.retention --seed "$seed" --tag "$tag"
      else
        echo "$ckpt did not finish; not sweeping it"
      fi
    done
  done
}

# -- stage 5: audit every arm, and repair whatever is missing -----------------
# "Did everything run?" needs a real answer, not a reading of the log: the
# overnight run recorded 12 FAILED steps of which 11 were false positives
# (the step finished its work and died on teardown when the disk filled) and
# one was real. Only the artifacts settle it, so check those -- and re-run the
# cheap pieces that are genuinely absent.
stage5() {
  say "verify"
  $PY -m src.verify --seeds $SEEDS 2>&1 | tee "$LOG_DIR/verify.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    say "verify found gaps -- repairing"
    # A missing sweep or retention costs seconds-to-minutes; re-running them is
    # always cheaper than reasoning about whether they were needed.
    for seed in $SEEDS; do
      for f in $FRACTIONS; do
        local tag="${TAG_PREFIX}${f}"
        local ckpt="checkpoints/vit_s16_waterbirds_${tag}_seed${seed}.pt"
        trained "$ckpt" || continue
        have "results/retention_${tag}_seed${seed}.csv" && continue
        require_disk
        step "repair $tag seed $seed: sweep" "$LOG_DIR/sweep_${tag}_seed${seed}.log" \
          -m src.sweep --seed "$seed" --tag "$tag"
        step "repair $tag seed $seed: retention" "$LOG_DIR/retention_${tag}_seed${seed}.log" \
          -m src.retention --seed "$seed" --tag "$tag"
      done
    done
    say "re-verifying after repair"
    $PY -m src.verify --seeds $SEEDS 2>&1 | tee -a "$LOG_DIR/verify.log"
  fi
}

main() {
  local t0=$SECONDS
  echo "variant: $VARIANT   seeds: $SEEDS   sparsities: $FRACTIONS (tags ${TAG_PREFIX}*)   score epoch: $SCORE_EPOCH"
  df -h . | tail -1

  # NOT `for stage in "${@:-stage0 stage1 ...}"` -- a quoted ${@:-default}
  # expands the default as ONE word, so the no-argument form tried to run a
  # command literally named "stage1 stage2 stage3 stage4" and died instantly.
  local stages=("$@")
  # A plain `[ ... ] && stages=(...)` would be the obvious way to default this,
  # but under `set -e` that list returns non-zero whenever arguments ARE given,
  # which kills the script silently before it runs anything. Hence `if`.
  if [ ${#stages[@]} -eq 0 ]; then
    # 1,3,4 then 2,3,4: seed 0's complete pruning answer before the seeds 1-2
    # ERM baselines. See the header.
    stages=(stage0 stage1 stage3 stage4 stage2 stage3 stage4 stage5)
  fi
  for stage in "${stages[@]}"; do "$stage"; done

  say "done in $(( (SECONDS - t0) / 60 )) min"
  df -h . | tail -1
  if [ ${#FAILED[@]} -gt 0 ]; then
    printf '\n%s STEP(S) FAILED:\n' "${#FAILED[@]}"
    printf '  - %s\n' "${FAILED[@]}"
    printf '\nRe-run the same command: completed work is skipped and\n'
    printf 'interrupted training resumes from its *_last.pt.\n'
    exit 1
  fi
  echo "all steps succeeded"
  echo "Next: python -m src.figures --seed 0 --tag prune05"
}

# Sourceable for testing: `RUN_PHASE2_LIB=1 source run_phase2.sh` defines the
# stages and helpers without executing anything, so the skip/resume/failure
# logic can be driven against a fake tree instead of nine hours of MPS time.
[ "${RUN_PHASE2_LIB:-0}" = "1" ] || main "$@"
