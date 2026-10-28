#!/usr/bin/env bash
# Drive run_phase2.sh's real logic against a fake tree. No python, no MPS.
PROJ="$1"
pass=0; fail=0
check() { # check <desc> <expected> <actual>
  if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  PASS  %s\n' "$1"
  else fail=$((fail+1)); printf '  FAIL  %s\n        expected: %s\n        actual:   %s\n' "$1" "$2" "$3"; fi
}

newtree() {
  T=$(mktemp -d); mkdir -p "$T/results/difficulty" "$T/results/prune_indices" \
     "$T/results/logs" "$T/checkpoints"
  cp "$PROJ/run_phase2.sh" "$T/"
}

# Record what PY would have been asked to do, instead of doing it.
load() { # load [env assignments...]
  cd "$T"
  CALLS="$T/calls.txt"; : > "$CALLS"
  # run_phase2.sh derives VARIANT/FRACTIONS/TAG_PREFIX from configs/base.yaml
  # when they are unset. Set them here so the fake tree needs no config and the
  # tests stay about the skip/resume/failure logic, not about yaml.
  export VARIANT=global TAG_PREFIX=prune FRACTIONS="05"
  RUN_PHASE2_LIB=1 source ./run_phase2.sh
  cd "$T"          # the script cd's to dirname $0, which when sourced is the harness
  PY="record"
  record() { printf '%s\n' "$*" >> "$CALLS"; return "${RECORD_RC:-0}"; }
}
calls() { tr '\n' ';' < "$CALLS"; }

echo "T1  trained(): a checkpoint with a surviving *_last.pt is NOT complete"
newtree; load
touch checkpoints/a.pt
trained checkpoints/a.pt && r=complete || r=incomplete
check "checkpoint alone -> complete" "complete" "$r"
touch checkpoints/a_last.pt
trained checkpoints/a.pt && r=complete || r=incomplete
check "checkpoint + last.pt -> incomplete" "incomplete" "$r"
rm checkpoints/a.pt
trained checkpoints/a.pt && r=complete || r=incomplete
check "no checkpoint -> incomplete" "incomplete" "$r"

echo "T2  stage3 skips a seed with no difficulty scores, runs one that has them"
newtree; load
touch results/difficulty/difficulty_seed0_ep1.pt
SEEDS="0 1 2" FRACTIONS="05" stage3 >/dev/null 2>&1
check "only seed 0 pruned, with the variant passed through" \
  "-m src.prune_data --seed 0 --variant global;" "$(calls)"

echo "T3  stage3 skips a seed whose fractions are all already built"
newtree; load
touch results/difficulty/difficulty_seed0_ep1.pt
touch results/prune_indices/prune05_seed0.pt results/prune_indices/prune10_seed0.pt
SEEDS="0" FRACTIONS="05 10" stage3 >/dev/null 2>&1
check "nothing re-run" "" "$(calls)"

echo "T4  stage4 skips a unit with no keep-index file"
newtree; load
SEEDS="0" FRACTIONS="05" stage4 >/dev/null 2>&1
check "no index -> no work" "" "$(calls)"

echo "T5  stage4 resumes an interrupted retrain, then sweeps+measures"
newtree; load
touch results/prune_indices/prune05_seed0.pt
touch checkpoints/vit_s16_waterbirds_prune05_seed0.pt \
      checkpoints/vit_s16_waterbirds_prune05_seed0_last.pt
# The retrain is recorded but the stub does not remove last.pt, so the model
# still has not finished -- it must NOT be swept.
SEEDS="0" FRACTIONS="05" stage4 >/dev/null 2>&1
check "resumes, refuses to sweep an unfinished model" \
  "-m src.train --seed 0 --prune-indices results/prune_indices/prune05_seed0.pt --resume;" "$(calls)"

echo "T6  stage4 on a COMPLETE model skips the retrain and still sweeps"
newtree; load
touch results/prune_indices/prune05_seed0.pt
touch checkpoints/vit_s16_waterbirds_prune05_seed0.pt
SEEDS="0" FRACTIONS="05" stage4 >/dev/null 2>&1
check "sweep + retention only" \
  "-m src.sweep --seed 0 --tag prune05;-m src.retention --seed 0 --tag prune05;" "$(calls)"

echo "T7  a failing step is recorded, not fatal, and main exits non-zero"
newtree; load
touch results/difficulty/difficulty_seed0_ep1.pt
RECORD_RC=1
SEEDS="0" FRACTIONS="05" stage3 >/dev/null 2>&1
check "failure captured in FAILED[]" "1" "${#FAILED[@]}"
check "failure labelled with the variant" "prune indices seed 0 (global)" "${FAILED[0]}"
RECORD_RC=0

echo "T8  stage2 never touches seed 0"
newtree; load
SEEDS="0" stage2 >/dev/null 2>&1
check "seed 0 excluded from stage2" "" "$(calls)"

echo "T9  require_disk waits for transient pressure, gives up on a full disk"
newtree; load
# DISK_WAIT_MAX_S=0: do not actually sleep in a test. An unreachable floor must
# still end in abort rather than looping forever.
( MIN_FREE_MB=99999999 DISK_WAIT_MAX_S=0 DISK_POLL_S=0 require_disk ) >/dev/null 2>&1   && r=passed || r=aborted
check "unreachable floor -> abort" "aborted" "$r"
( MIN_FREE_MB=1 require_disk ) >/dev/null 2>&1 && r=passed || r=aborted
check "ample free space -> pass immediately" "passed" "$r"
# The bug this guard caused: aborting on pressure that clears. With a reachable
# floor it must WAIT and then proceed, not exit.
out=$( ( MIN_FREE_MB=1 DISK_WAIT_MAX_S=4 DISK_POLL_S=2 require_disk ) 2>&1 )   && r=passed || r=aborted
check "does not abort while space is available" "passed" "$r"
case "$out" in *"waiting for disk"*) r=waited ;; *) r=no-wait ;; esac
check "no spurious wait when space is fine" "no-wait" "$r"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
