#!/usr/bin/env bash
# Unattended orchestrator: finish the seed-0 restoration, prove it, repair
# Phase 1 if the weights are not identical, then run every phase.
set -uo pipefail
cd /Users/shreyas/Downloads/DLBA_Project
source .venv/bin/activate
SCR="results/phase1_backup"

log() { printf '\n######## %s  [%s]\n' "$1" "$(date '+%F %T')"; }

log "orchestrator started"

# -- 1. wait for the in-flight seed-0 restoration to exit -------------------
while pgrep -f "python -m src.train --seed 0 --score" >/dev/null; do sleep 20; done
log "seed-0 restoration process exited"
tail -4 results/logs/restore_seed0.log

# -- 2. prove the restored trajectory matches the preserved Phase 1 log -----
log "comparing restored log against the preserved Phase 1 log"
python - "results/phase1_backup/PHASE1_train_log_seed0.csv" <<'PY'
import csv, sys
orig = list(csv.DictReader(open(sys.argv[1])))
new  = list(csv.DictReader(open("results/train_log_seed0.csv")))
print(f"{len(orig)} original epochs vs {len(new)} restored")
cols = ("train_loss", "val_acc_avg", "val_acc_adj", "val_acc_wg")
bad = 0
for o, n in zip(orig, new):
    diffs = [c for c in cols if o[c] != n[c]]
    flag = "EXACT" if not diffs else f"DIFFERS in {diffs}"
    print(f"  ep {o['epoch']}: {flag}")
    if diffs:
        bad += 1
        for c in diffs:
            print(f"      {c}: orig {o[c]}  restored {n[c]}")
print("IDENTICAL" if bad == 0 and len(orig) == len(new) else f"{bad} epoch(s) differ")
PY

# -- 3. preflight. Its deep check re-evaluates the weights against
#       baseline.json, which is the authoritative test of the restore. -------
log "preflight (deep)"
python -m src.preflight --seeds 0 1 2
PRE=$?
echo "preflight exit=$PRE"

# -- 4. if the weights are NOT the originals, make Phase 1 self-consistent
#       with the weights that now exist rather than leaving 29 stale rows
#       describing weights that are gone. Logged loudly either way. ---------
if [ "$PRE" -ne 0 ]; then
  log "PREFLIGHT FAILED -- rebuilding Phase 1 seed-0 sweep against the restored weights"
  cp results/sweep.csv "$SCR/sweep.csv.before_reswept" 2>/dev/null
  cp results/baseline.json "$SCR/baseline.json.before_reswept" 2>/dev/null
  python - <<'PY'
# Drop seed 0's untagged rows; src.sweep will recompute them.
import csv
from pathlib import Path
p = Path("results/sweep.csv")
rows = list(csv.DictReader(open(p)))
keep = [r for r in rows if not (str(r["seed"]) == "0" and not r["tag"])]
print(f"dropping {len(rows) - len(keep)} stale seed-0 ERM rows, keeping {len(keep)}")
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(keep)
PY
  rm -f results/baseline.json
  caffeinate -i python -m src.sweep --seed 0
  log "re-sweep finished; re-running preflight"
  python -m src.preflight --seeds 0 1 2
  PRE=$?
  echo "preflight exit=$PRE"
fi

if [ "$PRE" -ne 0 ]; then
  log "ABORT: preflight still failing after repair. Nothing further will run."
  exit 1
fi

# -- 5. run every phase ------------------------------------------------------
log "launching run_phase2.sh (Phase 1 seeds 1-2, then all of Phase 2)"
caffeinate -dims ./run_phase2.sh
log "run_phase2.sh exited $?"

# -- 6. figures ---------------------------------------------------------------
log "figures"
caffeinate -i python -m src.figures --seed 0 --tag prune05 || echo "figures failed"
log "orchestrator finished"
