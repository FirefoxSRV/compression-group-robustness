#!/usr/bin/env bash
# Keep results/logs/PROGRESS.txt fresh for as long as the run is alive.
#
# Deliberately a shell loop around single-shot renders rather than
# `src.progress --watch`: that long-lived python process was killed silently on
# 2026-10-26 and left the file stale for 39 minutes with no indication. Here a
# failed render costs one frame, and the loop ends when the run does -- after
# writing one final frame.
cd "$(dirname "$0")"
STARTED=$(cat results/logs/started_at 2>/dev/null || date +%s)
render() {
  python -m src.progress --started "$STARTED" \
    --out results/logs/PROGRESS.txt >/dev/null 2>>results/logs/progress_watch.log || true
}
while pgrep -f "[r]un_phase2.sh" >/dev/null; do render; sleep 60; done
render     # final frame, so the file reflects the finished state
echo "[progress_watch] run finished at $(date '+%F %T'); wrote final frame" \
  >> results/logs/progress_watch.log
