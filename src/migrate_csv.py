"""Add the Phase 2 `tag` column to a Phase 1 sweep CSV.

Phase 1 wrote 25 columns and knew about exactly one training arm. Phase 2 adds
`tag` so ERM and pruned rows are distinguishable in the same schema. Existing
rows are ERM rows, so they get tag="". Values are copied verbatim; only the
header and the column order change.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from .sweep import COLUMNS


def migrate(path, backup: bool = True) -> bool:
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == COLUMNS:
            print(f"{path}: already migrated")
            return False
        missing = set(COLUMNS) - set(reader.fieldnames or []) - {"tag"}
        if missing:
            raise SystemExit(f"{path}: unexpected header, missing {sorted(missing)}")
        rows = [{**r, "tag": r.get("tag", "")} for r in reader]

    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".phase1.bak"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: migrated {len(rows)} rows"
          + (f" (backup at {path}.phase1.bak)" if backup else ""))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["results/sweep.csv"])
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    for p in args.paths:
        if Path(p).exists():
            migrate(p, backup=not args.no_backup)
        else:
            print(f"{p}: not present, skipping")


if __name__ == "__main__":
    main()
