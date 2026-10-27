Pristine Phase 1 (seed 0) records, copied before the seed-0 checkpoint was
rebuilt on 2026-10-26.

The checkpoint had been overwritten with epoch-0 weights by a truncated
--stop-after-epoch run (see README.md, "Seed 0 is scored without being
retrained", and gate V13). checkpoints/ is gitignored and there were no
snapshots, so these files were the only remaining description of the original
weights, and the restoration was verified against them:

  PHASE1_train_log_seed0.csv      10 epochs; epoch 3 selected, val_acc_avg
                                  0.9199332777314428
  PHASE1_train_summary_seed0.json the selection record
  PHASE1_sweep.csv                the 29 Phase 1 configurations
  PHASE1_summary_table.csv        the published table

Do not delete. preflight C3 checks live checkpoints against records like these.
