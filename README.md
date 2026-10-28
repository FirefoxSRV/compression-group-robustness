# Compression and group robustness on Waterbirds

Does post-hoc model compression damage the groups a model was already worst at?

Short answer, on this setup: **yes, and average accuracy hides it almost
completely.** Pruning 30% of a fine-tuned ViT-S/16's transformer weights costs
**0.8 points** of adjusted average accuracy and **16.0 points** of worst-group
accuracy — a 21x difference. An evaluation that reports only average accuracy
would call that compression free.

## Setup

| | |
|---|---|
| Model | `vit_small_patch16_224` (timm), ImageNet-pretrained, 2-way head |
| Data | Waterbirds v1.0, official splits, read directly from `metadata.csv` |
| Training | Plain ERM, 10 epochs, AdamW `lr=1e-5`, cosine, batch 32 |
| Selection | Best **validation average** accuracy (seed 0: epoch 3) |
| Compression | Applied to the trained checkpoint, evaluated immediately, **never retrained** |
| Scope | Only in-block Linear weights: `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` (21,233,664 params) |

Four groups, indexed `g = 2*y + background`: `landbird_on_land` (majority),
`landbird_on_water` (minority), `waterbird_on_land` (minority),
`waterbird_on_water` (majority). The two minority groups are the ones that
*conflict* with the background shortcut.

### Three choices that decide what this measures

**Training is deliberately group-blind.** No reweighting, no balanced sampler,
no GroupDRO. The model is *supposed* to learn the background shortcut — if
training were group-balanced, worst-group accuracy would start high and the
experiment would measure nothing.

**Model selection is on validation average accuracy, never worst-group.**
Selecting on worst-group is itself a group-aware intervention; it would inflate
the ERM baseline and void the premise. Adjusted-average selection is out for the
same reason — the adjustment weights need group labels, so using it would leak
group information into a pipeline that is meant to have none.

**No retraining after compression.** Fine-tuning afterwards lets the model re-fit
the shortcut and answers a different question.

`acc_adj_avg` is per-group accuracy reweighted by *training* frequency. Waterbirds'
val/test splits are far more group-balanced than train, so a plain test-set mean is
not the ~97% figure quoted in the literature; the adjusted average is.

## Results

Baseline (uncompressed, seed 0): **98.2% adjusted average, 70.4% worst group**,
gap 27.8 points.

| Configuration | Targeted sparsity | Adjusted avg | Worst group | Gap |
|---|---|---|---|---|
| uncompressed | 0.000 | 98.2% | 70.4% | 27.8 |
| quantize 8-bit, per-channel | 0.014 | 98.2% (−0.0) | 70.4% (−0.0) | 27.8 |
| quantize 6-bit, per-tensor | 0.190 | 97.8% (−0.4) | 62.6% (**−7.8**) | 35.2 |
| magnitude, 30% | 0.300 | 97.5% (−0.8) | 54.4% (**−16.0**) | 43.1 |
| quantize 4-bit, per-channel | 0.232 | 97.1% (−1.2) | 60.6% (**−9.8**) | 36.5 |
| remove 6 of 72 heads | 0.028 | 96.5% (−1.7) | 48.1% (**−22.3**) | 48.4 |
| FFN, 30% of neurons | 0.200 | 83.3% (−14.9) | 2.5% (−67.9) | 80.9 |

Removing **6 attention heads out of 72** — 2.8% of the targeted weights — costs
1.7 points of average accuracy and **22.3 points** of worst-group accuracy.

Three figures in `figures/`:

- `fig1_avg_vs_worst_group.png` — average vs worst-group accuracy for all five
  configurations. The shaded band is the gap; it widens everywhere.
- `fig2_tradeoff.png` — worst-group plotted against average accuracy, so methods
  are comparable at equal average accuracy despite non-commensurable knobs.
- `fig3_per_group.png` — which groups pay. `landbird_on_land` holds at ~100%
  throughout while `waterbird_on_land` collapses first.

### Two things in this data that are easy to misread

**The gap narrows again at extreme compression — that is not robustness
returning.** Past roughly 80% magnitude sparsity or 70% FFN removal the model
collapses to predicting a single class. Accuracy is still well-defined, but a
model with one output has no group disparity left to measure. Those points are
detected by `pred_mean` (gate V7b), drawn hollow in every figure, and excluded
from `fig2`.

**Head removal is not monotonic.** Removing 48 heads scores *better* on
worst-group (34.1%) than removing 36 (1.1%), then falls again at 60. Global
magnitude ranking does not remove heads in order of behavioral importance, and
near-degenerate models reshuffle which group is worst. Treat the deep end of the
head sweep as noise around a collapsed model, not as signal.

**Quantization granularity matters more than bit width.** Per-channel 4-bit keeps
60.6% worst-group; per-tensor 4-bit keeps 23.8%. Same bit budget, different
outcome.

## Reproducing

```bash
pip install -r requirements.txt
python -m src.data                 # download Waterbirds, build uint8 eval caches
python -m src.train --seed 0       # ~20 min on an M2 Mac (MPS), 10 epochs
python -m src.sweep --seed 0       # 29 configurations, ~15 min
python -m src.figures
pytest tests/ -q                   # 52 validation gates
```

Budget ~35 min per seed end to end. The `~2h` figure an earlier draft of this
file quoted was measured on a run that slept: `pmset -g` reports `sleep 1` on
the development machine, and epochs 3, 7, 8 and 9 of `train_log_seed0.csv` took
840s, 678s, 3338s and 616s against 114s for every epoch that ran uninterrupted.
Prefix long runs with `caffeinate -i` and keep the machine on mains power.

`src.sweep` appends and flushes each row, and `--resume` (the default) skips
configurations already in `results/sweep.csv`, so an interrupted run continues
where it stopped. A row's identity is `(tag, method, param_value, granularity,
seed, checkpoint)` — through Phase 1 it was only the middle three, which meant
a second seed read the first seed's rows, declared all 29 configurations done,
and exited having computed nothing.

## Phase 2: does data pruning survive compression?

Phase 1 compresses a model that was never made robust. Phase 2 makes it robust
first — with the data pruning of Mulchandani and Kim (ICLR 2025) — and asks
whether compression takes the robustness back.

```bash
./run_phase2.sh                    # everything, ~9h; resumable
SEEDS=0 ./run_phase2.sh            # seed 0 only, ~3h, a complete single-seed answer
./run_phase2.sh stage1 stage3      # any subset of stages, in any order
python -m src.preflight            # the stage-0 checks on their own
```

The default stage order is **1, 3, 4, then 2, 3, 4** — not 1, 2, 3, 4. Stages 3
and 4 skip whatever is not ready yet and do whatever is, so this order lands
seed 0's complete pruning answer (the gate on whether pruning buys anything at
all) before spending ~1.3 h on the seeds 1–2 ERM baselines. If the night is cut
short, what survives is a result rather than a prefix.

Three properties make it safe to leave running unattended:

* **Stage 0 is a preflight.** `src/preflight.py` checks the device, all three
  caches and their group counts, checkpoint integrity, the sweep CSV header,
  and the disk against what this particular run will write — then refuses to
  start if anything is wrong. Nine hours is too long to discover a missing
  cache at hour six.
* **A failed step is recorded, not fatal.** One transient failure must not cost
  the remaining hours. Every failure is listed at the end and the script exits
  non-zero. Re-running the same command resumes.
* **Completion means a checkpoint *and* no `*_last.pt`.** `src/train.py` saves
  best-so-far on every validation improvement, so a run killed at epoch 8
  leaves a healthy-looking epoch-3 checkpoint behind. Treating that as finished
  would seat a half-trained model in a 10-epoch slot and the sweep would just
  compress it. Gates V14 and V15 pin this down, in both languages.

What that script runs, and what each piece is for:

```bash
python -m src.data --splits train                 # un-augmented train cache (~700 MB)
python -m src.train --seed 0 --score --stop-after-epoch 1   # difficulty scores only
python -m src.prune_data --seed 0                 # keep-index sets per fraction
python -m src.train --seed 0 \
    --prune-indices results/prune_indices/prune05_seed0.pt  # retrain from scratch
python -m src.sweep --seed 0 --tag prune05        # same 29 configs, pruned checkpoint
python -m src.retention --seed 0 --tag prune05    # the headline number
python -m src.figures --seed 0 --tag prune05
```

**Difficulty** is `d_i = ||p(w_e, x_i) - onehot(y_i)||_2`, read from one ERM run
at a fixed early epoch (`prune.score_epoch`, default 1) over the *un-augmented*
train cache — difficulty must be a property of the sample and the weights, not
of which random crop was drawn. The hardest samples are removed class-balanced
(equal count per class, so the removed set is balanced to within one sample)
and the network is retrained from scratch on the rest. No group labels, no
spurious-attribute access: the `prune_general.py` variant, because a group-aware
pruner would leak exactly the information the ERM baseline is denied.

**Seed 0 is scored without being retrained.** `--stop-after-epoch` runs the real
cosine schedule and exits early, writing no selection checkpoint, so the Phase 1
checkpoint that produced every number above is never touched.

It did not always hold. The guard used to be ordered

```python
if stop_after_epoch is not None and epoch >= stop_after_epoch:
    break                       # only fires AT the truncation epoch
if val_acc_avg > best[...]:
    save_checkpoint(...)        # epoch 0 of a stop_after_epoch=1 run reaches this
```

so with `score_epoch: 1` epoch 0 failed the break test and fell straight through
to the save. `vit_s16_waterbirds_seed0.pt` was overwritten with epoch-0 weights
(`val_acc_avg` 0.8616 in place of 0.9199) by the very flag that exists to
prevent it. Checkpoints are gitignored and not backed up, so the only recovery
was a full retrain. Three things now stand in the way:

* the selection save and the resume write are both gated on `truncated`, not on
  the break — **gate V13** asserts the exact failing case;
* a *fresh* run refuses to start on top of an existing selection checkpoint
  unless given `--allow-overwrite` (`--resume` continues one instead);
* **preflight C3** compares every checkpoint against the records that describe
  it — `train_summary_seed{s}.json` for the selected epoch, and `baseline.json`
  by re-evaluating the weights — so a checkpoint holding the wrong epoch is
  caught before 348 sweep configurations are computed against it.

**On reproducibility.** The older claim here was that MPS training is not
bit-reproducible. Measured, it is: at `--num-workers 0` seed 0's epoch-0 loss
came back as `0.20122195619534278` on three independent runs, and a full
10-epoch rerun reproduced the whole trajectory. That is what made recovering the
lost checkpoint possible. `--num-workers > 0` does shift it, because workers are
seeded from the loader generator and each sample gets a different augmentation
draw — epoch-0 loss moves to 0.2005. Workers also measured no faster here (189 s
against a 125–208 s spread at 0 workers, on a fanless M2 whose epoch time is set
by thermal and disk state), so **0 is the default** and bit-comparability with
Phase 1 is free.

**The headline quantity is the retention ratio.** With `W_p` and `W_e` the
worst-group accuracy of the pruned and ERM models at the same configuration,

```
D(s) = W_p(s) - W_e(s)        R(s) = D(s) / D(0)
```

`R ≈ 1` means the intervention survives deployment; `R ≈ 0` means compression
silently undid it. `src/retention.py` reports `R` per compression family,
because Phase 1 already showed the families are not interchangeable.

Two refusals are built in. `R` is left undefined when the dense gain `D(0)` is
below `--min-gain` — if data pruning buys nothing here there is nothing for
compression to undo, and that null result is reported as such rather than as a
ratio manufactured by dividing by noise. And collapsed rows (V7b, `pred_mean`
pinned near 0 or 1) are excluded, because a model with one output has no group
disparity left to retain.

## Phase 2 results: the intervention survives compression until compression breaks the model

The identifiable-setting rule reproduces the paper on ViT-S/16. Dense
(uncompressed) test accuracy, 3 seeds; the paper's `Mean%` convention is our
`acc_adj_avg`, so these are directly comparable:

| | Worst% | Mean% |
|---|---|---|
| ERM (ours) | 73.31 | 98.32 |
| ERM (paper) | 74.81 | 98.10 |
| spur25 | 86.40 | 94.69 |
| spur50 | 87.90 | 94.35 |
| spur75 | 87.28 | 92.10 |
| Ours (paper) | 90.93 | 92.48 |

The gain over ERM is **+13.9 pp (sd 3.1), positive in 9/9 arms**, and flat in
sparsity across 25–75% — the claim of the paper's Fig. 8. We land 3–4 pp under
their 90.93 on a different architecture (ViT-S/16 vs ResNet-50), a tenth of the
schedule (10 epochs vs 100), and group-blind model selection (`val_acc_avg`,
never worst-group), which makes these numbers a lower bound.

**The retention question does not split by compression family.** Over all 195
non-collapsed configurations, what decides R is whether compression left a
working model at all:

| regime | n | R mean | R ≥ 0.75 |
|---|---|---|---|
| compressed worst-group ≥ 0.40 (usable) | 114 | **+1.30** | 96% |
| 0.25–0.40 (degraded) | 14 | +1.30 | 86% |
| < 0.25 (broken) | 67 | −0.20 | 13% |

Wherever the compressed model keeps usable worst-group accuracy, the gain is
retained and often widens — across *all four* families, R from +0.29 to +3.58.
Negative R never appears above a compressed worst-group of 0.269, because there
R is a ratio between two degenerate predictors and means nothing.

What the families actually differ in is **where that cliff falls**, which is
Phase 1's conclusion restated:

| family | gain retained up to | destroyed by |
|---|---|---|
| magnitude | 50% sparsity (R 2.35) | 70% (R −0.08) |
| heads | 12–24 of 72 (R 1.21) | 36 (R −0.50) |
| ffn | — (R 0.22 already at 30%) | 30% |
| quantize, per-channel | 3–4 bits (R 1.09–1.28) | — |
| quantize, per-tensor | 6 bits (R 1.20) | 4 bits (R −1.06) |

Quantization granularity again matters more than bit width: per-channel holds
the gain down to 3 bits while per-tensor loses it at 4. Averaging R within a
family hides this — a mean of +1.26 for magnitude smears a +2.35 at 50%
sparsity together with a −0.08 at 70% and describes neither.

Consistent with the compression-and-fairness literature, which finds
unstructured pruning less damaging than structured removal and quantization
gentler than pruning.

## Layout

```
configs/base.yaml   every number the pipeline depends on
src/data.py         Waterbirds splits, group indexing, uint8 eval caches
src/metrics.py      group-aware metrics (adjusted average, worst-group, gap)
src/model.py        ViT construction, checkpoint I/O, the frozen compression scope
src/train.py        group-blind ERM fine-tuning
src/evaluate.py     one evaluation pass over a cached split
src/compress/       magnitude, heads, ffn, quantize -- one shared contract
src/sweep.py        runs every configuration, writes results/sweep.csv
src/figures.py      the three figures + results/summary_table.csv
src/prune_data.py   Phase 2: difficulty ranking -> class-balanced keep-index sets
src/retention.py    Phase 2: the retention ratio R(s), per compression family
src/migrate_csv.py  one-shot: adds the Phase 2 `tag` column to a Phase 1 CSV
src/preflight.py    Phase 2 stage 0: fail-fast checks before a 9h unattended run
run_phase2.sh       Phase 2 end to end, resumable, under caffeinate
tests/              the V-gates the rest of the code refers to by number
```

`tests/test_validation.py` holds the Phase 1 gates (V1–V9) and
`tests/test_phase2.py` the Phase 2 ones: V10 the removed set is class-balanced
to within one sample, V10b removal is hardest-first and deterministic under
ties, V10c the train cache is index-aligned with the train split, V11 keep
indices are a strict subset, V12 the sweep's resume key separates arms, seeds
and checkpoints, V13 a truncated run never writes the selection checkpoint, V14
`train.py` and `run_phase2.sh` agree on the `*_last.pt` name that decides
whether a run finished, V15 the runner's own skip/resume/failure logic
(`tests/test_runner.sh`, driven against a fake tree — no MPS).

### The compression contract

Every method is `apply(model, **params) -> (new_model, stats)`, returns a **new**
model, never mutates its input, and reports `n_params_targeted`,
`n_params_zeroed`, `achieved_sparsity`.

Structured methods (`heads`, `ffn`) mask rather than physically narrow tensors.
Shapes stay ViT-S/16, so `achieved_sparsity` means the same thing across all four
methods — and masking is *exact*, not an approximation:

- A head owns `proj` input columns `[h*64:(h+1)*64]`; zeroing them removes its
  contribution outright. Its `qkv` rows are zeroed too, so the parameter count is
  honest, though they cannot affect the output once `proj` is zeroed.
- An FFN neuron is `fc1` row *i* plus `fc2` column *i*. `fc1.bias` stays untouched
  (biases are out of scope) and costs nothing: the leftover `GELU(bias_i)` is
  multiplied by a zeroed `fc2` column.

`tests/test_validation.py` checks both claims against a reference model built by
zeroing *only* the output-side tensor.

Quantization is simulated (quantize-dequantize). The measurement here is accuracy,
not latency: the numerical effect is reproduced exactly, and real int4/int8 kernels
are CUDA-only and irrelevant to an accuracy study.

## Scope and limits

- **One seed, for now.** `configs/base.yaml` lists seeds `[0, 1, 2]`; at the time
  the results above were written only seed 0 had been trained and swept, and the
  resume bug described under *Reproducing* is why a second seed appeared to
  complete instantly. The headline gaps (16–22 points) are far larger than typical
  seed variance on Waterbirds, but the per-configuration numbers — especially in
  the collapsed regime — should not be read as tight.
- **One dataset, one architecture.** Waterbirds is a synthetic-shortcut benchmark
  with a small, clean spurious correlation; nothing here establishes the effect
  size on natural distribution shift.
- **Data-free importance criteria.** Heads and neurons are ranked by weight L2
  norm. A calibration-data-based criterion might prune in a more
  robustness-preserving order — testing that is the obvious next experiment.
- Magnitude pruning zeroes *exactly* k weights: the fp32 threshold value is
  genuinely non-unique at 21M weights (ties run 1–3 deep), so the implementation
  carries an explicit tie budget rather than a `> threshold` mask.
