"""Phase 2 validation gates.

V10  the removed set is class-balanced to within one sample
V10b removal is hardest-first, and a pure function of (difficulty, y, fraction)
V10c the train cache is index-aligned with the train split
V11  keep indices are a strict subset of the original training split
V12  the sweep's resume key separates arms, seeds and checkpoints
V13  a truncated run never writes the selection checkpoint
V14  train.py and run_phase2.sh agree on the *_last.pt name
V15  run_phase2.sh's skip/resume/failure logic (tests/test_runner.sh)
V16  sweep, retention and figures agree on where an arm's rows live
V17  the spurious variant never removes a counter-spurious sample
V18  the spurious variant equalizes remaining class sizes
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.prune_data import (
    class_balanced_budget,
    group_report,
    select_keep_indices,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "waterbirds_v1.0"
TRAIN_CACHE = ROOT / "results" / "train_cache.pt"

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "metadata.csv").exists(), reason="Waterbirds not downloaded"
)
needs_train_cache = pytest.mark.skipif(
    not TRAIN_CACHE.exists(), reason="no train cache (python -m src.data --splits train)"
)

N = 4795  # the real Waterbirds train split size


@pytest.fixture
def toy():
    """Labels in the real 3682/1113 class proportion, with reproducible scores."""
    g = torch.Generator().manual_seed(0)
    y = torch.cat([torch.zeros(3682, dtype=torch.long),
                   torch.ones(1113, dtype=torch.long)])
    d = torch.rand(N, generator=g) * (2 ** 0.5)
    return d, y


# ------------------------------------------------- V10: class-balanced removal

@pytest.mark.parametrize("n_remove,expected", [
    (0, [0, 0]), (1, [1, 0]), (2, [1, 1]), (5, [3, 2]), (480, [240, 240]),
])
def test_v10_budget_split_is_balanced(n_remove, expected):
    b = class_balanced_budget(n_remove, 2)
    assert b == expected
    assert sum(b) == n_remove
    assert max(b) - min(b) <= 1


@pytest.mark.parametrize("fraction", [0.01, 0.02, 0.05, 0.10])
def test_v10_removed_set_is_class_balanced(toy, fraction):
    """THE gate: the pruned-away set differs by at most one sample per class."""
    d, y = toy
    sel = select_keep_indices(d, y, fraction, class_balanced=True)
    per_class = sel["removed_per_class"]
    assert sum(per_class) == sel["n_removed"] == int(round(fraction * N))
    assert max(per_class) - min(per_class) <= 1, per_class


def test_v10b_removal_is_hardest_first(toy):
    """Every removed sample is at least as hard as every kept sample of its class."""
    d, y = toy
    sel = select_keep_indices(d, y, 0.05, class_balanced=True)
    for c in (0, 1):
        cls = (y == c)
        removed_c = sel["removed"][cls[sel["removed"]]]
        kept_c = sel["keep"][cls[sel["keep"]]]
        assert d[removed_c].min() >= d[kept_c].max(), f"class {c} not hardest-first"


def test_v10b_selection_is_deterministic(toy):
    d, y = toy
    a = select_keep_indices(d, y, 0.05)
    b = select_keep_indices(d, y, 0.05)
    assert torch.equal(a["keep"], b["keep"])


def test_v10b_ties_break_by_index():
    """All-equal difficulty must still give a defined, index-ordered answer."""
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    d = torch.ones(8)
    sel = select_keep_indices(d, y, 0.25, class_balanced=True)
    # One per class, highest index first within the tie.
    assert sel["removed"].tolist() == [3, 7]


def test_v10_unbalanced_mode_ignores_class(toy):
    d, y = toy
    sel = select_keep_indices(d, y, 0.05, class_balanced=False)
    order = torch.argsort(d, stable=True).flip(0)[:sel["n_removed"]]
    assert set(sel["removed"].tolist()) == set(order.tolist())


# ----------------------------------------------------- V11: subset is strict

@pytest.mark.parametrize("fraction", [0.01, 0.02, 0.05, 0.10])
def test_v11_keep_is_a_strict_subset(toy, fraction):
    d, y = toy
    sel = select_keep_indices(d, y, fraction)
    keep = sel["keep"]
    assert keep.unique().numel() == keep.numel(), "duplicate keep indices"
    assert int(keep.min()) >= 0 and int(keep.max()) < N, "index out of range"
    assert keep.numel() < N, "nothing was pruned"
    assert keep.numel() + sel["removed"].numel() == N
    assert set(keep.tolist()).isdisjoint(set(sel["removed"].tolist()))
    assert set(keep.tolist()) < set(range(N))


def test_v11_rejects_degenerate_fractions(toy):
    d, y = toy
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            select_keep_indices(d, y, bad)


def test_v11_rejects_fraction_that_empties_a_class(toy):
    """1113 waterbirds: a class-balanced 50% removal needs 1198 from each."""
    d, y = toy
    with pytest.raises(ValueError, match="too large for a class-balanced"):
        select_keep_indices(d, y, 0.5, class_balanced=True)


# ------------------------------------------- V10c: cache / split index alignment

@needs_data
@needs_train_cache
def test_v10c_train_cache_aligns_with_split():
    """Prune indices index the split; difficulty indexes the cache. Same order."""
    from src.data import WaterbirdsSplit, load_eval_cache

    ds = WaterbirdsSplit(DATA_DIR, "train", transform=None)
    cache = load_eval_cache(TRAIN_CACHE)
    assert len(cache["y"]) == len(ds) == N
    assert torch.equal(cache["y"], ds.y)
    assert torch.equal(cache["group"], ds.groups())


@needs_data
@needs_train_cache
def test_v10c_group_report_is_diagnostic_only(toy):
    """group_report must describe the removed set without changing it."""
    from src.data import load_eval_cache

    cache = load_eval_cache(TRAIN_CACHE)
    d = torch.rand(N, generator=torch.Generator().manual_seed(1))
    sel = select_keep_indices(d, cache["y"].long(), 0.05)
    before = sel["removed"].clone()
    rep = group_report(sel["removed"], cache["group"])
    assert torch.equal(sel["removed"], before)
    assert sum(v["removed"] for v in rep.values()) == sel["n_removed"]


# ------------------------------------------------- V12: resume key separates arms

def test_v12_resume_key_separates_arms_and_seeds():
    """The Phase 1 key was (method, param_value, granularity) only, so a second
    seed read the first seed's rows and skipped the entire grid."""
    from src.sweep import _key

    base = ("magnitude", 0.3, "")
    k0 = _key(*base, "", 0, "checkpoints/a.pt")
    assert _key(*base, "", 1, "checkpoints/a.pt") != k0, "seed not in key"
    assert _key(*base, "prune05", 0, "checkpoints/a.pt") != k0, "tag not in key"
    assert _key(*base, "", 0, "checkpoints/b.pt") != k0, "checkpoint not in key"
    assert _key(*base, "", 0, "checkpoints/a.pt") == k0, "key is unstable"


def test_v12_sweep_columns_include_tag():
    from src.sweep import COLUMNS

    assert COLUMNS[0] == "tag"
    assert len(COLUMNS) == len(set(COLUMNS))


# ------------------------------- V13: a truncated run never writes a checkpoint

def test_v13_truncated_run_never_saves_selection_checkpoint():
    """`--stop-after-epoch` exists so seed 0 can be scored without retraining.
    It stopped protecting anything the moment the save was reachable before the
    break: with stop_after_epoch=1, epoch 0 saved, and the Phase 1 checkpoint
    (epoch 3, val_acc_avg 0.9199) was replaced by epoch-0 weights (0.8616).
    No epoch of a truncated run may write it, however good the validation."""
    from src.train import should_save_selection

    # The exact case that caused the loss: epoch 0 of a stop_after_epoch=1 run,
    # beating the initial best of -1.0.
    assert not should_save_selection(True, 0.8615512927439533, -1.0)
    # Nor any later epoch, nor an implausibly good one.
    assert not should_save_selection(True, 0.9999, 0.5)
    # A full run is unaffected: it still selects on improvement only.
    assert should_save_selection(False, 0.8615512927439533, -1.0)
    assert should_save_selection(False, 0.92, 0.8615512927439533)
    assert not should_save_selection(False, 0.86, 0.9199332777314428)


def test_v13_truncated_run_uses_its_own_log_and_summary_names():
    """A truncated run is a scoring run, not a training record. It must not
    claim `train_log_seed0.csv` / `train_summary_seed0.json` -- README.md and
    explain/understand.tex quote that log's per-epoch timings."""
    from src.train import run_suffix

    assert run_suffix(None, None) == ""
    assert run_suffix(None, 1) == "_ep1"
    assert run_suffix("prune05", None) == "_prune05"
    assert run_suffix("prune05", 1) == "_prune05_ep1"
    # The canonical name is reachable ONLY by a full, untagged run.
    assert run_suffix(None, 0) != "", "stop_after_epoch=0 must still be marked"


# -------------------------- V14: the cross-language resume-file name contract

def test_v14_last_checkpoint_name_matches_the_shell():
    """`run_phase2.sh` decides whether a run finished with

        trained() { [ -e "$1" ] && [ ! -e "${1%.pt}_last.pt" ]; }

    and `src/train.py` writes that file from `last_checkpoint_path`. The two
    are written in different languages and nothing links them, so if either
    name drifts the script silently calls every interrupted run complete --
    which is precisely the half-trained-model failure it exists to prevent."""
    from src.train import checkpoint_name, last_checkpoint_path

    for seed in (0, 1, 2):
        for tag in (None, "prune01", "prune10"):
            ckpt = f"checkpoints/{checkpoint_name(seed, tag)}"
            shell = ckpt[:-len(".pt")] + "_last.pt"   # == ${1%.pt}_last.pt
            python = str(last_checkpoint_path("checkpoints", seed, tag))
            assert python == shell, f"{python} != {shell}"


def test_v14_is_run_complete_is_the_single_definition(tmp_path):
    """progress.py and verify.py must both defer to train.is_run_complete --
    there were briefly three hand-rolled copies of this rule, and a drift in
    any one of them silently reclassifies interrupted runs as finished."""
    from src.progress import _trained as progress_trained
    from src.train import is_run_complete
    from src.verify import _trained as verify_trained

    ckpt = tmp_path / "vit_s16_waterbirds_prune05_seed1.pt"
    last = tmp_path / "vit_s16_waterbirds_prune05_seed1_last.pt"

    for setup, expected in (((), False), ((ckpt,), True), ((ckpt, last), False)):
        for f in (ckpt, last):
            f.unlink(missing_ok=True)
        for f in setup:
            f.touch()
        for fn in (is_run_complete, progress_trained, verify_trained):
            assert fn(ckpt) is expected, f"{fn.__module__}.{fn.__name__} on {setup}"


# ----------------------------- V15: the runner's own skip/resume/failure logic

def test_v15_run_phase2_logic():
    """Drives the real stage functions against a fake tree -- no python, no MPS.
    Covers `trained()`, both stage-3 and stage-4 skip paths, the refusal to
    sweep a model that did not finish, failure capture, and the disk floor."""
    import subprocess

    harness = ROOT / "tests" / "test_runner.sh"
    r = subprocess.run(["bash", str(harness), str(ROOT)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 failed" in r.stdout, r.stdout


# ------------------- V16: one arm, one CSV path, agreed by all three modules

def test_v16_arm_csv_path_agrees_across_modules():
    """`src/sweep.py` writes a tagged arm to results/sweep_<tag>.csv and keeps
    results/sweep.csv for the untagged ERM arms. `retention.py` resolved that
    correctly; `figures.py` did not, and hardcoded sweep.csv -- so
    `--tag prune05` always died on "no rows for seed=0 tag='prune05'". Nothing
    linked the three, so pin them together."""
    import yaml

    from src.figures import arm_csv
    from src.sweep import default_paths

    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    for tag in ("", "prune01", "prune05", "prune10"):
        for seed in (0, 1, 2):
            want = default_paths(cfg, seed, tag or None)["csv"]
            assert Path(arm_csv("results", tag)) == Path(want), (
                f"figures disagrees with sweep for tag={tag!r} seed={seed}: "
                f"{arm_csv('results', tag)} != {want}")
            # retention.load_arm builds the same path inline; mirror its rule.
            retention_path = (Path("results") / "sweep.csv" if not tag
                              else Path("results") / f"sweep_{tag}.csv")
            assert retention_path == Path(want), (
                f"retention disagrees for tag={tag!r}: {retention_path} != {want}")


def test_v16_untagged_arms_of_every_seed_share_one_csv():
    """The ERM arms of all three seeds live in ONE sweep.csv, separated by the
    `seed` column -- which is why V12's resume key has to include seed."""
    from src.figures import arm_csv

    assert arm_csv("results", "") == Path("results/sweep.csv")
    assert arm_csv("results", "prune02") == Path("results/sweep_prune02.csv")


# ------------- V17/V18: the identifiable-setting rule (Mulchandani & Kim 6.3)

@needs_train_cache
def test_v17_spurious_variant_never_touches_counter_spurious_groups():
    """The whole point of the group-aware rule. Waterbirds g1 (184) and g2 (56)
    are the counter-spurious groups and g2 is the worst group every headline
    number is reported on. The group-BLIND rule deleted 48% of g1 and 68% of g2
    at a 10% budget, which is why worst-group accuracy fell; the paper says
    exactly that ("will primarily prune samples that do not contain the
    spurious features"). This rule must draw only from g0 and g3."""
    import torch

    from src.prune_data import ALIGNED_GROUP, select_keep_indices_spurious

    cache = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    y, group = cache["y"].long(), cache["group"]
    # Difficulty that deliberately makes the counter-spurious samples hardest,
    # so a rule that drew globally would certainly take them.
    difficulty = torch.where((group == 1) | (group == 2),
                             torch.ones_like(y, dtype=torch.float),
                             torch.zeros_like(y, dtype=torch.float))

    assert ALIGNED_GROUP == {0: 0, 1: 3}
    for sparsity in (0.1, 0.25, 0.5, 0.75, 0.9):
        sel = select_keep_indices_spurious(difficulty, y, group, sparsity)
        removed_groups = group[sel["removed"]]
        for g in (1, 2):
            n = int((removed_groups == g).sum())
            assert n == 0, (f"sparsity {sparsity} removed {n} samples from the "
                            f"counter-spurious group g{g}")
        # and every one it did remove came from an aligned group
        assert set(removed_groups.unique().tolist()) <= {0, 3}


@needs_train_cache
def test_v18_spurious_variant_equalizes_remaining_class_sizes():
    """"we prune such that the total number of samples in both classes is the
    same" -- equal REMAINING sizes, not equal counts removed. The group-blind
    rule does the latter and preserves the 3682/1113 imbalance, which is a
    different intervention."""
    import torch

    from src.prune_data import select_keep_indices_spurious

    cache = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    y, group = cache["y"].long(), cache["group"]
    difficulty = torch.rand(y.numel(), generator=torch.Generator().manual_seed(0))

    for sparsity in (0.25, 0.5, 0.75):
        sel = select_keep_indices_spurious(difficulty, y, group, sparsity,
                                           equalize_classes=True)
        kept = sel["kept_per_class"]
        assert abs(kept[0] - kept[1]) <= 1, f"sparsity {sparsity}: kept {kept}"
        assert sel["n_keep"] == sum(kept) == sel["n_total"] - sel["n_removed"]
        # The counter-spurious groups are never removed, so they bound what is
        # achievable: class 1 can never fall below its g2 count.
        assert kept[1] >= int((group == 2).sum())

    # Without equalization the imbalance survives, which is the contrast.
    sel = select_keep_indices_spurious(difficulty, y, group, 0.25,
                                       equalize_classes=False)
    assert sel["kept_per_class"][0] > sel["kept_per_class"][1]


@needs_train_cache
def test_v18_spurious_variant_is_deterministic():
    """Same purity requirement V10b puts on the group-blind rule."""
    import torch

    from src.prune_data import select_keep_indices_spurious

    cache = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    y, group = cache["y"].long(), cache["group"]
    d = torch.rand(y.numel(), generator=torch.Generator().manual_seed(7))
    a = select_keep_indices_spurious(d, y, group, 0.5)
    b = select_keep_indices_spurious(d, y, group, 0.5)
    assert torch.equal(a["keep"], b["keep"])
    assert torch.equal(a["removed"], b["removed"])
