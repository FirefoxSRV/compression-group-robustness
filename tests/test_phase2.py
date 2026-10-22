"""Phase 2 validation gates.

V10  the removed set is class-balanced to within one sample
V10b removal is hardest-first, and a pure function of (difficulty, y, fraction)
V10c the train cache is index-aligned with the train split
V11  keep indices are a strict subset of the original training split
V12  the sweep's resume key separates arms, seeds and checkpoints
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
