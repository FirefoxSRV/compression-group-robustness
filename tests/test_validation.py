"""The validation gates the pipeline's correctness rests on.

Each test is named for the gate it enforces, because the rest of the source
refers to them by number in comments ("V4 is an equality check", "V7b collapse
detector"). If you renumber them here, fix those comments too.

The gates that need the trained checkpoint or the 872 MB eval cache skip
cleanly when those artifacts are absent, so a fresh clone can still run the
structural half of the suite.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from src import metrics
from src.compress import PARAM_NAME, REGISTRY, ffn, heads, magnitude, quantize
from src.data import WaterbirdsSplit, group_counts
from src.model import (
    assert_structure,
    count_zeros,
    create_model,
    fresh_model,
    n_targeted_params,
    targeted_named_weights,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "waterbirds_v1.0"
CKPT = ROOT / "checkpoints" / "vit_s16_waterbirds_seed0.pt"
TEST_CACHE = ROOT / "results" / "test_cache.pt"

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "metadata.csv").exists(), reason="Waterbirds not downloaded"
)
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
needs_cache = pytest.mark.skipif(not TEST_CACHE.exists(), reason="no test cache")


@pytest.fixture(scope="module")
def model():
    """One untrained ViT-S/16, shared. Structure is what these gates test."""
    torch.manual_seed(0)
    m = create_model(pretrained=False).eval()
    assert_structure(m)
    return m


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.randn(2, 3, 224, 224)


# ---------------------------------------------------------------- V1, V2: data

@needs_data
def test_v1_train_group_counts():
    """Group indexing is right, or every downstream number is meaningless.

    These four counts are the published Waterbirds train composition. If
    `group_index` ever swapped its (background, y) columns, counts 1 and 2
    (184 and 56) would trade places -- which is exactly why the asymmetric
    minority sizes make this a real test rather than a checksum.
    """
    ds = WaterbirdsSplit(DATA_DIR, "train")
    assert group_counts(ds.groups()) == metrics.TRAIN_GROUP_COUNTS


@needs_data
@pytest.mark.parametrize("split", ["val", "test"])
def test_v2_no_empty_eval_group(split):
    """Worst-group accuracy over an empty group would be NaN, silently."""
    ds = WaterbirdsSplit(DATA_DIR, split)
    counts = group_counts(ds.groups())
    assert all(c > 0 for c in counts), f"{split} has an empty group: {counts}"


def test_v1b_adjusted_weights_sum_to_one():
    """Exact fractions, not the spec's rounded literals (which sum to 1.00001)."""
    assert sum(metrics.TRAIN_GROUP_FRACTIONS) == pytest.approx(1.0, abs=1e-12)


def test_metrics_worst_group_and_gap():
    """gap_adj is the headline quantity -- check it against hand-built counts."""
    # g0: 3/4, g1: 1/2, g2: 1/1, g3: 2/2  -> worst group is g1 at 0.5.
    # Labels obey the group encoding g = 2*y + background, so g0/g1 are y=0
    # and g2/g3 are y=1; a group holding the wrong label would not be a
    # Waterbirds group at all.
    preds = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1, 1])
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3, 3])
    m = metrics.compute_metrics(preds, labels, groups)
    assert m["acc_g0"] == pytest.approx(0.75)
    assert m["acc_g1"] == pytest.approx(0.5)
    assert m["worst_group_id"] == 1
    assert m["acc_wg"] == pytest.approx(0.5)
    assert m["gap_adj"] == pytest.approx(m["acc_adj_avg"] - m["acc_wg"])
    assert m["acc_avg_plain"] == pytest.approx(7 / 9)


# ------------------------------------------------------- V3: quantize identity

@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
def test_v3_quantize_identity_at_32_bits(model, x, granularity):
    """bits>=32 must be exactly identity, via the short-circuit.

    Without it, W/scale reaches ~2.1e9 -- past fp32's 24-bit mantissa -- and the
    round trip carries ~1e-7 relative error instead of being identity.
    """
    q, stats = quantize.apply(model, bits=32, granularity=granularity)
    assert stats["identity"] is True
    with torch.no_grad():
        assert torch.allclose(q(x), model(x), atol=1e-6)


def test_v3b_quantize_degrades_monotonically(model):
    """Fewer bits must not round-trip *closer* to the original weights."""
    W = model.blocks[0].attn.qkv.weight.detach()
    errs = [
        float((quantize.fake_quantize(W, b, per_channel=False) - W).abs().max())
        for b in (8, 6, 4, 3)
    ]
    assert errs == sorted(errs), f"error not monotone in bits: {errs}"


def test_v3c_per_channel_beats_per_tensor(model):
    """Per-channel scales are strictly finer-grained, so error cannot be worse."""
    W = model.blocks[0].mlp.fc1.weight.detach()
    per_t = float((quantize.fake_quantize(W, 4, False) - W).pow(2).mean())
    per_c = float((quantize.fake_quantize(W, 4, True) - W).pow(2).mean())
    assert per_c <= per_t


# --------------------------------------------------- V4: magnitude exactness

@pytest.mark.parametrize("sparsity", [0.0, 0.3, 0.5, 0.9])
def test_v4_magnitude_zeroes_exactly_k(model, sparsity):
    """EQUALITY, not a tolerance -- that is the point of the tie budget.

    The fp32 threshold value is genuinely non-unique at 21M weights (ties run
    1-3 deep), so a naive `|w| > threshold` mask overshoots. Exactly k is the
    claim; this is where it gets checked.

    Any weights that were ALREADY zero are among the k smallest by definition,
    so they are absorbed into k rather than added to it.
    """
    n_targeted = n_targeted_params(model)
    pre_existing = count_zeros(model)
    k = int(round(sparsity * n_targeted))

    m, stats = magnitude.apply(model, sparsity=sparsity)
    assert count_zeros(m) == (pre_existing if k == 0 else k)
    assert stats["n_params_zeroed"] == count_zeros(m)


def test_v4b_magnitude_prunes_the_smallest(model):
    """Nothing above the threshold may be pruned, and nothing below it survive.

    `>=`, not `>`: weights tied exactly at the threshold sit on both sides of
    the cut -- the tie budget zeroes some and keeps the rest, which is what
    makes the count come out at exactly k.
    """
    m, stats = magnitude.apply(model, sparsity=0.5)
    thresh = stats["threshold"]
    for (_n, orig), (_n2, pruned) in zip(
        targeted_named_weights(model), targeted_named_weights(m)
    ):
        a = orig.detach().abs()
        kept = pruned != 0
        assert bool((a[kept] >= thresh).all()), "kept a weight below the threshold"
        assert bool((a[~kept] <= thresh).all()), "pruned a weight above the threshold"


def test_v4c_magnitude_rejects_sparsity_one(model):
    with pytest.raises(ValueError):
        magnitude.apply(model, sparsity=1.0)


# ------------------------------------ V5: structured removal is EXACT, not approx

def test_v5_head_masking_matches_true_deletion(model, x):
    """Zeroing qkv rows must not change the output -- proj already blocks the head.

    This is the load-bearing claim behind head removal: if timm ever inserts a
    non-Identity `attn.norm` or a gate between the head concat and `proj`, the
    proj-column argument collapses and this test is what catches it.
    """
    masked, stats = heads.apply(model, n_heads_removed=12)
    ref = copy.deepcopy(model).eval()
    with torch.no_grad():
        for b, h in stats["removed"]:
            _rows, cols = heads.head_slices(ref.blocks[b].attn, h)
            ref.blocks[b].attn.proj.weight[:, cols] = 0.0  # proj only
        assert torch.allclose(masked(x), ref(x), atol=1e-6)


def test_v5b_ffn_masking_matches_true_deletion(model, x):
    """Zeroing fc1 rows must not change the output -- fc2's zeroed column absorbs
    the leftover GELU(bias) constant. This is why biases stay out of scope."""
    masked, _stats = ffn.apply(model, neuron_fraction=0.5)
    ref = copy.deepcopy(model).eval()
    with torch.no_grad():
        for _s, b, i in ffn.select_neurons(model, 0.5):
            ref.blocks[b].mlp.fc2.weight[:, i] = 0.0  # fc2 only
        assert torch.allclose(masked(x), ref(x), atol=1e-6)


def test_v5c_head_floor_is_respected(model):
    """A block stripped to zero heads is a different architecture, not a
    compressed one -- and its accuracy cliff would be misread as head sparsity."""
    _m, stats = heads.apply(model, n_heads_removed=60, min_heads_per_block=1)
    assert stats["n_heads_removed"] == 60
    assert max(stats["heads_removed_per_block"]) <= 5
    assert all(r <= 5 for r in stats["heads_removed_per_block"])


def test_v5d_head_budget_cannot_exceed_the_floor(model):
    """12 blocks x (6 - 1) = 60 removable. 61 must raise, not silently clamp."""
    with pytest.raises(ValueError):
        heads.apply(model, n_heads_removed=61, min_heads_per_block=1)


def test_v5e_structured_zero_counts_are_exact(model):
    """Each head owns 3*head_dim*dim qkv weights + dim*head_dim proj weights."""
    pre = count_zeros(model)
    n = 6
    _m, stats = heads.apply(model, n_heads_removed=n)
    attn = model.blocks[0].attn
    per_head = 3 * attn.head_dim * attn.qkv.weight.shape[1] + \
        attn.proj.weight.shape[0] * attn.head_dim
    assert stats["n_params_zeroed"] - pre == n * per_head


# ------------------------------------------- V8, V9: the no-mutation contract

@pytest.mark.parametrize("method,params", [
    ("magnitude", {"sparsity": 0.5}),
    ("heads", {"n_heads_removed": 12}),
    ("ffn", {"neuron_fraction": 0.5}),
    ("quantize", {"bits": 4}),
])
def test_v8_apply_never_mutates_its_input(model, x, method, params):
    """Compressing an already-compressed model produces smooth, plausible,
    entirely wrong curves. Nothing about the output looks wrong -- so this
    contract is enforced here and re-checked per-config inside the sweep."""
    with torch.no_grad():
        before = model(x).clone()
    before_zeros = count_zeros(model)

    out, _stats = REGISTRY[method](model, **params)

    assert out is not model
    assert count_zeros(model) == before_zeros
    with torch.no_grad():
        assert torch.equal(model(x), before)


@pytest.mark.parametrize("method,params", [
    ("magnitude", {"sparsity": 0.0}),
    ("heads", {"n_heads_removed": 0}),
    ("ffn", {"neuron_fraction": 0.0}),
])
def test_v9_no_op_configs_are_bit_identical(model, x, method, params):
    """The sweep's baseline rows. If a no-op config moves a single logit, the
    harness is wrong and nothing downstream of it can be trusted."""
    out, _stats = REGISTRY[method](model, **params)
    with torch.no_grad():
        assert torch.equal(out(x), model(x))


def test_registry_param_names_match_signatures():
    """PARAM_NAME drives the CSV's param_value column; a mismatch would file
    every row of a method under the wrong knob."""
    assert set(REGISTRY) == set(PARAM_NAME)
    for method, knob in PARAM_NAME.items():
        import inspect
        assert knob in inspect.signature(REGISTRY[method]).parameters


def test_compression_scope_is_frozen(model):
    """Only in-block Linear weights. patch_embed / pos_embed / head / norms /
    biases must never appear, or methods stop being comparable to each other."""
    names = [n for n, _ in targeted_named_weights(model)]
    assert len(names) == 12 * 4
    assert all(n.startswith("blocks.") for n in names)
    for forbidden in ("patch_embed", "pos_embed", "cls_token", "head.", "norm", "bias"):
        assert not any(forbidden in n for n in names), forbidden
    assert n_targeted_params(model) == 21_233_664


# ----------------------------------------------- V7b: collapse detector, Gate 3

@needs_ckpt
@needs_cache
def test_v7b_baseline_does_not_collapse():
    """A model predicting one class everywhere can still post a respectable
    adjusted average on this label distribution. pred_mean near 0 or 1 is the
    tell, and it is why the column exists in the CSV."""
    from src.data import get_device, load_eval_cache
    from src.evaluate import evaluate_cached

    device = get_device()
    m = fresh_model(CKPT, device)
    cache = load_eval_cache(TEST_CACHE)
    out = evaluate_cached(m, cache, device)
    assert 0.05 < out["pred_mean"] < 0.95, f"collapsed: pred_mean={out['pred_mean']}"
    assert out["acc_adj_avg"] > 0.8


@needs_ckpt
def test_fresh_model_is_reproducible():
    """Every sweep config starts here; two loads must be bit-identical."""
    a = fresh_model(CKPT, torch.device("cpu"))
    b = fresh_model(CKPT, torch.device("cpu"))
    for (_n, p), (_n2, q) in zip(targeted_named_weights(a), targeted_named_weights(b)):
        assert torch.equal(p, q)
