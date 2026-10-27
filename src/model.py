"""ViT-S/16 construction, checkpoint I/O, and the frozen compression scope."""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

TIMM_NAME = "vit_small_patch16_224"
NUM_CLASSES = 2

# Compression scope, decided once (spec 6.1) and reused by EVERY method.
# Only the Linear weights inside transformer blocks.
TARGET_SUFFIXES = (
    "attn.qkv.weight",
    "attn.proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)
# Logged into the CSV so the choice is auditable.
EXCLUDED_DESCRIPTION = (
    "patch_embed, cls_token, pos_embed, all LayerNorm, all biases, head"
)

EXPECTED_STRUCTURE = dict(dim=384, num_heads=6, head_dim=64, hidden=1536, depth=12)


def create_model(pretrained: bool = False) -> nn.Module:
    return timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)


def assert_structure(model: nn.Module) -> None:
    """Fail loudly if timm's ViT layout ever drifts from what compression assumes."""
    e = EXPECTED_STRUCTURE
    assert len(model.blocks) == e["depth"], f"depth {len(model.blocks)} != {e['depth']}"
    for i, blk in enumerate(model.blocks):
        a = blk.attn
        assert a.num_heads == e["num_heads"], f"block {i} num_heads {a.num_heads}"
        assert a.head_dim == e["head_dim"], f"block {i} head_dim {a.head_dim}"
        # attn_dim == num_heads * head_dim is what makes exact head removal work:
        # forward ends `x.transpose(1,2).reshape(B,N,attn_dim)` then `proj(x)`,
        # so head h owns proj input columns [h*head_dim : (h+1)*head_dim].
        assert a.attn_dim == e["num_heads"] * e["head_dim"], f"block {i} attn_dim"
        assert a.qkv.weight.shape == (3 * a.attn_dim, e["dim"]), f"block {i} qkv shape"
        assert a.proj.weight.shape == (e["dim"], a.attn_dim), f"block {i} proj shape"
        assert blk.mlp.fc1.weight.shape == (e["hidden"], e["dim"]), f"block {i} fc1"
        assert blk.mlp.fc2.weight.shape == (e["dim"], e["hidden"]), f"block {i} fc2"
        # scale_norm / gated variants would break the proj-column argument.
        assert isinstance(a.norm, nn.Identity), f"block {i} unexpected attn.norm"
        assert getattr(a, "gate", None) is None, f"block {i} unexpected attn.gate"


def targeted_named_weights(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """The exact tensors every compression method is allowed to touch."""
    return [
        (name, p)
        for name, p in model.named_parameters()
        if name.startswith("blocks.") and name.endswith(TARGET_SUFFIXES)
    ]


def n_targeted_params(model: nn.Module) -> int:
    return sum(p.numel() for _, p in targeted_named_weights(model))


def count_zeros(model: nn.Module) -> int:
    return sum(int((p == 0).sum()) for _, p in targeted_named_weights(model))


def fresh_model(ckpt_path, device) -> nn.Module:
    """A pristine model from the checkpoint. EVERY compression config starts here.

    Never compress an already-compressed model: that produces smooth, plausible,
    entirely wrong curves. V5/V8/V9 guard this.
    """
    m = create_model(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["state_dict"])
    assert_structure(m)
    return m.to(device).eval()


def save_checkpoint(model, path, *, seed, epoch, val_acc_avg) -> None:
    """Write atomically: a kill partway through a 87 MB torch.save would
    otherwise leave a truncated file that loads as a crash hours later, in a
    pipeline whose whole point is surviving an unattended overnight run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "seed": seed,
        "epoch": epoch,
        "val_acc_avg": val_acc_avg,
        "timm_model": TIMM_NAME,
    }, tmp)
    tmp.replace(path)   # os.replace is atomic within a filesystem
