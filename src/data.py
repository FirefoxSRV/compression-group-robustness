"""Waterbirds loading, group indexing, and uint8 tensor caches.

Deliberately does NOT depend on the `wilds` package. WILDS 2.0.0 is a 2021
release that drags in `ogb` and was written against pandas 1.x; this project
runs Python 3.13 / pandas 3.x. We read the same `metadata.csv` from the same
tarball ourselves.

Equivalence to the WILDS loader is provable rather than assumed: WILDS builds
`_metadata_array = stack((place, y), dim=1)` with
`_metadata_fields = ['background', 'y']`, which is exactly what `metadata()`
returns here -- and V1/V2 assert the resulting group counts.
"""
from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .metrics import GROUP_NAMES, N_GROUPS, group_index  # noqa: F401  (re-export)

DOWNLOAD_URL = (
    "https://worksheets.codalab.org/rest/bundles/"
    "0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"
)
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.3333)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Same geometry as eval_tf but stopping before Normalize, so the cache holds
# raw uint8. Normalizing here and again at eval time is the double-normalize
# trap; caching pre-normalization is what avoids it.
cache_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


def ensure_data(data_dir: str | Path, download: bool = True) -> Path:
    """Download + extract Waterbirds if `metadata.csv` is not already present."""
    data_dir = Path(data_dir)
    if (data_dir / "metadata.csv").exists():
        return data_dir
    if not download:
        raise FileNotFoundError(f"{data_dir}/metadata.csv missing and download=False")

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir.parent / "waterbirds_v1.0.tar.gz"
    if not archive.exists():
        # curl -C - so an interrupted 489 MB download resumes instead of restarting.
        subprocess.run(
            ["curl", "-L", "-C", "-", "--retry", "3", "-o", str(archive), DOWNLOAD_URL],
            check=True,
        )
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(data_dir)
    archive.unlink(missing_ok=True)  # reclaim 489 MB; disk is tight
    return data_dir


class WaterbirdsSplit(Dataset):
    """One official split. Yields 3-tuples (x, y, metadata_row), like WILDS."""

    def __init__(self, data_dir, split: str, transform=None):
        if split not in SPLIT_CODES:
            raise ValueError(f"unknown split {split!r}")
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        df = pd.read_csv(self.data_dir / "metadata.csv")
        df = df[df["split"] == SPLIT_CODES[split]].reset_index(drop=True)
        self.df = df

        self.filenames = df["img_filename"].tolist()
        self.y = torch.as_tensor(df["y"].to_numpy().copy()).long()
        self.place = torch.as_tensor(df["place"].to_numpy().copy()).long()
        # Column order matches WILDS metadata_fields == ['background', 'y'].
        self._metadata = torch.stack([self.place, self.y], dim=1)

    def __len__(self) -> int:
        return len(self.filenames)

    def metadata(self) -> torch.Tensor:
        return self._metadata

    def groups(self) -> torch.Tensor:
        return group_index(self._metadata)

    def raw_image(self, idx: int) -> Image.Image:
        return Image.open(self.data_dir / self.filenames[idx]).convert("RGB")

    def __getitem__(self, idx: int):
        x = self.raw_image(idx)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx], self._metadata[idx]


def group_counts(groups: torch.Tensor) -> list[int]:
    return [int(groups.eq(g).sum()) for g in range(N_GROUPS)]


# --------------------------------------------------------------------------
# uint8 eval cache
# --------------------------------------------------------------------------

def build_eval_cache(data_dir, split: str, out_path, log_every: int = 1000) -> dict:
    """Decode a split once at 224 (post-CenterCrop) and store as raw uint8.

    The test split is evaluated ~30x during the sweep; JPEG decode would
    otherwise dominate every pass.
    """
    ds = WaterbirdsSplit(data_dir, split, transform=None)
    n = len(ds)
    images = torch.empty((n, 3, 224, 224), dtype=torch.uint8)

    to_uint8 = transforms.PILToTensor()
    for i in range(n):
        images[i] = to_uint8(cache_tf(ds.raw_image(i)))
        if log_every and (i + 1) % log_every == 0:
            print(f"  cached {i + 1}/{n}", flush=True)

    cache = {
        "images": images,          # (N,3,224,224) uint8, PRE-normalization
        "y": ds.y.clone(),
        "group": ds.groups().clone(),
        "split": split,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    return cache


def load_eval_cache(path, mmap: bool | None = None) -> dict:
    """Load a uint8 cache, memory-mapped by default.

    Why mmap matters here. test_cache.pt is 832 MB and train_cache.pt 688 MB,
    and a plain load makes them ANONYMOUS memory: on a 16 GB machine the kernel
    can only relieve pressure by writing those pages to swap. macOS swap lives
    on the data volume, and on the night of 2026-10-26 it grew to 33 GB and
    filled the disk twice, costing one arm and several hours.

    Memory-mapped, the same pages are FILE-BACKED and clean, so the kernel
    simply drops them under pressure and re-reads from disk if they are needed
    again -- no swap write at all. Measured on test_cache.pt: peak RSS 1025 MB
    plain against 241 MB mapped.

    Access is read-only and strided (`images[i:i+batch]` in evaluate_cached),
    which is exactly the pattern mmap suits. Set mmap=False to force a full
    read, or PHASE2_NO_MMAP=1 in the environment to do it globally.
    """
    if mmap is None:
        mmap = os.environ.get("PHASE2_NO_MMAP", "0") == "0"
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, TypeError, ValueError) as e:
        # An older archive format, or a torch without mmap support, is a reason
        # to be slower -- not a reason to fail a 9-hour run.
        if not mmap:
            raise
        print(f"[data] mmap load of {path} failed ({type(e).__name__}: {e}); "
              f"falling back to a full read", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)


def normalize_uint8(batch_uint8: torch.Tensor, device) -> torch.Tensor:
    """uint8 (B,3,224,224) -> normalized float32 on `device`. Normalizes ONCE."""
    x = batch_uint8.to(device).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def get_device() -> torch.device:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def main() -> None:
    """Download Waterbirds (if needed) and build the uint8 caches.

    val and test are the evaluation caches. `--splits train` additionally
    builds the TRAIN cache, which Phase 2 needs: difficulty scoring requires a
    deterministic, un-augmented pass over the training split, which is exactly
    what `cache_tf` (Resize 256 + CenterCrop 224, no flip, no RandomResizedCrop)
    gives. It is NOT used for training itself -- training still reads JPEGs
    through `train_tf` every epoch, because a fixed cache would defeat the
    augmentation.

    Cache row i corresponds to `WaterbirdsSplit(data_dir, split)[i]`: both read
    the same metadata.csv, filter on the same split code, and `reset_index`.
    Phase 2's prune indices are indices into that shared ordering, so the
    correspondence is load-bearing -- gate V10c asserts it.
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--data-dir", default="data/waterbirds_v1.0")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    data_dir = ensure_data(args.data_dir, download=not args.no_download)
    for split in args.splits:
        out = Path(args.results_dir) / f"{split}_cache.pt"
        if out.exists():
            print(f"{out} exists, skipping")
            continue
        print(f"building {split} cache -> {out}")
        cache = build_eval_cache(data_dir, split, out)
        counts = group_counts(cache["group"])
        print(f"  {split}: n={len(cache['y'])} group counts {counts}")
        # An empty group would make worst-group accuracy silently NaN (gate V2).
        assert all(c > 0 for c in counts), f"{split} has an empty group: {counts}"


if __name__ == "__main__":
    main()
