"""Single evaluation pass over a cached split."""
from __future__ import annotations

import time

import torch

from .data import normalize_uint8
from .metrics import compute_metrics


@torch.no_grad()
def evaluate_cached(model, cache, device, batch_size: int = 64) -> dict:
    """Evaluate from a uint8 cache. Returns the full metric dict + timing."""
    model.eval()  # drop-path/dropout noise would masquerade as a compression effect
    images, y, groups = cache["images"], cache["y"], cache["group"]

    t0 = time.time()
    preds = torch.empty(len(y), dtype=torch.long)
    for i in range(0, len(y), batch_size):
        x = normalize_uint8(images[i:i + batch_size], device)
        preds[i:i + batch_size] = model(x).argmax(dim=1).cpu()
    if device.type == "mps":
        torch.mps.synchronize()

    m = compute_metrics(preds, y, groups)
    m["eval_seconds"] = time.time() - t0
    m["pred_mean"] = float(preds.float().mean())  # collapse detector for V7b
    return m


@torch.no_grad()
def evaluate_dataset(model, dataset, device, batch_size: int = 64) -> dict:
    """Evaluate straight from the Dataset (JPEG decode each time).

    Only used to prove the cache is faithful (Gate 3); the sweep uses the cache.
    """
    from torch.utils.data import DataLoader

    from .metrics import group_index

    model.eval()
    # num_workers=0: MPS + multiprocessing workers hangs intermittently.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    preds, ys, gs = [], [], []
    for x, y, metadata in loader:  # WILDS-style 3-tuple, not (x, y)
        preds.append(model(x.to(device)).argmax(dim=1).cpu())
        ys.append(y)
        gs.append(group_index(metadata))

    return compute_metrics(torch.cat(preds), torch.cat(ys), torch.cat(gs))
