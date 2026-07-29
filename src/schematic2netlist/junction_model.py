"""Learned junction-vs-crossover decision at stroke intersections (C2).

Connected components answers "do these strokes touch?" when the
question net assembly actually needs answered is "do they CONNECT?".
Those differ exactly at a crossing. Measurement says the gap is large:
of ~20 stroke intersections per image the detector labels 2.2 as
`Wire Crossover`, and 72.6% of the wire nodes carrying component
terminals fuse two or more ground-truth nets
(``results/intersections/``). Deterministic tuning cannot close it —
sweeping every stitching guard bought ~+0.02 terminal-pair F1 against
+0.47 of headroom.

This module applies the CNN trained by ``scripts/train_junction.py`` on
CGHD's `junction` / `crossover` annotations. It is inference only: given
a wire mask and a list of intersection sites, return per-site
probabilities that the site is a CROSSING.

**The threshold is a policy, not a detail.** The two errors are not
symmetric. Calling a crossing a junction WELDS two nets — one mistake
corrupts every component on both. Calling a junction a crossing SPLITS
one net, which is more localized and often recoverable downstream. The
default therefore sits below 0.5, biased toward predicting "crossing",
and lives in config so it can be tuned against the benchmark rather
than guessed.

Torch is imported lazily: the pipeline must remain usable, on the
classical path, in an environment without it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .skeleton import crop_site


def _build_model(size: int):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 16), block(16, 32), block(32, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(64, 2),
    )


@lru_cache(maxsize=4)
def load_model(weights: str):
    """Load a checkpoint once and keep it; returns (model, patch_size)."""
    import torch

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    size = int(ckpt.get("size", 64))
    model = _build_model(size)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, size


def crossing_probabilities(
    wire_mask: np.ndarray, sites: list[tuple[int, int]], weights: str,
    context: float = 3.0, batch: int = 256, thin_input: bool = False,
) -> np.ndarray:
    """P(site is a crossing) for each site, in the order given.

    ``context`` sets the crop half-width as a multiple of a nominal
    stroke neighbourhood, mirroring how training patches were cropped
    around annotated boxes — get this wrong and the model sees a
    different object than it was trained on.

    ``thin_input`` skeletonizes the mask before cropping. The classifier
    was trained on CGHD photographs binarized to ~1-2 px strokes; our
    1024-px pipeline masks are 3-5 px thick after morphology and
    stitching, and that thickness is a measured domain gap — on real
    inference patches, skeletonizing lifts crossing-vs-junction AUC from
    0.72 to 0.80 (scripts/diagnose_junction.py). It affects ONLY what
    the classifier sees; the mask used to build nodes is untouched.
    """
    import torch

    if not sites:
        return np.zeros(0, dtype=np.float32)
    model, size = load_model(weights)
    half = max(4, int(round(size * context / 8)))

    if thin_input:
        from .skeleton import thin
        wire_mask = (thin(wire_mask) * 255).astype(np.uint8)

    patches = np.stack([
        crop_site(wire_mask, x, y, half, size) for x, y in sites
    ]).astype(np.float32) / 255.0

    out = []
    with torch.no_grad():
        for i in range(0, len(patches), batch):
            xb = torch.from_numpy(patches[i:i + batch]).unsqueeze(1)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].numpy())
    return np.concatenate(out)
