"""Re-decide component CLASS with a dedicated head, boxes untouched.

The detector localises well and classifies less well, and the two failures are
separable. mAP50 is 0.9725 while class confusion accounts for the whole of the
detection headroom worth chasing: injecting ground-truth classes is worth +0.0263
strict success (5 win / 0 lose, significant), and injecting ground-truth
CROSSOVER boxes is worth LESS THAN NOTHING. So this changes labels only.

WHY A SEPARATE HEAD BEATS THE DETECTOR AT ITS OWN CLASSES. The confusions are
near-symmetric pairs decided by a small feature -- the arrow direction in
MOSFET-N against MOSFET-P, coil against zigzag in Inductor against Resistor. The
detector infers at image_size 640 over a 1024 px frame, so a 60 px component
occupies about 37 px of network input and the arrow is a handful of pixels. A
128 px crop of the same component carries roughly 3.5x the linear detail on
exactly the discriminative feature. Measured on the same 2782 test components:

    detector    overall 0.9784   balanced 0.9583   MOSFET-N 0.7778  MOSFET-P 0.7955
    this head   overall 0.9849   balanced 0.9805   MOSFET-N 0.9259  MOSFET-P 0.9545

THE THRESHOLD IS NOT A TUNING KNOB, it is a risk setting. Strict success is a
product over every component in an image, so a wrong relabel destroys an image
that was already correct while a right one helps only where the rest of that
image is already perfect. The head is BETTER than the detector on the confusable
pairs but WORSE on some easy ones (Resistor 0.9766 against 0.9957), so relabelling
everything would lose. At 0.95 the change set is 26 corrected against 1 broken,
96.3% precision, and the benchmark gains on all three detector seeds.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from schematic2netlist.classes import canonical_class

_CACHE: dict = {}


def _crop(gray, cx, cy, w, h, size, pad_frac):
    """Square crop, padded and resized -- square so aspect ratio cannot encode
    the class (a wide box would otherwise leak "resistor")."""
    side = max(w, h) * (1.0 + pad_frac)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    H, W = gray.shape
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    sub = gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if sub.size == 0:
        return None
    if pl or pt or pr or pb:
        sub = cv2.copyMakeBorder(sub, pt, pb, pl, pr, cv2.BORDER_CONSTANT,
                                 value=255)
    return cv2.resize(sub, (size, size), interpolation=cv2.INTER_AREA)


def reclassify(detections: list[dict], gray: np.ndarray, cfg: dict) -> dict:
    """Relabel detections in place where the head disagrees confidently.

    Returns a small report. Boxes, confidences and ordering are untouched, so
    everything downstream of geometry is identical and any metric change is
    attributable to labels alone.
    """
    hcfg = (cfg.get("detect", {}) or {}).get("class_head", {}) or {}
    if not hcfg.get("enabled"):
        return {"applied": False, "n_changed": 0}

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm2d(cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm2d(cout)
            self.sc = (nn.Sequential() if stride == 1 and cin == cout else
                       nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                     nn.BatchNorm2d(cout)))

        def forward(self, x):
            o = F.relu(self.b1(self.c1(x)))
            o = self.b2(self.c2(o))
            return F.relu(o + self.sc(x))

    class Net(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, 32, 5, 2, 2, bias=False),
                                      nn.BatchNorm2d(32), nn.ReLU(),
                                      nn.MaxPool2d(2))
            self.s1, self.s2, self.s3 = Block(32, 64, 2), Block(64, 128, 2), \
                Block(128, 192, 2)
            self.head = nn.Linear(192, n_cls)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            x = self.stem(x)
            x = self.s3(self.s2(self.s1(x)))
            return self.head(self.drop(F.adaptive_avg_pool2d(x, 1).flatten(1)))

    weights = hcfg.get("weights") or "experiments/class_head/best.pt"
    device = hcfg.get("device", "cpu")
    thr = float(hcfg.get("threshold", 0.95))
    pad = float(hcfg.get("pad_frac", 0.25))

    key = (weights, device)
    if key not in _CACHE:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        m = Net(len(ck["names"]))
        m.load_state_dict(ck["state"])
        m.to(device).eval()
        _CACHE[key] = (m, [canonical_class(n) for n in ck["names"]],
                       int(ck.get("size", 128)))
    model, names, size = _CACHE[key]

    crops, idx = [], []
    for i, d in enumerate(detections):
        c = _crop(gray, d["x"], d["y"], d["width"], d["height"], size, pad)
        if c is not None:
            crops.append(c)
            idx.append(i)
    if not crops:
        return {"applied": True, "n_changed": 0}

    with torch.no_grad():
        xb = torch.from_numpy(np.array(crops)).float().div(255).unsqueeze(1)
        p = torch.softmax(model(xb.to(device)), 1).cpu().numpy()

    n_changed = 0
    changes: dict[str, int] = {}
    for k, i in enumerate(idx):
        j = int(p[k].argmax())
        if float(p[k][j]) < thr:
            continue
        cur = canonical_class(detections[i]["class"])
        if names[j] != cur:
            changes[f"{cur} -> {names[j]}"] = changes.get(
                f"{cur} -> {names[j]}", 0) + 1
            detections[i]["class"] = names[j]
            detections[i]["class_head_conf"] = float(p[k][j])
            n_changed += 1
    return {"applied": True, "n_changed": n_changed, "changes": changes,
            "threshold": thr}
