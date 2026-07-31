#!/usr/bin/env python3
"""Component crops for a dedicated class head, from the TRAIN split.

Class confusion is the only remaining lever with a significant positive oracle:
injecting ground-truth classes is worth +0.0263 strict success (5 win / 0 lose)
and did not shrink as connectivity improved. It is also the ONLY learnable target
in this project, and that asymmetry is the reason to build here rather than
anywhere else -- net-level ground truth covers the 190 test images and nothing
else, which is why a learned connectivity model is untrainable, but component
boxes and classes exist for all 1277 images. So this trains on 895 images,
early-stops on a real 192-image validation split, and reports on test, with no
contamination anywhere.

WHY A DEDICATED HEAD SHOULD BEAT THE DETECTOR, given a previous attempt at this
lost. That attempt used a 72k-parameter CNN on 64 px crops and lost on every
group. The confusions are near-symmetric pairs decided by a small feature -- the
arrow direction in MOSFET-N against MOSFET-P (16 cases), coil against zigzag in
Inductor against Resistor (7), BJT-NPN against BJT-PNP (3). The detector runs at
image_size 640 over a 1024 px frame, so a 60 px component occupies roughly 37 px
of network input and the arrow is a handful of pixels. A crop resized to 128 px
gives about 3.5x the linear detail on exactly the discriminative feature.

AUGMENTATION IS ROTATION-ONLY, NEVER REFLECTION. A mirrored MOSFET-N *is* a
MOSFET-P; flipping would teach the model that the two are the same class, which
is the one thing it must not learn. Rotation is safe because orientation does not
change identity, and components genuinely appear at all orientations.

Usage:
    python scripts/build_class_dataset.py --out data/class_crops
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_names(ds_yaml: Path) -> list[str]:
    d = yaml.safe_load(ds_yaml.read_text())
    names = d["names"]
    return [names[i] for i in range(len(names))]


def crop_component(gray, cx, cy, w, h, size, pad_frac):
    """Square crop around a box, padded, resized. Square so aspect ratio does
    not encode the class -- a wide box would otherwise leak 'resistor'."""
    side = max(w, h) * (1.0 + pad_frac)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2))
    y1 = int(round(cy + side / 2))
    H, W = gray.shape
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
    sub = gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if sub.size == 0:
        return None
    if pad_l or pad_t or pad_r or pad_b:
        sub = cv2.copyMakeBorder(sub, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(sub, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-root", default="data/yolo_1024")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pad-frac", type=float, default=0.25)
    ap.add_argument("--out", default="data/class_crops")
    args = ap.parse_args()

    root = ROOT / args.yolo_root
    names = load_names(root / "dataset.yaml")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"names": names, "size": args.size, "pad_frac": args.pad_frac,
               "splits": {}}

    for split in ("train", "val", "test"):
        idir, ldir = root / "images" / split, root / "labels" / split
        if not idir.exists():
            continue
        X, Y, SRC = [], [], []
        cnt = Counter()
        for lp in sorted(ldir.glob("*.txt")):
            ip = None
            for ext in (".jpg", ".png", ".jpeg"):
                if (idir / (lp.stem + ext)).exists():
                    ip = idir / (lp.stem + ext)
                    break
            if ip is None:
                continue
            gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            H, W = gray.shape
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                cx, cy, bw, bh = (float(p[1]) * W, float(p[2]) * H,
                                  float(p[3]) * W, float(p[4]) * H)
                crop = crop_component(gray, cx, cy, bw, bh,
                                      args.size, args.pad_frac)
                if crop is None:
                    continue
                X.append(crop)
                Y.append(c)
                SRC.append(lp.stem)
                cnt[names[c]] += 1
        X = np.array(X, np.uint8)
        Y = np.array(Y, np.int64)
        np.savez_compressed(out / f"{split}.npz", X=X, y=Y,
                            src=np.array(SRC))
        summary["splits"][split] = {"n": int(len(X)),
                                    "per_class": dict(cnt.most_common())}
        print(f"  {split:5s} {len(X):6d} crops from {len(set(SRC))} images")

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")
    tr = summary["splits"].get("train", {}).get("per_class", {})
    if tr:
        print(f"\n  rarest train classes (these are also the most confused):")
        for k, v in sorted(tr.items(), key=lambda kv: kv[1])[:6]:
            print(f"    {k:22s} {v:5d}")


if __name__ == "__main__":
    main()
