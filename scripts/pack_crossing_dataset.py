#!/usr/bin/env python3
"""Pack a patch dataset into one .npz for cloud training.

The generated dataset is ~150k individual PNGs. That is fine locally but
poor for a remote GPU: tarring many small files is slow, the upload is
dominated by per-file overhead, and the training script then spends
minutes in cv2.imread before the first epoch. A single uint8 array file
uploads and loads in seconds.

Emits X_train / y_train / X_val / y_val as uint8 (patches) and int64
(labels), plus the source dataset_meta.json inline so provenance travels
with the data.

Usage:
    python scripts/pack_crossing_dataset.py \
        --data data/crossings_v2 --out data/crossings_v2.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("junction", "crossover")     # index 1 = crossover = positive


def load_split(root: Path, split: str, size: int | None):
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        for p in sorted((root / split / cls).glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if size and im.shape[0] != size:
                im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
            X.append(im)
            y.append(label)
    if not X:
        raise SystemExit(f"no patches under {root / split}")
    return np.stack(X), np.array(y, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=None,
                    help="resize patches (default: keep as generated)")
    args = ap.parse_args()

    root = Path(args.data)
    Xtr, ytr = load_split(root, "train", args.size)
    Xva, yva = load_split(root, "val", args.size)
    meta_path = root / "dataset_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    np.savez_compressed(
        args.out, X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        meta=json.dumps(meta),
    )
    mb = Path(args.out).stat().st_size / 1e6
    print(f"train {Xtr.shape} ({int((ytr == 1).sum())} crossover)")
    print(f"val   {Xva.shape} ({int((yva == 1).sum())} crossover)")
    print(f"wrote {args.out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
