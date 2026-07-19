#!/usr/bin/env python3
"""Record the preprocessing transform for every raw image (Phase C
prerequisite).

Re-runs preprocessing on data/raw with transform capture and verifies
that the produced JPEG is byte-identical to the existing data/cleaned
file — proving the recorded transform describes exactly the transform
that produced the cleaned set. Writes data/transforms.json
(stem -> transform meta) plus a verification report.

Usage:
    python scripts/record_transforms.py [--raw-dir data/raw] [--clean-dir data/cleaned]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from schematic2netlist.config import load_config
from schematic2netlist.preprocess import preprocess_image_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--clean-dir", default="data/cleaned")
    ap.add_argument("--out", default="data/transforms.json")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    images = sorted(
        f for f in os.listdir(args.raw_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if args.limit:
        images = images[: args.limit]

    transforms: dict[str, dict] = {}
    identical, differing, missing = 0, [], []
    for name in tqdm(images, desc="Recording transforms"):
        result = preprocess_image_meta(os.path.join(args.raw_dir, name), cfg)
        if result is None:
            missing.append(name)
            continue
        canvas, meta = result

        clean_path = Path(args.clean_dir) / name
        if clean_path.exists():
            ok, buf = cv2.imencode(Path(name).suffix, canvas)
            if ok and buf.tobytes() == clean_path.read_bytes():
                identical += 1
                meta["verified_byte_identical"] = True
            else:
                differing.append(name)
                meta["verified_byte_identical"] = False
        else:
            missing.append(name)
        transforms[Path(name).stem] = meta

    with open(args.out, "w") as f:
        json.dump(transforms, f)

    print(f"[OK] {len(transforms)} transforms -> {args.out}")
    print(f"[VERIFY] byte-identical reproductions: {identical}/{len(images)}")
    if differing:
        print(f"[WARN] {len(differing)} differ from existing cleaned files "
              f"(first 5: {differing[:5]})")
    if missing:
        print(f"[WARN] {len(missing)} missing/unloadable (first 5: {missing[:5]})")


if __name__ == "__main__":
    main()
