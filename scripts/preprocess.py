#!/usr/bin/env python3
"""Batch-preprocess raw schematic photos into cleaned 512x512 binarized
images.

Usage:
    python scripts/preprocess.py [--raw-dir data/raw] [--clean-dir data/cleaned]
"""

from __future__ import annotations

import argparse
import os

import cv2
from tqdm import tqdm

from schematic2netlist.config import load_config
from schematic2netlist.preprocess import preprocess_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--clean-dir", default="data/cleaned")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.clean_dir, exist_ok=True)

    images = [
        f
        for f in sorted(os.listdir(args.raw_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    failed = []
    for img_name in tqdm(images, desc="Preprocessing images"):
        processed = preprocess_image(os.path.join(args.raw_dir, img_name), cfg)
        if processed is None:
            failed.append(img_name)
            continue
        cv2.imwrite(os.path.join(args.clean_dir, img_name), processed)

    print(f"[OK] preprocessed {len(images) - len(failed)}/{len(images)} images")
    if failed:
        print(f"[WARN] failed to load: {failed}")


if __name__ == "__main__":
    main()
