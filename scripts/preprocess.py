#!/usr/bin/env python3
"""Preprocess raw schematic photos into cleaned 512x512 images.

Single image (writes the result and, with --show, a raw|cleaned
side-by-side you can open):
    python scripts/preprocess.py --image data/raw/circuit_42.jpg --show

...with the published annotation boxes drawn on, to confirm nothing is
cropped out of frame (green = inside canvas, red = off-canvas):
    python scripts/preprocess.py --image data/raw/circuit_42.jpg --show --annotations

Whole directory:
    python scripts/preprocess.py [--raw-dir data/raw] [--clean-dir data/cleaned]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from schematic2netlist.config import load_config
from schematic2netlist.preprocess import (
    preprocess_image,
    preprocess_image_meta,
    project_bbox,
)

COCO_PATH = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/component_annotations.json")


def _load_boxes(coco_path: str, file_name: str) -> list:
    """Published COCO boxes for one image (original coordinates)."""
    coco = json.load(open(coco_path))
    by_id = defaultdict(list)
    for a in coco["annotations"]:
        by_id[a["image_id"]].append(a["bbox"])
    for i in coco["images"]:
        if i["file_name"] == file_name:
            return by_id[i["id"]]
    return []


def run_single(args, cfg) -> None:
    src = Path(args.image)
    boxes = []
    if args.annotations:
        boxes = _load_boxes(args.coco, src.name)

    result = preprocess_image_meta(str(src), cfg, ann_boxes=boxes or None)
    if result is None:
        raise SystemExit(f"[FAIL] could not read {src}")
    canvas, meta = result

    out = Path(args.out) if args.out else Path("experiments/preprocess") / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)

    print(f"[OK] {src}  ->  {out}")
    print(f"     skew {meta['angle_deg']:+.2f} deg | rot90 {meta['rotated90']} | "
          f"crop {meta['crop']} | scale {meta['scale']:.4f}")

    vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if boxes:
        T = meta["target_size"]
        outside = 0
        for b in boxes:
            cx, cy, w, h = project_bbox(meta, *b)
            inside = 0 <= cx < T and 0 <= cy < T
            outside += not inside
            cv2.rectangle(vis, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)),
                          (0, 170, 0) if inside else (0, 0, 255), 2)
        print(f"     annotations: {len(boxes)} total, {outside} off-canvas"
              + ("  <-- PROBLEM" if outside else "  (all inside)"))

    if args.show:
        raw = cv2.imread(str(src))
        T = canvas.shape[0]
        scale = T / max(raw.shape[:2])
        rw, rh = int(raw.shape[1] * scale), int(raw.shape[0] * scale)
        raw_fit = np.full((T, T, 3), 255, np.uint8)
        rs = cv2.resize(raw, (rw, rh), interpolation=cv2.INTER_AREA)
        raw_fit[(T - rh) // 2:(T - rh) // 2 + rh, (T - rw) // 2:(T - rw) // 2 + rw] = rs
        cv2.putText(raw_fit, "RAW", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 0, 0), 2)
        cv2.putText(vis, "CLEANED", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 0), 2)
        cmp_path = out.with_name(out.stem + "_compare.png")
        cv2.imwrite(str(cmp_path), np.hstack([raw_fit, np.full((T, 12, 3), 255, np.uint8), vis]))
        print(f"[OK] side-by-side -> {cmp_path}")
        print(f"     open it:  open {cmp_path}")


def run_batch(args, cfg) -> None:
    os.makedirs(args.clean_dir, exist_ok=True)
    images = [f for f in sorted(os.listdir(args.raw_dir))
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="preprocess a single image")
    ap.add_argument("--out", default=None,
                    help="single-image output path (default experiments/preprocess/<name>)")
    ap.add_argument("--show", action="store_true",
                    help="also write a raw|cleaned side-by-side PNG")
    ap.add_argument("--annotations", action="store_true",
                    help="overlay published COCO boxes on the cleaned image")
    ap.add_argument("--coco", default=COCO_PATH)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--clean-dir", default="data/cleaned")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.image:
        run_single(args, cfg)
    else:
        run_batch(args, cfg)


if __name__ == "__main__":
    main()
