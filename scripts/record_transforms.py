#!/usr/bin/env python3
"""Re-run preprocessing on data/raw, writing cleaned images + the exact
geometric transform for each (Phase C prerequisite).

Two guards run on every image:

1. **Annotation containment (hard failure).** Every published COCO box is
   projected through the recorded transform; any box whose centre lands
   outside the canvas is reported. Preprocessing is annotation-aware, so
   this MUST be 0 — it is the assertion that would have caught the
   "annotations cropped out of frame" bug (831 boxes / 343 images before
   the fix). Non-zero ⇒ non-zero exit status.
2. **Reproduction check (informational).** Compares against any existing
   data/cleaned file; a mismatch is expected and fine when preprocessing
   has intentionally changed.

Usage:
    python scripts/record_transforms.py                  # verify only
    python scripts/record_transforms.py --write-images   # regenerate data/cleaned
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from schematic2netlist.config import load_config
from schematic2netlist.preprocess import preprocess_image_meta, project_bbox

COCO_PATH = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/component_annotations.json")


def load_annotations(coco_path: str) -> dict[str, list]:
    """image file_name -> [ [x, y, w, h], ... ] in original coordinates."""
    coco = json.load(open(coco_path))
    by_id = defaultdict(list)
    for a in coco["annotations"]:
        by_id[a["image_id"]].append(a["bbox"])
    return {i["file_name"]: by_id[i["id"]] for i in coco["images"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--clean-dir", default="data/cleaned")
    ap.add_argument("--coco", default=COCO_PATH)
    ap.add_argument("--out", default="data/transforms.json")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write-images", action="store_true",
                    help="overwrite data/cleaned with the new output")
    ap.add_argument("--no-annotation-aware", action="store_true",
                    help="diagnostic: crop without consulting annotations")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ann_by_image = load_annotations(args.coco)

    images = sorted(
        f for f in os.listdir(args.raw_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if args.limit:
        images = images[: args.limit]

    transforms: dict[str, dict] = {}
    identical, differing, unreadable = 0, [], []
    boxes_total, boxes_outside, images_with_outside = 0, 0, []

    for name in tqdm(images, desc="Preprocessing"):
        ann = ann_by_image.get(name, [])
        result = preprocess_image_meta(
            os.path.join(args.raw_dir, name), cfg,
            ann_boxes=None if args.no_annotation_aware else ann,
        )
        if result is None:
            unreadable.append(name)
            continue
        canvas, meta = result

        # --- guard 1: annotation containment ---
        target = meta["target_size"]
        outside_here = 0
        for bbox in ann:
            cx, cy, _, _ = project_bbox(meta, *bbox)
            boxes_total += 1
            if not (0 <= cx < target and 0 <= cy < target):
                outside_here += 1
        if outside_here:
            boxes_outside += outside_here
            images_with_outside.append(name)
        meta["boxes_outside_canvas"] = outside_here

        # --- guard 2: reproduction check ---
        clean_path = Path(args.clean_dir) / name
        if clean_path.exists():
            ok, buf = cv2.imencode(Path(name).suffix, canvas)
            if ok and buf.tobytes() == clean_path.read_bytes():
                identical += 1
                meta["matches_existing_cleaned"] = True
            else:
                differing.append(name)
                meta["matches_existing_cleaned"] = False

        if args.write_images:
            Path(args.clean_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clean_path), canvas)

        transforms[Path(name).stem] = meta

    with open(args.out, "w") as f:
        json.dump(transforms, f)

    print(f"\n[OK] {len(transforms)} transforms -> {args.out}")
    print(f"[INFO] reproduces existing data/cleaned byte-identically: "
          f"{identical}/{len(images)}"
          + ("  (differences expected after a preprocessing change)"
             if differing else ""))
    if unreadable:
        print(f"[WARN] unreadable: {len(unreadable)} (first 5: {unreadable[:5]})")

    print(f"[GUARD] annotation containment: {boxes_outside} of {boxes_total} "
          f"boxes outside canvas, across {len(images_with_outside)} image(s)")
    if boxes_outside:
        print(f"[FAIL] annotations are being cropped out of frame "
              f"(first 5: {images_with_outside[:5]})")
        sys.exit(1)
    print("[PASS] every annotated component lies inside the cleaned canvas")


if __name__ == "__main__":
    main()
