#!/usr/bin/env python3
"""Replace bootstrapped GT bounding boxes with the PUBLISHED ones (C1).

The verified ground truth was bootstrapped from pipeline output and then
human-verified for **topology** — the net assignments. Its bounding
boxes were never verified, and about a fifth of them are square: a
squared-off box around an elongated symbol cannot exceed roughly IoU
0.25 against a correctly-shaped detection, so the benchmark's alignment
step discards components the pipeline actually got right. Measured on
the verified GT: 8.4% of components cannot reach the 0.3 threshold
against any same-class detection, and six images match nothing at all
and score a spurious net F1 of 0.000.

Digitize-HCD already ships the correct geometry — the published COCO
boxes are human-drawn — so the fix is to use them rather than to tune
the threshold. This script projects each published box into the current
cleaned frame and substitutes it, leaving nets, terminals, classes,
``verified`` and ``annotator`` untouched: geometry only.

Safety: dry-run by default, and ``--apply`` writes to a NEW directory
rather than mutating the canonical GT. A component is only rewritten
when its class agrees with the published annotation it maps to;
disagreements are counted and left alone. The report includes the
metric that actually matters — the change in how many GT components are
matchable against real detections at the benchmark threshold.

Usage:
    python scripts/fix_gt_boxes.py
    python scripts/fix_gt_boxes.py --apply --out-dir data/gt_netlists_verified_v3
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from schematic2netlist.benchmark import iou_center
from schematic2netlist.classes import canonical_class
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.preprocess import project_bbox

COCO_PATH = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/component_annotations.json")


def squarish(bbox) -> bool:
    w, h = bbox[2], bbox[3]
    return abs(w - h) / max(w, h) < 0.05


def best_iou(bbox, cls, dets) -> float:
    best = 0.0
    for d in dets:
        if canonical_class(d["class"]) != canonical_class(cls):
            continue
        best = max(best, iou_center(
            [d["x"], d["y"], d["width"], d["height"]], bbox))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", default="data/gt_netlists_verified_v2")
    ap.add_argument("--out-dir", default="data/gt_netlists_verified_v3")
    ap.add_argument("--transforms", default="data/transforms.json")
    ap.add_argument("--coco", default=COCO_PATH)
    ap.add_argument("--det-dir", default="data/detections")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--apply", action="store_true",
                    help="write the corrected files (default: dry run)")
    args = ap.parse_args()

    transforms = json.loads(Path(args.transforms).read_text())
    coco = json.loads(Path(args.coco).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    id_by_name = {i["file_name"]: i["id"] for i in coco["images"]}

    out_dir = Path(args.out_dir)
    if args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_comps = n_rewritten = n_class_mismatch = n_no_ref = 0
    shifts, before_sq, after_sq = [], 0, 0
    iou_before, iou_after = [], []

    for f in sorted(Path(args.gt_dir).glob("circuit_*.json")):
        gt = json.loads(f.read_text())
        stem = f.stem
        meta = transforms.get(stem)
        img_id = id_by_name.get(gt.get("image", ""))
        if meta is None or img_id is None:
            n_no_ref += len(gt["components"])
            continue

        ref = [
            (cats[a["category_id"]], project_bbox(meta, *a["bbox"]))
            for a in sorted(anns_by_img[img_id], key=lambda a: a["id"])
            if cats[a["category_id"]] != "Wire Crossover"
        ]
        det_path = Path(args.det_dir) / f"{stem}.json"
        dets = load_cached_detections(det_path) if det_path.exists() else []

        n_files += 1
        for c in gt["components"]:
            n_comps += 1
            if "bbox" not in c:
                continue
            old = list(c["bbox"])
            before_sq += squarish(old)
            if dets:
                iou_before.append(best_iou(old, c["class"], dets))

            if c["id"] >= len(ref):
                n_no_ref += 1
                after_sq += squarish(old)
                if dets:
                    iou_after.append(best_iou(old, c["class"], dets))
                continue

            ref_cls, ref_box = ref[c["id"]]
            if canonical_class(ref_cls) != canonical_class(c["class"]):
                n_class_mismatch += 1
                after_sq += squarish(old)
                if dets:
                    iou_after.append(best_iou(old, c["class"], dets))
                continue

            new = [round(v, 1) for v in ref_box]
            shifts.append(max(abs(new[0] - old[0]), abs(new[1] - old[1])))
            c["bbox"] = new
            n_rewritten += 1
            after_sq += squarish(new)
            if dets:
                iou_after.append(best_iou(new, c["class"], dets))

        if args.apply:
            gt["bbox_source"] = "digitize_hcd_coco_projected"
            (out_dir / f.name).write_text(json.dumps(gt, indent=2) + "\n")

    thr = args.iou_threshold
    print(f"GT box correction ({args.gt_dir})"
          + ("" if args.apply else "  [DRY RUN — nothing written]"))
    print(f"  files {n_files}  components {n_comps}")
    print(f"  rewritten from published COCO : {n_rewritten}")
    print(f"  left alone (class mismatch)   : {n_class_mismatch}")
    print(f"  left alone (no published ref) : {n_no_ref}")
    if shifts:
        print(f"  centre shift px: median {statistics.median(shifts):.1f}  "
              f"max {max(shifts):.1f}")
    print(f"  square-ish boxes: {before_sq} -> {after_sq}")
    if iou_before and iou_after:
        mb = sum(1 for v in iou_before if v >= thr) / len(iou_before)
        ma = sum(1 for v in iou_after if v >= thr) / len(iou_after)
        print(f"  best-IoU vs detections: median "
              f"{statistics.median(iou_before):.3f} -> "
              f"{statistics.median(iou_after):.3f}")
        print(f"  matchable at IoU>={thr}: {mb:.1%} -> {ma:.1%}")
    if args.apply:
        print(f"\nwrote {n_files} files to {out_dir}")
        print("The canonical GT was NOT modified. Re-run the benchmark "
              "against the new directory and compare before adopting it.")


if __name__ == "__main__":
    main()
