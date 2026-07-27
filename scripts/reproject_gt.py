#!/usr/bin/env python3
"""Re-project GT component bboxes between two preprocessing generations.

The verified ground truth is topology (nets, classes, terminal order) —
all image-geometry-independent, so it survives a preprocessing change
untouched. Only the ``bbox`` field is frame-dependent, and the benchmark
uses it solely to align predicted components to GT (Hungarian by IoU
within class). This script migrates those bboxes:

    old cleaned frame --unproject(old)--> original --project(new)--> new frame

Nothing else in the file is modified: nets, terminals, classes,
``verified``, ``annotator`` and ``notes`` are all preserved byte-for-byte
in content. A cross-check re-derives each box straight from the published
COCO annotations through the new transform and reports agreement, so a
silent mis-migration cannot pass unnoticed.

Usage:
    python scripts/reproject_gt.py \
        --gt-dir data/gt_netlists_verified \
        --old-transforms backups/transforms_old.json \
        --new-transforms data/transforms.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from schematic2netlist.preprocess import project_bbox, unproject_point

COCO_PATH = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/component_annotations.json")


def reproject(bbox_center, old_meta, new_meta):
    """(cx, cy, w, h) in old cleaned frame -> same in new cleaned frame."""
    cx, cy, w, h = bbox_center
    corners = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
               (cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2)]
    orig = [unproject_point(old_meta, px, py) for px, py in corners]
    xs = [p[0] for p in orig]
    ys = [p[1] for p in orig]
    ox, oy = min(xs), min(ys)
    ow, oh = max(xs) - ox, max(ys) - oy
    return project_bbox(new_meta, ox, oy, ow, oh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", default="data/gt_netlists_verified")
    ap.add_argument("--old-transforms", default="backups/transforms_old.json")
    ap.add_argument("--new-transforms", default="data/transforms.json")
    ap.add_argument("--coco", default=COCO_PATH)
    ap.add_argument("--out-dir", default=None,
                    help="write migrated copies here instead of in place "
                    "(keeps the verified originals read-only/untouched)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    old_tf = json.load(open(args.old_transforms))
    new_tf = json.load(open(args.new_transforms))

    coco = json.load(open(args.coco))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    id_by_name = {i["file_name"]: i["id"] for i in coco["images"]}

    files = sorted(Path(args.gt_dir).glob("circuit_*.json"))
    n_files, n_boxes, skipped = 0, 0, []
    agree, checked, worst = [], 0, 0.0

    for f in files:
        gt = json.load(open(f))
        stem = f.stem
        om, nm = old_tf.get(stem), new_tf.get(stem)
        if om is None or nm is None:
            skipped.append(f.name)
            continue

        # cross-check reference: published boxes through the NEW transform,
        # in the same order the bootstrap assigned component ids
        ref = [project_bbox(nm, *a["bbox"])
               for a in sorted(anns_by_img[id_by_name[gt["image"]]],
                               key=lambda a: a["id"])
               if cats[a["category_id"]] != "Wire Crossover"]

        for c in gt["components"]:
            if "bbox" not in c:
                continue
            new_bbox = reproject(c["bbox"], om, nm)
            if c["id"] < len(ref):
                d = max(abs(new_bbox[0] - ref[c["id"]][0]),
                        abs(new_bbox[1] - ref[c["id"]][1]))
                agree.append(d)
                worst = max(worst, d)
                checked += 1
            c["bbox"] = [round(v, 1) for v in new_bbox]
            n_boxes += 1

        if not args.dry_run:
            gt["bbox_frame"] = "cleaned_v2"
            dest = (out_dir / f.name) if out_dir else f
            with open(dest, "w") as fh:
                json.dump(gt, fh, indent=2)
        n_files += 1

    mean_d = sum(agree) / len(agree) if agree else 0.0
    print(f"[OK] re-projected {n_boxes} bboxes across {n_files} GT files"
          + ("  (dry run — nothing written)" if args.dry_run else ""))
    print(f"[CHECK] vs COCO-through-new-transform on {checked} boxes: "
          f"mean |Δcentre| {mean_d:.2f} px, worst {worst:.2f} px")
    if skipped:
        print(f"[WARN] skipped {len(skipped)} (missing transform): {skipped[:5]}")
    if worst > 5.0:
        print("[WARN] worst-case disagreement above 5 px — inspect before trusting")


if __name__ == "__main__":
    main()
