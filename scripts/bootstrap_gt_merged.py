#!/usr/bin/env python3
"""Re-bootstrap GT topology files by MERGING two sources (Phase C prep):

- Component inventory (identity, class, bbox) comes from the PUBLISHED
  Digitize-HCD COCO annotations — authoritative classes and complete
  coverage, projected into cleaned-image coordinates via the recorded
  preprocessing transforms.
- Net assignments come from the current pipeline's terminal snapping,
  transferred by IoU-matching each projected published box against the
  pipeline's detections (legacy hosted-detector output).

This gives the annotator a component list that is already complete and
correctly classified, so the human pass is mostly about checking NETS.
Wire Crossover annotations are excluded (drawing aids, not electrical
components). Files are written as unverified `source:
"coco+pipeline_bootstrap"`, replacing earlier pipeline-only bootstraps
that carried detector-vocabulary classes and missed classes the hosted
model could not emit.

Usage:
    python scripts/bootstrap_gt_merged.py --images data/splits/test.txt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from schematic2netlist.classes import class_role, class_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import SCHEMA_VERSION, save_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.preprocess import project_bbox

COCO_PATH = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/component_annotations.json")


def iou_center_boxes(a, b) -> float:
    """IoU of two center-based (cx, cy, w, h) boxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/splits/test.txt")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--gt-dir", default="data/gt_netlists")
    ap.add_argument("--transforms", default="data/transforms.json")
    ap.add_argument("--coco", default=COCO_PATH)
    ap.add_argument("--config", default=None)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    det_dir = Path(cfg["detect"]["cache_dir"])
    gt_dir = Path(args.gt_dir)

    coco = json.load(open(args.coco))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_name = {i["file_name"]: i["id"] for i in coco["images"]}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    transforms = json.load(open(args.transforms))
    names = Path(args.images).read_text().split()

    written, matched_total, comp_total = 0, 0, 0
    skipped: list[str] = []
    for name in names:
        stem = Path(name).stem
        meta = transforms.get(stem)
        det_path = det_dir / (stem + ".json")
        if meta is None or not meta.get("verified_byte_identical") or not det_path.exists():
            skipped.append(name)
            continue

        # pipeline nets on the cleaned image
        detections = load_cached_detections(det_path)
        result = run_pipeline(Path(args.images_dir) / name, cfg, detections=detections)
        pipe_comps = result["components"]
        pipe_boxes = [
            (detections[c["id"]]["x"], detections[c["id"]]["y"],
             detections[c["id"]]["width"], detections[c["id"]]["height"])
            for c in pipe_comps
        ]

        components = []
        new_id = 0
        for ann in sorted(anns_by_image[img_by_name[name]], key=lambda a: a["id"]):
            cls = cats[ann["category_id"]]
            if class_role(cls) == "none":
                continue  # Wire Crossover: not an electrical component
            bx = project_bbox(meta, *ann["bbox"])

            # transfer nets from the best-overlapping pipeline component
            best_iou, best = 0.0, None
            for pc, pb in zip(pipe_comps, pipe_boxes):
                i = iou_center_boxes(bx, pb)
                if i > best_iou:
                    best_iou, best = i, pc
            nets = [None] * class_terminals(cls)
            if best is not None and best_iou >= args.iou_threshold:
                names_from_pipe = best.get("node_names", [])
                for i in range(min(len(nets), len(names_from_pipe))):
                    nets[i] = names_from_pipe[i]
                matched_total += 1

            components.append({
                "id": new_id,
                "class": cls,
                "bbox": [round(v, 1) for v in bx],
                "terminals": [
                    {"index": i, "net": nets[i]} for i in range(len(nets))
                ],
            })
            new_id += 1
        comp_total += len(components)

        save_gt(
            {
                "schema_version": SCHEMA_VERSION,
                "image": name,
                "source": "coco+pipeline_bootstrap",
                "verified": False,
                "annotator": None,
                "notes": "",
                "components": components,
            },
            gt_dir / (stem + ".json"),
        )
        written += 1

    print(f"[OK] wrote {written} merged GT file(s) "
          f"({comp_total} components, {matched_total} with transferred nets)")
    if skipped:
        print(f"[WARN] skipped {len(skipped)}: missing transform/verification/"
              f"detections (first 5: {skipped[:5]})")


if __name__ == "__main__":
    main()
