#!/usr/bin/env python3
"""Build a detection cache with GT Wire Crossover boxes injected.

Oracle question: how much of the remaining connectivity error is missed
DRAWN crossovers (hops the detector failed to label)? The published
annotations contain 1,473 Wire Crossover boxes; this replaces each
image's predicted crossover boxes with the projected GT ones, leaving
every component detection untouched. Benchmarking against this cache
bounds the gain available from a better crossover detector — before
spending a GPU night training one.

Component detections are NOT replaced: the question is specifically
about crossover coverage, holding everything else at pipeline quality.

Usage:
    python scripts/inject_gt_crossovers.py --out data/detections_1024_gtxover
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.preprocess import project_bbox

COCO = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
        "Component Symbol and Text Label Data/component_annotations.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--coco", default=COCO)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = Path(cfg["detect"]["cache_dir"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coco = json.load(open(args.coco))
    xover_cat = next(c["id"] for c in coco["categories"]
                     if c["name"] == "Wire Crossover")
    by_name = {i["file_name"]: i["id"] for i in coco["images"]}
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        if a["category_id"] == xover_cat:
            anns.setdefault(a["image_id"], []).append(a)

    transforms = json.load(open(
        cfg["preprocess"].get("transforms_json", "data/transforms_1024.json")))
    names = [l.strip() for l in
             open(f"data/splits/{args.split}.txt") if l.strip()]

    n_pred_removed = n_gt_added = n_imgs = 0
    for nm in names:
        stem = Path(nm).stem
        cache = src / f"{stem}.json"
        if not cache.exists() or stem not in transforms:
            continue
        data = json.load(open(cache))
        dets = data.get("detections", data.get("predictions", []))
        kept = [d for d in dets
                if canonical_class(d.get("class") or d.get("class_name"))
                != "Wire Crossover"]
        n_pred_removed += len(dets) - len(kept)

        meta = transforms[stem]
        for a in anns.get(by_name.get(nm, -1), []):
            x, y, w, h = a["bbox"]
            cx, cy, bw, bh = project_bbox(meta, x, y, w, h)
            if bw <= 0 or bh <= 0:
                continue
            kept.append({"class": "Wire Crossover", "confidence": 1.0,
                         "x": cx, "y": cy, "width": bw, "height": bh})
            n_gt_added += 1

        with (out / f"{stem}.json").open("w") as f:
            json.dump({"image": nm,
                       "min_confidence": data.get("min_confidence"),
                       "detections": kept}, f, indent=2)
        n_imgs += 1

    print(f"[OK] {n_imgs} images -> {out}: removed {n_pred_removed} "
          f"predicted crossovers, injected {n_gt_added} GT crossovers")


if __name__ == "__main__":
    main()
