#!/usr/bin/env python3
"""Build a detection cache with CLASS LABELS corrected, boxes left predicted.

76% of the images that cannot reach strict success are blocked by class
confusion rather than missed detection: the box is present and localizes to
IoU >= 0.3, only the label is wrong, and the confusions are near-symmetric
visual pairs -- MOSFET-N against MOSFET-P (16 cases), Inductor read as
Resistor (7), BJT-NPN against BJT-PNP (3), I-AC as I-DC (3). 26 of the 38
blocked images are blocked by exactly ONE such component.

Detection mAP is 0.9725 and hides this completely, because mAP averages over
classes and never asks whether every component in an image is right -- which
is exactly what strict success demands.

This isolates the label from the box. Each predicted detection keeps its own
geometry and is relabelled with the class of the GT component it best
overlaps (IoU >= the threshold, unmatched detections untouched). Benchmarking
against the resulting cache answers what a fine-grained classifier on the
detected crop would be worth, before training one -- the same discipline that
had the crossover oracle rule out a GPU night by showing perfect crossover
boxes make strict success WORSE.

Unlike the crossover case this is not expected to be capped: the input is a
component symbol crop rather than an ambiguous wire patch, the confusable
classes carry thousands of real human-labelled boxes on the TRAIN split (327
MOSFET-N / 314 MOSFET-P, 2206 Inductor / 3186 Resistor, 650 BJT-NPN / 483
BJT-PNP, 414 I-AC / 792 I-DC over 1277 images), and training on the same
preprocessed frames the pipeline consumes avoids the mask-domain shift that
sank the render-trained crossing classifiers.

Usage:
    python scripts/inject_gt_classes.py --out data/detections_1024_gtclass
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.gt import load_gt


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(f"data/splits/{args.split}.txt") if l.strip()]

    n_relabelled = n_total = n_files = 0
    changes = {}
    for nm in names:
        stem = Path(nm).stem
        src = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
        gp = Path(cfg["benchmark"]["gt_dir"]) / f"{stem}.json"
        if not src.exists() or not gp.exists():
            continue
        cache = json.loads(src.read_text())
        gt = load_gt(str(gp))
        gcomps = gt["components"]
        for d in cache["detections"]:
            n_total += 1
            db = (d["x"], d["y"], d["width"], d["height"])
            best, bj = 0.0, None
            for g in gcomps:
                o = iou(db, g["bbox"])
                if o > best:
                    best, bj = o, g
            if bj is None or best < args.iou:
                continue
            if canonical_class(d["class"]) != canonical_class(bj["class"]):
                key = f"{canonical_class(d['class'])} -> {canonical_class(bj['class'])}"
                changes[key] = changes.get(key, 0) + 1
                d["class"] = bj["class"]
                n_relabelled += 1
        (out / f"{stem}.json").write_text(json.dumps(cache) + "\n")
        n_files += 1

    print(f"wrote {n_files} caches to {out}")
    print(f"relabelled {n_relabelled}/{n_total} detections "
          f"({n_relabelled/max(n_total,1):.2%})")
    print("\nchanges applied:")
    for k, v in sorted(changes.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k:34s} {v:4d}")


if __name__ == "__main__":
    main()
