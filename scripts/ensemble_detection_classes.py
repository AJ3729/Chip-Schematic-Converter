#!/usr/bin/env python3
"""Decide each component's CLASS by vote across the three detector seeds.

76% of the images that cannot reach strict success are blocked by class
confusion rather than by a missed detection: the box is present and localizes
to IoU >= 0.3, only the label is wrong, and the confusions are near-symmetric
visual pairs (MOSFET-N against MOSFET-P 16 cases, Inductor read as Resistor 7,
BJT-NPN against BJT-PNP 3, I-AC as I-DC 3). 26 of the 38 blocked images are
blocked by exactly ONE component.

Training a dedicated per-group classifier on real crops already failed -- it
lost to the detector on every group, by -0.008 to -0.253 balanced accuracy,
because a 72k-parameter CNN on 64-px crops cannot match a YOLOv8s trained on
1277 images at 640 px with surrounding context.

But three independently seeded detectors already exist, and their errors on a
near-symmetric pair are the kind that need not coincide. An ensemble asks a
different question from a bigger model: not "can one model do better" but "do
the three disagree where one is wrong". That costs no training and no GPU.

Boxes come from the primary seed and are NOT changed, so localization,
detection count and everything downstream of geometry stay identical to the
baseline. Only the label moves, and only when the other seeds outvote the
primary. That keeps the intervention narrow enough to attribute: any metric
change is caused by labels alone.

Ties are left with the primary seed rather than broken arbitrarily -- a 1-1-1
three-way disagreement carries no majority, and confidence is not comparable
across independently trained models.

Usage:
    python scripts/ensemble_detection_classes.py --out data/detections_1024_vote
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config


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
    ap.add_argument("--split", default="val",
                    help="exploration/oracle-injection, so it reads val by "
                         "default; --split test only for a reported number")
    ap.add_argument("--primary", default="data/detections_1024")
    ap.add_argument("--others", nargs="*",
                    default=["data/detections_seed1_1024",
                             "data/detections_seed2_1024"])
    ap.add_argument("--iou", type=float, default=0.5,
                    help="how close another seed's box must be to count as "
                         "the same component")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    conf = cfg["detect"].get("confidence")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(f"data/splits/{args.split}.txt")
             if l.strip()]

    changed = Counter()
    n_files = n_det = n_changed = n_tie = n_novote = 0
    for nm in names:
        stem = Path(nm).stem
        pp = Path(args.primary) / f"{stem}.json"
        if not pp.exists():
            continue
        cache = json.loads(pp.read_text())
        peers = []
        for d in args.others:
            q = Path(d) / f"{stem}.json"
            if q.exists():
                peers.append([x for x in json.loads(q.read_text())["detections"]
                              if conf is None
                              or x.get("confidence", 1.0) >= conf])
        for det in cache["detections"]:
            n_det += 1
            if conf is not None and det.get("confidence", 1.0) < conf:
                continue
            db = (det["x"], det["y"], det["width"], det["height"])
            votes = [canonical_class(det["class"])]
            for peer in peers:
                best, bc = 0.0, None
                for x in peer:
                    o = iou(db, (x["x"], x["y"], x["width"], x["height"]))
                    if o > best:
                        best, bc = o, canonical_class(x["class"])
                if bc is not None and best >= args.iou:
                    votes.append(bc)
            if len(votes) < 2:
                n_novote += 1
                continue
            tally = Counter(votes)
            top, n_top = tally.most_common(1)[0]
            # a majority is required, and a tie keeps the primary: confidence
            # is not comparable across independently trained models, so there
            # is nothing principled to break a tie with
            if n_top * 2 <= len(votes):
                n_tie += 1
                continue
            if top != canonical_class(det["class"]):
                changed[f"{canonical_class(det['class'])} -> {top}"] += 1
                det["class"] = top
                n_changed += 1
        (out / f"{stem}.json").write_text(json.dumps(cache) + "\n")
        n_files += 1

    print(f"wrote {n_files} caches to {out}")
    print(f"detections {n_det}, relabelled by vote {n_changed} "
          f"({n_changed/max(n_det,1):.2%})")
    print(f"  no peer box found (kept primary): {n_novote}")
    print(f"  no majority / tie (kept primary): {n_tie}")
    print(f"\nlabel changes:")
    for k, v in changed.most_common(15):
        print(f"  {k:34s} {v:4d}")
    print(f"\nFor reference, inject_gt_classes found 67 detections whose label")
    print(f"disagrees with GT. A vote helps only if it corrects those rather")
    print(f"than churning labels that were already right, which the benchmark")
    print(f"decides -- not this count.")


if __name__ == "__main__":
    main()
