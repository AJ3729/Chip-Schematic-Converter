#!/usr/bin/env python3
"""Vote on component CLASS across polarity-preserving test-time augmentations.

Class confusion is the whole of the detection headroom worth chasing: injecting
GT classes gains +0.0263 strict success (5 images win, 0 lose, significant),
and the confusions are near-symmetric visual pairs -- MOSFET-N against MOSFET-P
16 cases, Inductor read as Resistor 7, BJT-NPN against BJT-PNP 3. Voting across
the three independently seeded detectors captured part of that: +0.0105 strict,
2 win / 0 lose, at 84.4% label precision. Three votes is simply not many, and
there are only three seeds.

More votes are available from ONE model by perturbing the input, and the reason
this is not just ``augment=True`` is domain-specific and important:

    ULTRALYTICS TTA INCLUDES A HORIZONTAL FLIP, AND A FLIP IS INADMISSIBLE HERE.

MOSFET-N against MOSFET-P and BJT-NPN against BJT-PNP are distinguished by the
direction of an arrow -- a mirrored symbol IS the other class. Flipping the
frame therefore asks the model to classify a symbol whose polarity has been
inverted, and it corrupts precisely the pairs this is meant to fix. The same
reasoning already ruled reflections out of the per-group class disambiguator.

So the augmentations here are restricted to ones that leave polarity intact:
inference scale, and small rotations well inside the deskew residual. Rotation
is applied about the frame centre and boxes are mapped back through the inverse
rotation, so every vote is expressed in original frame coordinates.

Geometry is NOT touched. Boxes come from the primary pass at the configured
image_size and are written through unchanged, exactly as the seed vote does, so
that any metric change is attributable to labels alone. A tie keeps the primary
label, because confidence is not comparable across scales either.

Usage:
    python scripts/detect_tta.py --out data/detections_1024_tta
    python scripts/detect_tta.py --out data/detections_1024_tta9 \\
        --primary data/detections_1024_vote      # compose with the seed vote
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def run_pass(model, img, imgsz, conf, deg):
    """Detections for one augmentation, in ORIGINAL frame coordinates."""
    h, w = img.shape[:2]
    if deg:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
        src = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
        Minv = cv2.invertAffineTransform(M)
    else:
        src, Minv = img, None

    res = model.predict(src, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = x2 - x1, y2 - y1
        if Minv is not None:
            p = Minv @ np.array([cx, cy, 1.0])
            cx, cy = float(p[0]), float(p[1])
            # a rotated axis-aligned box maps to a slightly larger one; the
            # centre is what matching uses, so the extent is left as measured
        out.append({"class": canonical_class(res.names[int(box.cls[0])]),
                    "confidence": float(box.conf[0]),
                    "x": cx, "y": cy, "width": bw, "height": bh})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--primary", default="data/detections_1024",
                    help="boxes and the tie-breaking label come from here")
    ap.add_argument("--scales", type=int, nargs="*", default=[544, 640, 736])
    ap.add_argument("--rotations", type=float, nargs="*", default=[-3.0, 3.0],
                    help="degrees; NO reflections -- see the module docstring")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = the whole split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    conf = cfg["detect"].get("confidence")
    base_sz = cfg["detect"]["image_size"]
    idir = Path(cfg["preprocess"]["images_dir"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(cfg["detect"]["weights"])
    try:
        model.to(args.device)
    except Exception as e:                       # pragma: no cover
        print(f"[warn] device {args.device} unavailable ({e}); using default")

    passes = [(s, 0.0) for s in args.scales] + \
             [(base_sz, r) for r in args.rotations]
    print(f"{len(passes)} augmentation passes per image: "
          f"scales {args.scales}, rotations {args.rotations} (no reflections)")

    names = [l.strip() for l in open(f"data/splits/{args.split}.txt")
             if l.strip()]
    if args.limit:
        names = names[: args.limit]
    changed = Counter()
    n_files = n_det = n_chg = n_tie = n_novote = 0
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        pp = Path(args.primary) / f"{stem}.json"
        ip = idir / nm
        if not (pp.exists() and ip.exists()):
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        cache = json.loads(pp.read_text())
        pools = [run_pass(model, img, sz, conf, deg) for sz, deg in passes]

        for det in cache["detections"]:
            n_det += 1
            if conf is not None and det.get("confidence", 1.0) < conf:
                continue
            db = (det["x"], det["y"], det["width"], det["height"])
            votes = [canonical_class(det["class"])]
            for pool in pools:
                best, bc = 0.0, None
                for x in pool:
                    o = iou(db, (x["x"], x["y"], x["width"], x["height"]))
                    if o > best:
                        best, bc = o, x["class"]
                if bc is not None and best >= args.iou:
                    votes.append(bc)
            if len(votes) < 2:
                n_novote += 1
                continue
            tally = Counter(votes)
            top, n_top = tally.most_common(1)[0]
            if n_top * 2 <= len(votes):
                n_tie += 1
                continue
            if top != canonical_class(det["class"]):
                changed[f"{canonical_class(det['class'])} -> {top}"] += 1
                det["class"] = top
                n_chg += 1
        (out / f"{stem}.json").write_text(json.dumps(cache) + "\n")
        n_files += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(names)}] relabelled {n_chg}", flush=True)

    print(f"\nwrote {n_files} caches to {out}")
    print(f"detections {n_det}, relabelled by TTA vote {n_chg} "
          f"({n_chg/max(n_det,1):.2%})")
    print(f"  no augmented box matched (kept primary): {n_novote}")
    print(f"  no majority / tie (kept primary):        {n_tie}")
    print(f"\nlabel changes:")
    for k, v in changed.most_common(20):
        print(f"  {k:34s} {v:4d}")
    print(f"\ninject_gt_classes found 67 detections whose label disagrees with")
    print(f"GT. Whether these changes land on those is for the benchmark to")
    print(f"say; scripts/inject_gt_classes.py --audit can price it first.")


if __name__ == "__main__":
    main()
