#!/usr/bin/env python3
"""Zero-shot component detection on CGHD: does any of this transfer?

The most natural objection to a single-dataset paper is that everything was tuned
on one dataset drawn by one population. CGHD (Zenodo 10056817, CC BY 4.0) is an
independent hand-drawn schematic corpus by 25 different drafters, so running the
Digitize-HCD detector on it -- no fine-tuning, no adaptation -- answers that for
the detection stage.

WHAT THIS CAN AND CANNOT SHOW, stated plainly because the limit is real. CGHD's
Pascal VOC annotations carry only <name> and <bndbox>; there is no net or
connectivity ground truth anywhere in them. So this measures DETECTION
generalisation and nothing else. Terminal-pair F1, net F1 and strict success are
not computable on CGHD, and no claim about end-to-end transfer can be made from
it.

Two approximations, both stated rather than hidden:

  class mapping  CGHD has 53 classes against our 17. 18 map (8 of them lossily,
                 collapsing a subtype into a broader target such as
                 resistor.adjustable -> Resistor) and 35 have no counterpart.
                 The unmapped ones become IGNORE REGIONS, which is the part that
                 has to be got right: they are 6580 of the 8466 annotated objects,
                 and the detector fires on plenty of them (a transformer reads as
                 an inductor). Scoring those as false positives punishes it for
                 finding things CGHD annotates and Digitize-HCD has no class for.
                 A detection matching an unmapped object is therefore dropped
                 from the precision-recall computation entirely -- neither true
                 nor false positive -- the standard ignore-region convention. Three of our
                 classes (I-DC, I-AC, V-DC (one port)) have no CGHD counterpart
                 and are reported as absent rather than as zero.
  preprocessing  CGHD images are pushed through OUR preprocessing and the ground
                 truth boxes are projected with the same transform, so the
                 detector sees the frame geometry it was trained on. Evaluating
                 on raw photographs instead would confound a dataset shift with a
                 preprocessing shift.

AP is computed here rather than delegated, so the matching rule is explicit:
greedy assignment at IoU 0.5, highest confidence first, each ground-truth box
claimed once, and 101-point interpolated precision-recall area.

Usage:
    python scripts/cghd_zero_shot.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.preprocess import preprocess_image_meta, project_bbox


def iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def average_precision(recs, n_gt) -> float:
    """101-point interpolated AP from (confidence, is_true_positive) records."""
    if n_gt == 0:
        return float("nan")
    if not recs:
        return 0.0
    recs = sorted(recs, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs])
    fp = np.cumsum([1 - r[1] for r in recs])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 101
    return float(ap)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default="data/cghd/subset")
    ap.add_argument("--split", default="data/splits/cghd_zero_shot.txt")
    ap.add_argument("--mapping", default="data/cghd/class_mapping.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out-dir", default="results/cghd_zero_shot")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    root = ROOT / args.root
    mp = yaml.safe_load((ROOT / args.mapping).read_text())
    mapped = {k: canonical_class(v["target"])
              for k, v in (mp.get("mapped") or {}).items()}
    lossy = {k for k, v in (mp.get("mapped") or {}).items() if v.get("lossy")}

    names = [l.strip() for l in open(ROOT / args.split) if l.strip()]
    if args.limit:
        names = names[: args.limit]

    from ultralytics import YOLO
    model = YOLO(cfg["detect"]["weights"])
    try:
        model.to("mps")
    except Exception:
        pass
    conf_floor = 0.05          # low, so AP integrates the whole PR curve

    per_class = defaultdict(list)
    n_gt = defaultdict(int)
    n_img = n_ignored = n_drafters = n_dropped = 0
    drafters = set()
    for i, rel in enumerate(names, 1):
        img_p = root / rel
        ann_p = root / rel.replace("/images/", "/annotations/")
        ann_p = ann_p.with_suffix(".xml")
        if not (img_p.exists() and ann_p.exists()):
            continue
        drafters.add(rel.split("/")[0])
        canvas, meta = preprocess_image_meta(str(img_p), cfg)

        gts, ignore = [], []
        for obj in ET.parse(ann_p).getroot().findall("object"):
            raw = (obj.findtext("name") or "").strip()
            if raw not in mapped:
                n_ignored += 1
                bb = obj.find("bndbox")
                x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
                x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
                cx, cy, w, h = project_bbox(meta, x1, y1, x2 - x1, y2 - y1)
                ignore.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
                continue
            bb = obj.find("bndbox")
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            cx, cy, w, h = project_bbox(meta, x1, y1, x2 - x1, y2 - y1)
            gts.append((mapped[raw],
                        (cx - w/2, cy - h/2, cx + w/2, cy + h/2), False))
            n_gt[mapped[raw]] += 1

        res = model.predict(canvas, conf=conf_floor,
                            imgsz=cfg["detect"]["image_size"], verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append((canonical_class(res.names[int(b.cls[0])]),
                         float(b.conf[0]), (x1, y1, x2, y2)))
        dets.sort(key=lambda d: -d[1])

        claimed = [False] * len(gts)
        for cls, cf, box in dets:
            best, bj = args.iou, -1
            for j, (gc, gb, _) in enumerate(gts):
                if claimed[j] or gc != cls:
                    continue
                o = iou_xyxy(box, gb)
                if o >= best:
                    best, bj = o, j
            if bj >= 0:
                claimed[bj] = True
                per_class[cls].append((cf, 1))
                continue
            # unmatched: drop it if it landed on something CGHD annotates but
            # our vocabulary has no class for, rather than call it a mistake
            if any(iou_xyxy(box, ib) >= args.iou for ib in ignore):
                n_dropped += 1
                continue
            per_class[cls].append((cf, 0))
        n_img += 1
        if i % 20 == 0:
            print(f"  [{i}/{len(names)}] images={n_img}", flush=True)

    aps = {c: average_precision(per_class.get(c, []), n_gt[c])
           for c in sorted(n_gt) if n_gt[c] > 0}
    present = {c: v for c, v in aps.items() if not np.isnan(v)}
    macro = float(np.mean(list(present.values()))) if present else float("nan")

    print(f"\n=== ZERO-SHOT DETECTION ON CGHD ===")
    print(f"{n_img} images, {len(drafters)} drafters, "
          f"{sum(n_gt.values())} mapped GT boxes; {n_ignored} unmapped CGHD "
          f"objects became ignore regions,\nand {n_dropped} detections landing "
          f"on them were dropped rather than scored as false positives.\n")
    print(f"  {'class':22s} {'gt':>5s} {'AP50':>7s}  note")
    for c in sorted(present, key=lambda k: -present[k]):
        note = "lossy mapping" if any(
            k in lossy and mapped[k] == c for k in mapped) else ""
        print(f"  {c:22s} {n_gt[c]:5d} {present[c]:7.4f}  {note}")
    absent = [c for c in ("I-DC", "I-AC", "V-DC (one port)")
              if n_gt.get(c, 0) == 0]
    print(f"\n  macro AP50 over {len(present)} classes present: {macro:.4f}")
    print(f"  classes with no CGHD counterpart (not scored): "
          f"{', '.join(absent) or 'none'}")
    print(f"\n  IN-DOMAIN reference (Digitize-HCD test, 3 seeds): "
          f"mAP50 0.9753 +- 0.0026")
    print(f"\n  This is DETECTION transfer only. CGHD has no net or connectivity")
    print(f"  annotation, so strict success is not computable here and no claim")
    print(f"  about end-to-end transfer follows from it.")

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "n_images": n_img, "n_drafters": len(drafters),
        "n_gt_boxes": int(sum(n_gt.values())), "n_ignored_objects": n_ignored, "n_detections_dropped": n_dropped,
        "iou": args.iou, "macro_ap50": macro,
        "per_class_ap50": present, "gt_counts": dict(n_gt),
        "absent_classes": absent,
        "in_domain_map50": 0.9753,
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
