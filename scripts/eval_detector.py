#!/usr/bin/env python3
"""Evaluate a trained YOLO detector on a split and emit detection metrics
(Phase C / C4): mAP@0.5, mAP@0.5:0.95, per-class AP WITH support counts,
and the confusion matrix → results/detection/.

Requires the 'train' extra and trained weights (training itself is a GPU
[HUMAN] step); this script only evaluates.

Usage:
    python scripts/eval_detector.py --weights experiments/train/<run>/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def support_counts(labels_dir: Path, names: list[str]) -> dict[str, int]:
    counts: Counter = Counter()
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text().split("\n"):
            if line.strip():
                counts[int(line.split()[0])] += 1
    return {names[i]: counts.get(i, 0) for i in range(len(names))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    # NOT data/yolo_cleaned: its labels predate the 2026-07-27
    # preprocessing change and score this detector at mAP@0.5 = 0.051
    # instead of 0.974. Defaulting there meant running this script bare
    # produced a plausible-looking, badly wrong number.
    ap.add_argument("--data", default="data/yolo_1024/dataset.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="results/detection_1024")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("pip install -e '.[train]'") from e

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz,
        device=args.device, plots=True, project=str(out), name="val",
    )

    names = [model.names[i] for i in range(len(model.names))]
    data_root = Path(args.data).parent
    supports = support_counts(data_root / "labels" / args.split, names)

    # Ultralytics returns per-class AP arrays indexed by ap_class_index
    # (only the classes present in the eval), NOT by full names order —
    # map them back to the class id before pairing with names/supports.
    ap50_by_cls = {
        int(ci): float(metrics.box.ap50[j])
        for j, ci in enumerate(metrics.box.ap_class_index)
    }
    ap_by_cls = {
        int(ci): float(metrics.box.ap[j].mean())
        for j, ci in enumerate(metrics.box.ap_class_index)
    }
    per_class = []
    for i, name in enumerate(names):
        per_class.append({
            "class": name, "support": supports[name],
            "ap50": ap50_by_cls.get(i), "ap50_95": ap_by_cls.get(i),
        })

    with open(out / "per_class_ap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "support", "ap50", "ap50_95"])
        w.writeheader()
        w.writerows(per_class)

    summary = {
        "weights": args.weights, "split": args.split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] mAP@0.5={summary['map50']:.3f}  mAP@0.5:0.95={summary['map50_95']:.3f}")
    print(f"[OK] per-class AP + supports -> {out}/per_class_ap.csv")
    print(f"[OK] confusion matrix + plots -> {out}/val/")


if __name__ == "__main__":
    main()
