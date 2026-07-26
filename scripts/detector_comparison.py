#!/usr/bin/env python3
"""Detector comparison table (Phase C / E1 size ablation).

Validates every trained run on a split and emits one row per run
(model, seed, params, mAP@0.5, mAP@0.5:0.95, precision, recall) plus
the seed mean±std for any model trained on multiple seeds.

Usage:
    python scripts/detector_comparison.py --runs-dir experiments/train_all/runs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

RUN_RE = re.compile(r"(yolov\d+[nsml])_\d+_seed(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="experiments/train_all/runs")
    ap.add_argument("--data", default="data/yolo_cleaned/dataset.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="results/detection")
    args = ap.parse_args()

    from ultralytics import YOLO

    rows = []
    for w in sorted(Path(args.runs_dir).glob("*/weights/best.pt")):
        run = w.parts[-3]
        m = RUN_RE.search(run)
        model, seed = (m.group(1), int(m.group(2))) if m else (run, -1)
        yolo = YOLO(str(w))
        n_params = sum(p.numel() for p in yolo.model.parameters())
        res = yolo.val(data=args.data, split=args.split, imgsz=args.imgsz,
                       device=args.device, verbose=False, plots=False)
        rows.append({
            "run": run, "model": model, "seed": seed,
            "params_M": round(n_params / 1e6, 2),
            "map50": round(float(res.box.map50), 4),
            "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4),
            "recall": round(float(res.box.mr), 4),
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "detector_comparison.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # seed stats per model
    seed_stats = {}
    by_model: dict[str, list] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in by_model.items():
        if len(rs) < 2:
            continue
        for key in ("map50", "map50_95", "precision", "recall"):
            vals = [r[key] for r in rs]
            seed_stats.setdefault(model, {})[key] = {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4),
                "n_seeds": len(vals),
            }
    with open(out / "seed_stats.json", "w") as f:
        json.dump(seed_stats, f, indent=2)

    print(f"[OK] {len(rows)} runs -> {out}/detector_comparison.csv")
    for r in rows:
        print(f"  {r['run']:22s} {r['params_M']:5.1f}M  "
              f"mAP50={r['map50']:.4f}  mAP50-95={r['map50_95']:.4f}")
    for model, st in seed_stats.items():
        print(f"  {model} ({st['map50']['n_seeds']} seeds): "
              f"mAP50={st['map50']['mean']:.4f}±{st['map50']['std']:.4f}  "
              f"mAP50-95={st['map50_95']['mean']:.4f}±{st['map50_95']['std']:.4f}")


if __name__ == "__main__":
    main()
