#!/usr/bin/env python3
"""How much do the headline numbers depend on the alignment threshold?

The benchmark aligns predicted components to GT by bounding-box IoU
within class (default 0.3). That threshold is a measurement choice, and
here it is load-bearing for a reason worth stating plainly:

**GT boxes were bootstrapped from pipeline output and then verified by a
human for TOPOLOGY — the net assignments — not for box geometry.** About
20% of them are square, while a real horizontal resistor detection is
roughly 42x16 px. A square box circumscribing an elongated symbol tops
out around IoU 0.25 against a correct detection, so it falls below the
threshold and the component is scored as unmatched even though the
prediction is right. Measured over the verified GT: 8.4% of components
cannot reach IoU 0.3 against any same-class detection, and six images
cannot match a single component, scoring a spurious 0.000.

This script runs the pipeline ONCE per image and scores the same
predictions at several thresholds, so the sensitivity is measured
rather than argued. Report the curve in the paper; it is the honest
answer to a reviewer asking whether the numbers are a threshold
artifact.

Usage:
    python scripts/threshold_sensitivity.py
    python scripts/threshold_sensitivity.py --thresholds 0.1,0.2,0.3,0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from schematic2netlist.benchmark import score_prediction
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.gt import load_gt
from schematic2netlist.pipeline import run_pipeline

import sys

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import gt_components, pred_components  # noqa: E402

METRICS = ["terminal_pair_f1", "net_f1", "per_component_connected_acc",
           "nged", "strict_success"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--gt-dir", default=None,
                    help="overrides benchmark.gt_dir from the config")
    ap.add_argument("--out-dir", default="results/threshold_sensitivity")
    ap.add_argument("--config", default=None)
    ap.add_argument("--thresholds", default="0.1,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    args.gt_dir = args.gt_dir or cfg["benchmark"]["gt_dir"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, cfg, seed, extra={"split": args.split})

    thresholds = [float(t) for t in args.thresholds.split(",")]
    det_dir = Path(cfg["detect"]["cache_dir"])
    names = (Path(args.splits_dir) / f"{args.split}.txt").read_text().split()
    if args.limit:
        names = names[: args.limit]

    per_thr: dict[float, list[dict]] = {t: [] for t in thresholds}
    rows = []
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gt_path = Path(args.gt_dir) / f"{stem}.json"
        det_path = det_dir / f"{stem}.json"
        if not gt_path.exists() or not det_path.exists():
            continue
        gt = load_gt(gt_path)
        if not gt.get("verified"):
            continue
        print(f"[{i}/{len(names)}] {nm}", flush=True)

        result = run_pipeline(
            Path(args.images_dir) / nm, cfg,
            detections=load_cached_detections(det_path),
        )
        pred, gtc = pred_components(result), gt_components(gt)

        row = {"image": nm}
        for t in thresholds:
            scored = score_prediction(pred, gtc, iou_threshold=t)
            per_thr[t].append(scored)
            row[f"net_f1@{t}"] = round(scored["net_f1"], 4)
            row[f"matched@{t}"] = scored["matched"]
        rows.append(row)

    summary = []
    for t in thresholds:
        s = per_thr[t]
        entry = {"iou_threshold": t, "n_images": len(s)}
        for m in METRICS:
            entry[m] = round(statistics.mean(float(r[m]) for r in s), 4)
        entry["mean_matched"] = round(
            statistics.mean(r["matched"] for r in s), 2)
        entry["mean_unmatched_gt"] = round(
            statistics.mean(r["unmatched_gt"] for r in s), 2)
        entry["images_with_zero_matches"] = sum(1 for r in s if r["matched"] == 0)
        summary.append(entry)

    with (out_dir / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "summary.json").open("w") as fh:
        json.dump({"thresholds": summary}, fh, indent=2)
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\nalignment-threshold sensitivity ({len(rows)} images)")
    print(f"  {'IoU':>5s} {'netF1':>7s} {'tpF1':>7s} {'perComp':>8s} "
          f"{'strict':>7s} {'unmatchedGT':>12s} {'zero-match imgs':>16s}")
    for e in summary:
        print(f"  {e['iou_threshold']:5.2f} {e['net_f1']:7.4f} "
              f"{e['terminal_pair_f1']:7.4f} "
              f"{e['per_component_connected_acc']:8.4f} "
              f"{e['strict_success']:7.4f} {e['mean_unmatched_gt']:12.2f} "
              f"{e['images_with_zero_matches']:16d}")
    print(f"\nwrote {out_dir}/summary.json + summary.csv + per_image.csv")


if __name__ == "__main__":
    main()
