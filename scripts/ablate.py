#!/usr/bin/env python3
"""Run one ablation axis: sweep a dotted config key over values and
evaluate each setting (Phase E).

Each run is one row keyed by config hash in
experiments/ablations/<axis>.csv, with full per-image results under
experiments/ablations/<axis>/<value>/.

Usage:
    python scripts/ablate.py --axis wires.min_blob_area --values 10,20,40,60,100
    python scripts/ablate.py --axis snapping.strategy --values directional,uniform
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from schematic2netlist.config import config_hash, load_config, set_by_dotted_key

from evaluate import evaluate  # scripts/evaluate.py


def parse_value(raw: str):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", required=True, help="dotted config key, e.g. wires.min_blob_area")
    ap.add_argument("--values", required=True, help="comma-separated values to sweep")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="experiments/ablations")
    ap.add_argument("--no-ngspice", action="store_true")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    values = [parse_value(v) for v in args.values.split(",")]

    axis_dir = Path(args.out_dir) / args.axis.replace(".", "_")
    axis_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for value in values:
        cfg = set_by_dotted_key(base_cfg, args.axis, value)
        run_dir = axis_dir / str(value)
        print(f"\n=== {args.axis} = {value} (config {config_hash(cfg)}) ===")
        summary = evaluate(
            cfg,
            images_dir=args.images_dir,
            out_dir=run_dir,
            limit=args.limit,
            with_ngspice=not args.no_ngspice,
        )
        results.append({"axis": args.axis, "value": value, **summary})

    csv_path = axis_dir.with_suffix(".csv")
    fields = [k for k in results[0] if k != "failure_reasons"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] ablation table written to {csv_path}")


if __name__ == "__main__":
    main()
