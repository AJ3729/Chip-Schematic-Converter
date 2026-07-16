#!/usr/bin/env python3
"""Batch pipeline evaluation.

Replaces the legacy evaluate_pipeline.py, which was broken in two ways:
(a) it applied ONE shared detections.json (from circuit_1199.jpg) to
    every image — here detections are loaded per image from the cache
    directory keyed by image stem;
(b) it read det["class_name"] while the JSON stored "class", so every
    image failed with KeyError — detections are now normalized on load.

Images without cached detections are recorded as "missing_detections"
rather than silently scored.

Coverage statistics (terminal_snap_rate, fully_connected_rate) measure
only whether snapping returned something, not whether it was correct;
ground-truth metrics land with Phase B/D.

Usage:
    python scripts/evaluate.py --images-dir data/cleaned --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from schematic2netlist.config import config_hash, load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.netlist import GroundNotFoundError
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.simulate import run_ngspice

FIELDS = [
    "image",
    "num_components",
    "num_wire_nodes",
    "terminal_snap_rate",
    "fully_connected_rate",
    "netlist_generated",
    "ngspice_ok",
    "failure_reason",
]


def evaluate(
    cfg: dict,
    images_dir: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    detections_dir: str | Path | None = None,
    with_ngspice: bool = True,
) -> dict:
    """Evaluate the pipeline over a directory of images.

    Writes per-image rows to <out_dir>/evaluation.csv, netlists to
    <out_dir>/netlists/, and an aggregate to <out_dir>/summary.json.
    Returns the summary dict.
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    netlist_dir = out_dir / "netlists"
    netlist_dir.mkdir(parents=True, exist_ok=True)
    seed = set_global_seed(cfg["seed"])
    write_run_metadata(
        out_dir, cfg, seed,
        extra={"images_dir": str(images_dir), "limit": limit},
    )
    det_dir = Path(detections_dir) if detections_dir else Path(cfg["detect"]["cache_dir"])

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if limit:
        images = images[:limit]

    rows = []
    for idx, img_path in enumerate(images):
        print(f"[{idx + 1}/{len(images)}] {img_path.name}")
        row = {
            "image": img_path.name,
            "num_components": 0,
            "num_wire_nodes": 0,
            "terminal_snap_rate": 0.0,
            "fully_connected_rate": 0.0,
            "netlist_generated": 0,
            "ngspice_ok": 0,
            "failure_reason": "",
        }

        det_path = det_dir / (img_path.stem + ".json")
        if not det_path.exists():
            row["failure_reason"] = "missing_detections"
            rows.append(row)
            continue

        try:
            detections = load_cached_detections(det_path)
            netlist_out = netlist_dir / img_path.stem
            result = run_pipeline(
                img_path, cfg, detections=detections, out_dir=netlist_out
            )
            cov = result["coverage"]
            row["num_components"] = cov["num_components"]
            row["num_wire_nodes"] = result["num_wire_nodes"]
            row["terminal_snap_rate"] = cov["terminal_snap_rate"]
            row["fully_connected_rate"] = cov["fully_connected_rate"]
            row["netlist_generated"] = 1

            if with_ngspice:
                ok, reason = run_ngspice(str(netlist_out / "netlist.sp"), cfg)
                row["ngspice_ok"] = int(ok)
                row["failure_reason"] = "" if ok else reason
        except GroundNotFoundError:
            row["failure_reason"] = "no_ground_node"
        except Exception as e:  # noqa: BLE001 — record, don't crash the batch
            row["failure_reason"] = type(e).__name__
        rows.append(row)

    csv_path = out_dir / "evaluation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    evaluated = [r for r in rows if r["netlist_generated"]]
    n_eval = len(evaluated)
    summary = {
        "config_hash": config_hash(cfg),
        "num_images": len(rows),
        "num_evaluated": n_eval,
        "num_missing_detections": sum(
            1 for r in rows if r["failure_reason"] == "missing_detections"
        ),
        "mean_terminal_snap_rate": (
            sum(r["terminal_snap_rate"] for r in evaluated) / n_eval if n_eval else 0.0
        ),
        "mean_fully_connected_rate": (
            sum(r["fully_connected_rate"] for r in evaluated) / n_eval if n_eval else 0.0
        ),
        "ngspice_ok_rate": (
            sum(r["ngspice_ok"] for r in evaluated) / n_eval if n_eval else 0.0
        ),
        "failure_reasons": {},
    }
    for r in rows:
        if r["failure_reason"]:
            summary["failure_reasons"][r["failure_reason"]] = (
                summary["failure_reasons"].get(r["failure_reason"], 0) + 1
            )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] wrote {csv_path} and summary.json ({n_eval}/{len(rows)} evaluated)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--config", default=None)
    ap.add_argument("--detections-dir", default=None,
                    help="default: detect.cache_dir from config")
    ap.add_argument("--out-dir", default="experiments/eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ngspice", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    evaluate(
        cfg,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        detections_dir=args.detections_dir,
        with_ngspice=not args.no_ngspice,
    )


if __name__ == "__main__":
    main()
