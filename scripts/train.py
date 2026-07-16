#!/usr/bin/env python3
"""Train a local YOLO detector on the frozen splits (Phase C).

Requires the 'train' extra:  pip install -e '.[train]'

Usage:
    python scripts/train.py --data data/dataset.yaml --model yolov8s.pt \
        --imgsz 640 --epochs 100
"""

from __future__ import annotations

import argparse

from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed, write_run_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. mps, cpu, 0")
    ap.add_argument("--project", default="experiments/train")
    ap.add_argument("--name", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    write_run_metadata(
        args.project, cfg, seed,
        extra={"data": args.data, "model": args.model,
               "imgsz": args.imgsz, "epochs": args.epochs},
    )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install -e '.[train]'"
        ) from e

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=seed,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
