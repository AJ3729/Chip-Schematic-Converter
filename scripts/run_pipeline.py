#!/usr/bin/env python3
"""Run the full schematic-to-netlist pipeline on a single image.

Usage:
    python scripts/run_pipeline.py --image data/cleaned/circuit_1199.jpg
    python scripts/run_pipeline.py --image IMG --detections DET.json --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="path to the input image")
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument(
        "--detections",
        default=None,
        help="explicit detections JSON; default: per-image cache, then the "
        "configured detection backend",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output directory (default: experiments/runs/<image stem>)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])

    detections = None
    if args.detections:
        detections = load_cached_detections(args.detections)

    out_dir = Path(args.out) if args.out else Path("experiments/runs") / Path(args.image).stem

    result = run_pipeline(args.image, cfg, detections=detections, out_dir=out_dir)
    write_run_metadata(
        out_dir, cfg, seed,
        extra={"image": args.image,
               "detections_source": args.detections or "cache/backend"},
    )

    cov = result["coverage"]
    print(f"[INFO] wire nodes detected: {result['num_wire_nodes']}")
    print(f"[INFO] components: {cov['num_components']}")
    print(f"[INFO] terminal snap coverage: {cov['terminal_snap_rate']:.3f}")
    print(f"[INFO] fully connected coverage: {cov['fully_connected_rate']:.3f}")
    if result["netlist"] and result["netlist"]["skipped"]:
        print(f"[INFO] {len(result['netlist']['skipped'])} component(s) skipped:")
        for s in result["netlist"]["skipped"]:
            print(f"       {s}")
    print(f"[OK] artifacts written to {result['out_dir']}")


if __name__ == "__main__":
    main()
