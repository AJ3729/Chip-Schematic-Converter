#!/usr/bin/env python3
"""Repair evaluation (C5 MSP, BUILD-F): solvability lift + minimality
from a benchmark run's artifacts, plus (--verify) the recomputed
topology-preservation proof and ground-choice gauge accuracy vs GT.

Usage:
    python scripts/benchmark_repair.py --run-dir results/v5_stitch_crossover
    python scripts/benchmark_repair.py --run-dir results/v5_stitch_crossover \
        --verify            # re-runs the pipeline per image (minutes)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.repair import repair_circuit
from schematic2netlist.repair_eval import (
    aggregate_repair,
    check_topology_preserved,
    ground_choice_accuracy,
)

# scripts/benchmark.py owns these adapters; duplicated signature-free here
from importlib import import_module
import sys

sys.path.insert(0, str(Path(__file__).parent))
_bench = import_module("benchmark")
pred_components = _bench.pred_components
gt_components = _bench.gt_components


def verify_pass(args, cfg) -> tuple[list[dict], dict]:
    """Per-image recompute: topology preservation + ground accuracy."""
    det_dir = Path(cfg["detect"]["cache_dir"])
    names = (Path(args.splits_dir) / f"{args.split}.txt").read_text().split()
    if args.limit:
        names = names[: args.limit]

    rows: list[dict] = []
    for idx, name in enumerate(names, 1):
        stem = Path(name).stem
        gt_path = Path(args.gt_dir) / (stem + ".json")
        det_path = det_dir / (stem + ".json")
        if not gt_path.exists() or not det_path.exists():
            continue
        gt = load_gt(gt_path)
        if not gt.get("verified"):
            continue
        print(f"[{idx}/{len(names)}] {name}", flush=True)

        detections = load_cached_detections(det_path)
        # run WITHOUT the pipeline's internal repair so the snapshot
        # brackets exactly one repair invocation below
        cfg_norepair = json.loads(json.dumps(cfg))
        cfg_norepair["repair"]["enabled"] = False
        result = run_pipeline(
            Path(args.images_dir) / name, cfg_norepair, detections=detections
        )
        comps = result["components"]

        before_nets = [list(c.get("node_names", [])) for c in comps]
        rep = repair_circuit(comps, result["node_name_map"], cfg)
        violations = check_topology_preserved(comps, rep, before_nets)

        gacc = ground_choice_accuracy(
            pred_components(result), gt_components(gt), args.iou_threshold
        )
        gnd_symbol = any(e.issue == "ground_selection" for e in rep.entries)
        rows.append({
            "image": name,
            "topology_violations": ";".join(violations),
            "n_extra_lines": len(rep.extra_lines),
            "ground_case": "gauge_gnd_symbol" if gnd_symbol else "assumed",
            "ground_mapped_gt_net": gacc["mapped_gt_net"] if gacc else "",
            "ground_ambiguous": "" if gacc is None else int(
                bool(gacc.get("ambiguous"))
            ),
            "ground_correct": (
                "" if gacc is None else int(gacc["correct"])
            ),
        })

    n_viol = sum(1 for r in rows if r["topology_violations"])
    summary: dict = {
        "verified_images": len(rows),
        "topology_violations": n_viol,
    }
    # Ground-choice accuracy is reported three ways, because an ambiguous
    # tie is not evidence of a wrong choice: the strict rate counts ties
    # as failures (lower bound), the resolved rate excludes them
    # (accuracy where the question is decidable), and the tie count is
    # reported so neither number can be quoted without its caveat.
    for case in ("gauge_gnd_symbol", "assumed"):
        scored = [
            r for r in rows if r["ground_case"] == case and r["ground_correct"] != ""
        ]
        resolved = [r for r in scored if not r["ground_ambiguous"]]
        summary[f"ground_accuracy_{case}_strict"] = (
            round(sum(r["ground_correct"] for r in scored) / len(scored), 4)
            if scored else None
        )
        summary[f"ground_accuracy_{case}_resolved"] = (
            round(sum(r["ground_correct"] for r in resolved) / len(resolved), 4)
            if resolved else None
        )
        summary[f"ground_n_{case}"] = len(scored)
        summary[f"ground_ambiguous_{case}"] = len(scored) - len(resolved)
        summary[f"ground_unanswerable_{case}"] = sum(
            1 for r in rows
            if r["ground_case"] == case and r["ground_correct"] == ""
        )
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/v5_stitch_crossover")
    ap.add_argument("--out-dir", default="results/repair")
    ap.add_argument("--verify", action="store_true",
                    help="re-run pipeline per image for the topology proof "
                         "+ ground gauge accuracy (minutes)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--gt-dir", default=None,
                    help="overrides benchmark.gt_dir from the config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    args.gt_dir = args.gt_dir or cfg["benchmark"]["gt_dir"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, cfg, seed, extra={"run_dir": args.run_dir})

    summary = {"source_run": args.run_dir}
    summary.update(aggregate_repair(args.run_dir))

    if args.verify:
        rows, vsummary = verify_pass(args, cfg)
        summary.update(vsummary)
        with (out_dir / "verify_per_image.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
