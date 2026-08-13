#!/usr/bin/env python3
"""DEPRECATED -- do not quote its output. Use scripts/measure_runtime.py.

This script cannot answer the question it appears to answer, for two reasons
documented at the top of scripts/measure_runtime.py:

1. ``--time-detector`` is a NO-OP. It calls ``detect()``, which returns the
   per-image cache whenever one exists; every evaluated image is cached, so the
   flag times a JSON read and reports it as detector inference.
2. Its stage list is NOT the pipeline. ``time_image`` re-implements a subset of
   ``run_pipeline`` and omits the component class head and connectivity repair,
   both ENABLED in the shipped config -- and the class head is the largest
   downstream stage.

Numbers from this script were the source of a retired "~46 ms per image"
headline and a "stitch is 54% of runtime" claim that does not replicate. It is
retained only so those retractions stay reproducible.

Original docstring follows.
---------------------------------------------------------------------------
Per-stage runtime benchmark (Week-3 MSP: runtime/cost).

Times each pipeline stage separately on the test split so the paper can
report where the wall clock goes and compare against published
end-to-end figures. Detection is timed in two ways, because they answer
different questions:

- ``cached``   — the per-image detection cache is read from disk, which
  is what every experiment in this repo does (detections are computed
  once and reused so results are reproducible).
- ``inference`` — the detector actually runs, which is what a user
  experiences. Enable with ``--time-detector``; requires the weights.

Timings are single-process on the host CPU/accelerator; the machine and
library versions land in run_meta.json alongside them, because a
runtime number without its hardware is not a measurement.

Usage:
    python scripts/benchmark_runtime.py --limit 40
    python scripts/benchmark_runtime.py --limit 40 --time-detector
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import cv2

from schematic2netlist.classes import canonical_class
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.netlist import (
    assign_node_names,
    build_node_name_map,
    export_spice_netlist,
)
from schematic2netlist.nodes import (
    build_wire_nodes,
    build_wire_nodes_crossover_aware,
)
from schematic2netlist.repair import repair_circuit
from schematic2netlist.snapping import build_component_pin_nets
from schematic2netlist.textmask import detect_text_mask
from schematic2netlist.wires import (
    build_non_wire_mask,
    extract_wires,
    stitch_wire_islands,
    stitchable_mask,
)

STAGES = [
    "load", "detect", "textmask", "wiremask", "wires", "stitch",
    "nodes", "snapping", "netlist", "repair", "export",
]


def time_image(path: Path, cfg: dict, det_path: Path, model=None) -> dict:
    t: dict[str, float] = {}

    t0 = time.perf_counter()
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if model is not None:
        from schematic2netlist import detect as detect_mod
        detections = detect_mod.detect(path, cfg)
    else:
        detections = load_cached_detections(det_path)
    t["detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    text_mask = detect_text_mask(gray, cfg) if cfg["textmask"]["enabled"] else None
    t["textmask"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    non_wire_mask = build_non_wire_mask(gray, detections, cfg, text_mask)
    t["wiremask"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _cand, clean_wires = extract_wires(gray, non_wire_mask, cfg)
    t["wires"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg["wires"].get("stitch_masked_gaps"):
        stitchable = stitchable_mask(gray.shape, detections, cfg, text_mask)
        clean_wires = stitch_wire_islands(clean_wires, stitchable, cfg)
    t["stitch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg["nodes"].get("handle_crossovers"):
        boxes = [d for d in detections
                 if canonical_class(d["class"]) == "Wire Crossover"]
        node_map, _n = build_wire_nodes_crossover_aware(
            clean_wires, boxes, connectivity=cfg["nodes"]["connectivity"]
        )
    else:
        node_map, _n = build_wire_nodes(
            clean_wires, connectivity=cfg["nodes"]["connectivity"]
        )
    t["nodes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    comps = build_component_pin_nets(detections, node_map, cfg)
    t["snapping"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    name_map = build_node_name_map(
        comps, ground_fallback=cfg["netlist"]["ground_fallback"]
    )
    assign_node_names(comps, name_map)
    t["netlist"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rep = repair_circuit(comps, name_map, cfg) if cfg["repair"]["enabled"] else None
    t["repair"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sp", delete=False) as fh:
        sp = fh.name
    export_spice_netlist(
        comps, sp, placeholders=cfg["netlist"]["placeholders"],
        extra_lines=rep.extra_lines if rep else None,
    )
    Path(sp).unlink(missing_ok=True)
    t["export"] = time.perf_counter() - t0

    t["total"] = sum(t.values())
    t["n_components"] = len(comps)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default=None,
                    help="preprocessed frames; defaults to "
                         "preprocess.images_dir from the config")
    ap.add_argument("--out-dir", default="results/runtime")
    ap.add_argument("--config", default=None)
    ap.add_argument("--time-detector", action="store_true",
                    help="run the detector instead of reading its cache")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = (Path(args.splits_dir) / f"{args.split}.txt").read_text().split()
    names = names[: args.limit + args.warmup]
    images_dir = resolve_and_check(args.images_dir, names, cfg)
    det_dir = Path(cfg["detect"]["cache_dir"])

    model = True if args.time_detector else None
    rows = []
    for i, nm in enumerate(names):
        stem = Path(nm).stem
        row = time_image(
            images_dir / nm, cfg, det_dir / f"{stem}.json", model
        )
        if i < args.warmup:            # discard: first calls pay import/JIT costs
            continue
        row["image"] = nm
        rows.append(row)
        print(f"[{len(rows)}/{args.limit}] {nm} {row['total'] * 1000:.0f} ms",
              flush=True)

    write_run_metadata(out_dir, cfg, seed, extra={
        "split": args.split,
        "n_timed": len(rows),
        "warmup_discarded": args.warmup,
        "detector_timed": bool(args.time_detector),
        "machine": platform.platform(),
        "processor": platform.processor(),
    })

    with (out_dir / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image"] + STAGES
                           + ["total", "n_components"])
        w.writeheader()
        w.writerows(rows)

    summary = {"n_images": len(rows), "detector_timed": bool(args.time_detector)}
    for stage in STAGES + ["total"]:
        vals = sorted(r[stage] for r in rows)
        summary[stage] = {
            "mean_ms": round(1000 * statistics.mean(vals), 2),
            "median_ms": round(1000 * vals[len(vals) // 2], 2),
            "p90_ms": round(1000 * vals[int(0.9 * (len(vals) - 1))], 2),
            "share_of_total": None,
        }
    tot = summary["total"]["mean_ms"]
    for stage in STAGES:
        summary[stage]["share_of_total"] = round(
            summary[stage]["mean_ms"] / tot, 4
        ) if tot else None

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nper-stage runtime over {len(rows)} images "
          f"({'detector timed' if args.time_detector else 'cached detections'})")
    print(f"  {'stage':10s} {'mean ms':>9s} {'median':>9s} {'p90':>9s} {'share':>7s}")
    for stage in STAGES:
        s = summary[stage]
        print(f"  {stage:10s} {s['mean_ms']:9.2f} {s['median_ms']:9.2f} "
              f"{s['p90_ms']:9.2f} {s['share_of_total']:7.1%}")
    print(f"  {'TOTAL':10s} {summary['total']['mean_ms']:9.2f} "
          f"{summary['total']['median_ms']:9.2f} "
          f"{summary['total']['p90_ms']:9.2f}")
    print(f"\nwrote {out_dir}/summary.json + per_image.csv")


if __name__ == "__main__":
    main()
