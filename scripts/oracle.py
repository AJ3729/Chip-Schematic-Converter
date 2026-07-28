#!/usr/bin/env python3
"""Oracle / GT-injection stage attribution (contribution C4, plan E5).

Runs the pipeline in escalating "cheat" modes and reports the metric
cascade for each. The deltas attribute end-to-end error to a specific
stage instead of speculation:

  A  predicted        detector + wire extraction + snapping   (baseline)
  B  GT detections    perfect boxes/classes, rest predicted   -> detection error = B - A
  C  GT wire mask     perfect boxes AND perfect connectivity
                      geometry, snapping still predicted      -> wire error   = C - B
  D  all GT           sanity ceiling, must score 1.0          -> snapping err = D - C

Mode C synthesises perfect connectivity from the ground-truth graph
(:mod:`schematic2netlist.oracle_render`): nets are routed as orthogonal
conductors that avoid foreign component bodies, and each pin gets an
outward stub, so snapping only has to find what is unambiguously there.
Whatever it still gets wrong is snapping's own error.

**Only images whose render passes verification are scored in mode C**
(every net routed, every pin carrying its own net's label, no foreign
net crossing a component body). The count of excluded images is
reported; A/B/D are additionally reported restricted to that same
subset, because a stage delta computed across different image sets is
not a stage delta. The earlier star-topology renderer produced an
impossible negative wire attribution — see oracle_render's module
docstring — so mode C numbers should not be quoted without this check.

Usage:
    python scripts/oracle.py --limit 60
    python scripts/oracle.py --limit 190 --out-dir results/oracle
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import cv2

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import (
    net_level_metrics,
    per_component_connected_accuracy,
    terminal_pair_metrics,
)
from schematic2netlist.oracle_render import render_gt_node_map
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.snapping import build_component_pin_nets


def gt_detections(gt: dict, extra: list[dict] | None = None) -> list[dict]:
    """GT components as detection dicts (perfect boxes + classes).

    ``extra`` carries through detections that GT cannot supply. This
    matters for ``Wire Crossover``: it is a drawing annotation, not an
    electrical component, so it has no entry in a GT topology file.
    Building mode B from GT alone therefore silently DELETED the
    crossover boxes that crossover-aware net assembly consumes, and the
    'perfect detections' mode scored worse than the baseline — an
    artifact of the harness, not a property of the detector.
    """
    dets = [{
        "class": c["class"], "confidence": 1.0,
        "x": c["bbox"][0], "y": c["bbox"][1],
        "width": c["bbox"][2], "height": c["bbox"][3],
    } for c in gt["components"]]
    return dets + list(extra or [])


def score(pred_comps, gt_comps):
    p, g, _ = align_components(pred_comps, gt_comps)
    p = canonicalize_terminals(p)
    g = canonicalize_terminals(g)
    return (terminal_pair_metrics(p, g)["f1"],
            net_level_metrics(p, g)["f1"],
            per_component_connected_accuracy(p, g))


def as_pred(components, dets):
    return [{
        "id": c["id"], "class": c["class"],
        "nets": list(c.get("node_names", [])),
        "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                 dets[c["id"]]["width"], dets[c["id"]]["height"]],
    } for c in components]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--gt-dir", default=None,
                    help="overrides benchmark.gt_dir from the config")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--out-dir", default="results/oracle")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    gt_dir = args.gt_dir or cfg["benchmark"]["gt_dir"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, cfg, seed, extra={"gt_dir": gt_dir})

    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()][: args.limit]
    rows: list[dict] = []

    for i, nm in enumerate(names, 1):
        stem = nm.rsplit(".", 1)[0]
        gt = load_gt(f"{gt_dir}/{stem}.json")
        gcomps = gt_to_components(gt)
        by_id = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by_id[c["id"]]["bbox"]

        img_path = f"{args.images_dir}/{nm}"
        print(f"[{i}/{len(names)}] {nm}", flush=True)

        # A — everything predicted
        pred_dets = load_cached_detections(f"data/detections/{stem}.json")
        rA = run_pipeline(img_path, cfg, detections=pred_dets)
        a = score(as_pred(rA["components"], rA["detections"]), gcomps)

        # B — GT component boxes (+ predicted crossovers, which GT has
        # no entries for), predicted wires + snapping
        crossovers = [
            d for d in pred_dets
            if canonical_class(d["class"]) == "Wire Crossover"
        ]
        gdets = gt_detections(gt, extra=crossovers)
        rB = run_pipeline(img_path, cfg, detections=gdets)
        b = score(as_pred(rB["components"], gdets), gcomps)

        # C — GT detections + GT-routed connectivity, predicted snapping.
        # Scored only if the synthetic wiring verifies (see module docs).
        img = cv2.imread(img_path)
        node_map, _labels, render_report = render_gt_node_map(gt, img.shape)
        c_valid = render_report["ok"]
        if c_valid:
            comps = build_component_pin_nets(gdets, node_map, cfg)
            for comp in comps:
                comp["node_names"] = [
                    None if n is None else f"n{n}" for n in comp["nodes"]
                ]
            c = score(as_pred(comps, gdets), gcomps)
        else:
            c = (None, None, None)

        # D — all GT (sanity ceiling)
        d = score(gcomps, gcomps)

        rows.append({
            "image": nm,
            "c_render_ok": int(c_valid),
            "c_unrouted_nets": len(render_report["unrouted_nets"]),
            "c_bad_pins": len(render_report["pins_with_wrong_label"]),
            "c_foreign_intrusions": len(render_report["components_with_foreign_net"]),
            **{f"A_{k}": v for k, v in zip(("tp_f1", "net_f1", "percomp"), a)},
            **{f"B_{k}": v for k, v in zip(("tp_f1", "net_f1", "percomp"), b)},
            **{f"C_{k}": ("" if v is None else v)
               for k, v in zip(("tp_f1", "net_f1", "percomp"), c)},
            **{f"D_{k}": v for k, v in zip(("tp_f1", "net_f1", "percomp"), d)},
        })

    with (out_dir / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["c_render_ok"]]
    labels = {
        "A": "A predicted (baseline)",
        "B": "B + GT detections",
        "C": "C + GT connectivity",
        "D": "D all GT (ceiling)",
    }

    def mean_of(subset, mode, metric):
        vals = [r[f"{mode}_{metric}"] for r in subset if r[f"{mode}_{metric}"] != ""]
        return st.mean(vals) if vals else float("nan")

    print(f"\noracle attribution over {len(rows)} images (GT: {gt_dir})")
    print(f"mode C scored on {len(valid)}/{len(rows)} images whose synthetic "
          f"wiring verified; {len(rows) - len(valid)} excluded.\n")

    # The waterfall is computed on the mode-C-valid subset only: a delta
    # taken across different image sets is not a stage delta.
    print(f"{'mode':26s} {'term-pair F1':>13s} {'net F1':>8s} {'per-comp':>9s}"
          f"   (n={len(valid)})")
    means = {}
    for k in ("A", "B", "C", "D"):
        t = mean_of(valid, k, "tp_f1")
        n = mean_of(valid, k, "net_f1")
        p = mean_of(valid, k, "percomp")
        means[k] = (t, n, p)
        print(f"{labels[k]:26s} {t:13.4f} {n:8.4f} {p:9.4f}")

    summary = {
        "n_images": len(rows),
        "n_mode_c_valid": len(valid),
        "means_on_valid_subset": {
            k: dict(zip(("tp_f1", "net_f1", "percomp"), means[k]))
            for k in ("A", "B", "C", "D")
        },
        "attribution_tp_f1": {
            "detection": means["B"][0] - means["A"][0],
            "wires": means["C"][0] - means["B"][0],
            "snapping": means["D"][0] - means["C"][0],
        },
        "full_set_means": {
            k: {m: mean_of(rows, k, m)
                for m in ("tp_f1", "net_f1", "percomp")}
            for k in ("A", "B", "D")
        },
    }

    print("\nerror attributed to each stage (terminal-pair F1):")
    for stage, key in (("detection", "detection"), ("wires", "wires"),
                       ("snapping", "snapping")):
        print(f"  {stage:15s}{summary['attribution_tp_f1'][key]:+.4f}")

    if means["D"][0] < 0.999:
        print(f"\n[WARN] ceiling is {means['D'][0]:.4f}, expected 1.0 — "
              "the harness itself is lossy; investigate before trusting deltas")
    if summary["attribution_tp_f1"]["wires"] < 0:
        print("\n[WARN] negative wire attribution — perfect connectivity "
              "scored WORSE than predicted wires. The synthetic render is "
              "unreadable by snapping; do not quote these deltas.")

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_dir}/summary.json + per_image.csv")


if __name__ == "__main__":
    main()
