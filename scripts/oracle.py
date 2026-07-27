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

Mode C synthesises a wire mask from the ground-truth graph: every pair
of terminals sharing a net is joined by a drawn polyline, so the mask
encodes exactly the right connectivity. Snapping must then only find
what is unambiguously there; whatever it still gets wrong is snapping's
own error.

Usage:
    python scripts/oracle.py --limit 60
"""

from __future__ import annotations

import argparse
import statistics as st

import cv2
import numpy as np

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import class_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import (
    net_level_metrics,
    per_component_connected_accuracy,
    terminal_pair_metrics,
)
from schematic2netlist.nodes import build_wire_nodes
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.snapping import build_component_pin_nets


def gt_detections(gt: dict) -> list[dict]:
    """GT components as detection dicts (perfect boxes + classes)."""
    return [{
        "class": c["class"], "confidence": 1.0,
        "x": c["bbox"][0], "y": c["bbox"][1],
        "width": c["bbox"][2], "height": c["bbox"][3],
    } for c in gt["components"]]


def render_gt_wire_mask(gt: dict, shape, thickness: int = 3) -> np.ndarray:
    """Synthesise a wire mask that encodes the GT connectivity exactly.

    Terminals are placed on the component bbox edge (spread along the
    long axis), and all terminals on a net are joined via the net's
    centroid — a star topology, which yields one connected component per
    net, which is what connectivity requires.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    by_net: dict[str, list] = {}
    for c in gt["components"]:
        cx, cy, w, h = c["bbox"]
        n_t = max(1, class_terminals(c["class"]))
        horiz = w >= h
        for t in c["terminals"]:
            i = t["index"]
            if t["net"] is None:
                continue
            # spread terminals along the long axis, on the bbox edge
            frac = (i + 0.5) / n_t
            if n_t == 1:
                px, py = cx, cy + h / 2
            elif horiz:
                px = cx - w / 2 + w * frac if n_t > 2 else (cx - w / 2 if i == 0 else cx + w / 2)
                py = cy
            else:
                px = cx
                py = cy - h / 2 + h * frac if n_t > 2 else (cy - h / 2 if i == 0 else cy + h / 2)
            by_net.setdefault(t["net"], []).append((float(px), float(py)))

    for pts in by_net.values():
        if len(pts) == 1:
            p = (int(pts[0][0]), int(pts[0][1]))
            cv2.circle(mask, p, thickness, 255, -1)
            continue
        hub = (int(st.mean(p[0] for p in pts)), int(st.mean(p[1] for p in pts)))
        for p in pts:
            cv2.line(mask, (int(p[0]), int(p[1])), hub, 255, thickness)
    return mask


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
    ap.add_argument("--gt-dir", default="data/gt_netlists_verified_v2")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()][: args.limit]
    acc = {k: [] for k in ("A", "B", "C", "D")}

    for nm in names:
        stem = nm.rsplit(".", 1)[0]
        gt = load_gt(f"{args.gt_dir}/{stem}.json")
        gcomps = gt_to_components(gt)
        by_id = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by_id[c["id"]]["bbox"]

        img_path = f"{args.images_dir}/{nm}"

        # A — everything predicted
        rA = run_pipeline(img_path, cfg,
                          detections=load_cached_detections(f"data/detections/{stem}.json"))
        acc["A"].append(score(as_pred(rA["components"], rA["detections"]), gcomps))

        # B — GT detections, predicted wires + snapping
        gdets = gt_detections(gt)
        rB = run_pipeline(img_path, cfg, detections=gdets)
        acc["B"].append(score(as_pred(rB["components"], gdets), gcomps))

        # C — GT detections + GT-derived wire mask, predicted snapping
        img = cv2.imread(img_path)
        wire = render_gt_wire_mask(gt, img.shape)
        node_map, _ = build_wire_nodes(wire, connectivity=cfg["nodes"]["connectivity"])
        comps = build_component_pin_nets(gdets, node_map, cfg)
        for c in comps:
            c["node_names"] = [None if n is None else f"n{n}" for n in c["nodes"]]
        acc["C"].append(score(as_pred(comps, gdets), gcomps))

        # D — all GT (sanity ceiling)
        acc["D"].append(score(gcomps, gcomps))

    labels = {
        "A": "A predicted (baseline)",
        "B": "B + GT detections",
        "C": "C + GT wire mask",
        "D": "D all GT (ceiling)",
    }
    print(f"oracle attribution over {len(names)} images "
          f"(GT: {args.gt_dir})\n")
    print(f"{'mode':26s} {'term-pair F1':>13s} {'net F1':>8s} {'per-comp':>9s}")
    means = {}
    for k in ("A", "B", "C", "D"):
        t = st.mean(x[0] for x in acc[k])
        n = st.mean(x[1] for x in acc[k])
        p = st.mean(x[2] for x in acc[k])
        means[k] = (t, n, p)
        print(f"{labels[k]:26s} {t:13.4f} {n:8.4f} {p:9.4f}")

    print("\nerror attributed to each stage (terminal-pair F1):")
    print(f"  detection      {means['B'][0] - means['A'][0]:+.4f}")
    print(f"  wire extraction{means['C'][0] - means['B'][0]:+.4f}")
    print(f"  snapping       {means['D'][0] - means['C'][0]:+.4f}")
    if means["D"][0] < 0.999:
        print(f"\n[WARN] ceiling is {means['D'][0]:.4f}, expected 1.0 — "
              "the harness itself is lossy; investigate before trusting deltas")


if __name__ == "__main__":
    main()
