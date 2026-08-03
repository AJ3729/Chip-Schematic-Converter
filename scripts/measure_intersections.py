#!/usr/bin/env python3
"""Phase 0: is a junction/crossover classifier worth building? (C2)

A learned junction-vs-crossover model can only fix one class of error —
a wrong decision where two strokes MEET. It cannot repair a wire broken
by a pen lift mid-span. Before spending days on it, this measures
whether such decisions are actually load-bearing on our data.

Three questions, in order of how much they matter:

1. **Coverage.** How many stroke intersections does an image contain,
   versus how many the detector labels `Wire Crossover`? Every
   unlabeled intersection is currently an unexamined decision: connected
   components treats touching strokes as connected, full stop.

2. **Load-bearing.** Of those unlabeled intersections, how many
   actually matter? Notch the wire mask at each and see whether its
   connected component splits. An intersection on a redundant loop can
   be cut without changing connectivity, and no classifier decision
   there would change the netlist.

3. **Correctness.** For the load-bearing ones, does ground truth say
   the two sides belong to DIFFERENT nets? Those are intersections
   where the pipeline is silently welding two nets together, and are
   exactly what a classifier would win back. Sides are attributed to
   GT nets through the components that snap to them.

The headline is question 3: welds per image. If that is near zero the
classifier is not the bottleneck and the effort belongs elsewhere.

Usage:
    python scripts/measure_intersections.py --limit 60
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from schematic2netlist.benchmark import align_components
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import load_gt
from schematic2netlist.nodes import bbox_xyxy, build_wire_nodes
from schematic2netlist.snapping import build_component_pin_nets
from schematic2netlist.textmask import detect_text_mask
from schematic2netlist.wires import (
    build_non_wire_mask,
    extract_wires,
    stitch_wire_islands,
    stitchable_mask,
)
from schematic2netlist.splits import add_split_arg, load_split

# 3x3 neighbour count; a thinned pixel with >=3 neighbours is a branch
_K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def thin(mask: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Zhang-Suen thinning to a 1-pixel-wide skeleton.

    OpenCV's ximgproc and skimage are both absent here. The morphological
    erode/open residue skeleton is NOT an adequate substitute — it leaves
    a thick, fragmented result on which branch-point counting badly
    under-reports intersections. This is the real algorithm, vectorized.
    """
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, dy, 0), dx, 1) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1))]
            P2, P3, P4, P5, P6, P7, P8, P9 = p
            B = sum(p)
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                    for i in range(8))
            if step == 0:
                c1, c2 = P2 * P4 * P6, P4 * P6 * P8
            else:
                c1, c2 = P2 * P4 * P8, P2 * P6 * P8
            kill = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                    & (c1 == 0) & (c2 == 0))
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            break
    return img


def intersection_sites(mask: np.ndarray, min_sep: int = 6) -> list[tuple[int, int]]:
    """Branch points of the thinned mask, clustered into distinct sites."""
    skel = thin(mask)
    # scipy, not cv2.filter2D: with a uint8 kernel filter2D returns wrong
    # neighbour counts (capped at 2 here), which silently reports zero
    # branch points on a skeleton that plainly has them.
    neigh = ndimage.convolve(skel.astype(np.int32), _K.astype(np.int32),
                             mode="constant")
    branch = ((skel > 0) & (neigh >= 3)).astype(np.uint8)
    if not branch.any():
        return []
    # merge adjacent branch pixels into one site
    branch = cv2.dilate(branch, np.ones((min_sep, min_sep), np.uint8))
    n, _lab, _stats, cents = cv2.connectedComponentsWithStats(branch, 8)
    return [(int(round(x)), int(round(y))) for x, y in cents[1:n]]


def gt_net_lookup(gt: dict, comps: list[dict], dets: list[dict]) -> dict[int, set[str]]:
    """Map each predicted wire-node id to the GT nets of the component
    terminals that snapped to it. A node carrying two different GT nets
    is a node the pipeline has welded together.

    Predictions are matched to GT by the benchmark's Hungarian IoU
    alignment. Assuming detection order equals GT order is wrong — they
    are independent orderings — and doing so reports welds that are
    really just mismatched components.
    """
    pred = [{"id": c["id"], "class": c["class"],
             "nets": [None] * len(c.get("nodes", [])),
             "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                      dets[c["id"]]["width"], dets[c["id"]]["height"]]}
            for c in comps if c["id"] < len(dets)]
    gt_comps = [{"id": g["id"], "class": g["class"],
                 "nets": [t["net"] for t in g["terminals"]],
                 "bbox": g["bbox"]} for g in gt["components"]]
    aligned, _g, _stats = align_components(pred, gt_comps)

    gt_by_id = {g["id"]: g for g in gt_comps}
    node_by_pred_id = {c["id"]: c.get("nodes", []) for c in comps}
    orig_by_new = {a["id"]: p["id"] for a, p in zip(aligned, pred)}

    out: dict[int, set[str]] = {}
    for new_id, orig_id in orig_by_new.items():
        gc = gt_by_id.get(new_id)
        if gc is None:                      # unmatched prediction
            continue
        nets = gc["nets"]
        for k, node in enumerate(node_by_pred_id.get(orig_id, [])):
            if node is None or k >= len(nets) or nets[k] is None:
                continue
            out.setdefault(int(node), set()).add(nets[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out-dir", default="results/intersections")
    ap.add_argument("--config", default=None)
    ap.add_argument("--notch", type=int, default=9, help="notch diameter px")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_dir = Path(args.gt_dir or cfg["benchmark"]["gt_dir"])
    names = load_split(args.split, args.splits_dir)[: args.limit]
    det_dir = Path(cfg["detect"]["cache_dir"])

    rows, tally = [], Counter()
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gp, dp = gt_dir / f"{stem}.json", det_dir / f"{stem}.json"
        if not gp.exists() or not dp.exists():
            continue
        gt = load_gt(gp)
        if not gt.get("verified"):
            continue

        img = cv2.imread(str(Path(args.images_dir) / nm))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = load_cached_detections(dp)
        tm = detect_text_mask(gray, cfg) if cfg["textmask"]["enabled"] else None
        nwm = build_non_wire_mask(gray, dets, cfg, tm)
        _c, wires = extract_wires(gray, nwm, cfg)
        if cfg["wires"].get("stitch_masked_gaps"):
            wires = stitch_wire_islands(
                wires, stitchable_mask(gray.shape, dets, cfg, tm), cfg)

        node_map, _n = build_wire_nodes(wires, connectivity=8)
        comps = build_component_pin_nets(dets, node_map, cfg)
        node_nets = gt_net_lookup(gt, comps, dets)

        cross_boxes = [bbox_xyxy(d) for d in dets
                       if canonical_class(d["class"]) == "Wire Crossover"]
        sites = intersection_sites(wires)

        n_labeled = n_unlabeled = n_loadbearing = n_weld = 0
        for (x, y) in sites:
            labeled = any(x1 <= x <= x2 and y1 <= y <= y2
                          for x1, y1, x2, y2 in cross_boxes)
            if labeled:
                n_labeled += 1
                continue
            n_unlabeled += 1

            node_id = int(node_map[min(max(y, 0), node_map.shape[0] - 1),
                                   min(max(x, 0), node_map.shape[1] - 1)])
            if node_id < 0:
                continue
            # counterfactual: cut here and see whether this node splits
            cut = wires.copy()
            cv2.circle(cut, (x, y), args.notch // 2, 0, -1)
            sub_lab, _k = ndimage.label((cut > 0) & (node_map == node_id))
            pieces = [p for p in np.unique(sub_lab) if p != 0]
            if len(pieces) < 2:
                continue
            n_loadbearing += 1

            # would GT put the pieces on different nets? attribute each
            # piece through the terminals that snapped to this node
            nets_here = node_nets.get(node_id, set())
            if len(nets_here) >= 2:
                n_weld += 1

        rows.append({"image": nm, "sites": len(sites), "labeled": n_labeled,
                     "unlabeled": n_unlabeled, "load_bearing": n_loadbearing,
                     "welds": n_weld,
                     "nodes_with_multiple_gt_nets":
                         sum(1 for v in node_nets.values() if len(v) >= 2),
                     "nodes_total": len(node_nets)})
        for k in ("sites", "labeled", "unlabeled", "load_bearing", "welds"):
            tally[k] += rows[-1][k]
        print(f"[{i}/{len(names)}] {nm}: {len(sites)} sites, "
              f"{n_labeled} labeled, {n_loadbearing} load-bearing, "
              f"{n_weld} welds", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    welded_nodes = sum(r["nodes_with_multiple_gt_nets"] for r in rows)
    total_nodes = sum(r["nodes_total"] for r in rows)
    summary = {
        "n_images": n,
        "mean_sites_per_image": round(tally["sites"] / n, 2),
        "mean_labeled_crossovers": round(tally["labeled"] / n, 2),
        "mean_unlabeled_sites": round(tally["unlabeled"] / n, 2),
        "mean_load_bearing": round(tally["load_bearing"] / n, 2),
        "mean_welds_per_image": round(tally["welds"] / n, 2),
        "labeled_fraction": round(tally["labeled"] / max(tally["sites"], 1), 4),
        "welded_nodes": welded_nodes,
        "total_snapped_nodes": total_nodes,
        "welded_node_rate": round(welded_nodes / max(total_nodes, 1), 4),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nPHASE 0 — intersection audit over {n} images\n")
    print(f"  stroke intersections per image     {summary['mean_sites_per_image']:8.2f}")
    print(f"  ...labeled Wire Crossover          {summary['mean_labeled_crossovers']:8.2f}"
          f"   ({summary['labeled_fraction']:.1%} of sites)")
    print(f"  ...unlabeled (unexamined)          {summary['mean_unlabeled_sites']:8.2f}")
    print(f"  ...of those, load-bearing          {summary['mean_load_bearing']:8.2f}")
    print(f"  ...of those, welding 2 GT nets     {summary['mean_welds_per_image']:8.2f}")
    print(f"\n  wire nodes carrying >1 GT net: {welded_nodes}/{total_nodes} "
          f"= {summary['welded_node_rate']:.1%}")
    print(f"\nwrote {out}/summary.json + per_image.csv")


if __name__ == "__main__":
    main()
