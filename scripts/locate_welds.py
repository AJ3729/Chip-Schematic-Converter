#!/usr/bin/env python3
"""Is any wire intersection actually the CUT POINT of a weld?

Four independent attempts to fix connectivity by splitting at wire
intersections have all LOST accuracy:

  learned patch classifier + notch      -0.110 terminal-pair F1
  PERFECT GT crossover boxes + notch    -0.026 strict success
  vector geometric crossing pairing     -0.025 vs the same tracer with
                                        splitting disabled
  every threshold / box-size sweep      monotonically better with FEWER
                                        splits

Meanwhile the pipeline IS over-merged (terminal-pair precision minus
recall = -0.030), so welds exist. Both facts can only hold together if
splitting at an intersection cannot generally undo a weld.

This script tests that causally rather than by coincidence. Asking
"does a degree-4 site lie inside this welded node?" is uninformative: a
welded node spans much of the drawing and therefore contains
intersections by chance (that weaker test reported 61%, which means
almost nothing). Instead, for every welded node this script:

1. anchors each fused GT net to a pixel of the node (the node pixel
   nearest each owning component's box centre);
2. removes a small disk at ONE candidate intersection;
3. re-labels the node and asks whether the anchors of different GT nets
   now land in different components.

A site that achieves that separation IS the weld's cut point, and
splitting there would fix the weld. If most welds have no such site,
intersection splitting cannot repair them no matter how well a
classifier ranks crossings — which is the hypothesis the four failures
above jointly imply.

Reported per weld: whether ANY single intersection cuts it
(``single_cut``), whether only a multi-site cut could
(``needs_multi_or_none``), and the degree of the cutting site when one
exists.

Usage:
    python scripts/locate_welds.py --limit 25
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import load_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.skeleton import intersection_sites_with_degree
from schematic2netlist.splits import add_split_arg, load_split


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def align(pred, dets, gt_comps):
    """Hungarian IoU alignment. A component's id is its DETECTION index,
    never a GT index; assuming otherwise mismatches every net."""
    if not pred or not gt_comps:
        return []
    cost = np.ones((len(pred), len(gt_comps)))
    for pi, c in enumerate(pred):
        d = dets[c["id"]]
        pb = (d["x"], d["y"], d["width"], d["height"])
        for gj, g in enumerate(gt_comps):
            if canonical_class(c["class"]) == canonical_class(g["class"]):
                cost[pi, gj] = 1.0 - iou(pb, g["bbox"])
    ri, ci = linear_sum_assignment(cost)
    return [(pi, gj) for pi, gj in zip(ri, ci) if cost[pi, gj] <= 0.7]


def nearest_pixel(mask: np.ndarray, x: float, y: float):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    d = (xs - x) ** 2 + (ys - y) ** 2
    k = int(np.argmin(d))
    return int(xs[k]), int(ys[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_split_arg(ap, "val")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--cut-radius", type=int, default=7,
                    help="disk radius removed when testing a site (px)")
    ap.add_argument("--out-dir", default="results/welds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)

    tally = Counter()
    rows = []
    for idx, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        dets = load_cached_detections(
            f"{cfg['detect']['cache_dir']}/{stem}.json",
            min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(images_dir / nm, cfg, detections=dets)
        node_map = res["node_map"]
        pred = res["components"]
        gt_comps = gt["components"]

        # node id -> [(gt_net, component box centre), ...]
        node_nets: dict[int, list] = {}
        for pi, gj in align(pred, dets, gt_comps):
            gnets = [t["net"] for t in gt_comps[gj]["terminals"]]
            gb = gt_comps[gj]["bbox"]
            for i, n in enumerate(pred[pi].get("nodes", [])):
                if n is None or i >= len(gnets) or gnets[i] is None:
                    continue
                node_nets.setdefault(int(n), []).append(
                    (gnets[i], (gb[0], gb[1])))

        welded = {n: v for n, v in node_nets.items()
                  if len({t[0] for t in v}) >= 2}

        sites = intersection_sites_with_degree(
            (node_map >= 0).astype(np.uint8))

        for node, members in welded.items():
            mask = (node_map == node).astype(np.uint8)
            # anchor every fused net to a pixel of this node
            anchors: dict[str, tuple] = {}
            for net, (bx, by) in members:
                if net in anchors:
                    continue
                p = nearest_pixel(mask > 0, bx, by)
                if p:
                    anchors[net] = p
            if len(anchors) < 2:
                continue

            # candidate sites are those lying on this node
            cand = [(x, y, d) for (x, y, d) in sites
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                    and mask[y, x]]

            cut_deg = None
            for (sx, sy, deg) in cand:
                probe = mask.copy()
                cv2.circle(probe, (sx, sy), args.cut_radius, 0, -1)
                n_lab, lab = cv2.connectedComponents(probe, connectivity=8)
                if n_lab <= 2:                     # nothing separated
                    continue
                got = {}
                for net, (ax, ay) in anchors.items():
                    got[net] = int(lab[ay, ax])
                if 0 in got.values():              # an anchor sat in the cut
                    continue
                if len(set(got.values())) >= 2:    # nets now separated
                    cut_deg = deg
                    break

            if cut_deg is None:
                tally["needs_multi_or_none"] += 1
            else:
                tally["single_cut"] += 1
                tally[f"single_cut_deg{min(cut_deg, 5)}"] += 1
            tally["_welds"] += 1
            rows.append({"image": nm, "node": node,
                         "n_fused_nets": len(anchors),
                         "candidate_sites": len(cand),
                         "single_cut": int(cut_deg is not None),
                         "cut_site_degree": cut_deg if cut_deg else ""})
        if idx % 5 == 0:
            print(f"[{idx}/{len(names)}] welds={tally['_welds']}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "per_weld.csv").open("w", newline="") as fh:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    tot = max(tally["_welds"], 1)
    summary = {
        "images": len(names),
        "welded_nodes": tally["_welds"],
        "single_intersection_cut_exists": tally["single_cut"],
        "single_cut_share": round(tally["single_cut"] / tot, 4),
        "no_single_cut_share": round(tally["needs_multi_or_none"] / tot, 4),
        "cut_site_degrees": {k: v for k, v in sorted(tally.items())
                             if k.startswith("single_cut_deg")},
        "interpretation": (
            "single_cut_share is the ceiling for ANY intersection-splitting "
            "method: welds without a single cutting site cannot be repaired "
            "by one split, however well crossings are classified."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
