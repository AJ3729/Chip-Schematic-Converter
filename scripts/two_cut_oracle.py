#!/usr/bin/env python3
"""Can a weld be separated by TWO cuts when no single cut works?

Every crossing mechanism this repo has built or bounded is a single-site
decision: notch at a crossover box, classify one intersection, split one site.
All of them failed, and `locate_welds.py` explains part of it -- 14% of welds
have no single cutting site at all. The standing conclusion is that the
residual connectivity error is unrecoverable from the wire graph.

That conclusion assumes one cut. Adjudicating the welds against the ORIGINAL
photographs shows the drafters DO draw hops -- a semicircular detour where a
wire crosses another without connecting -- and that a hand-drawn hop's arc
meets the crossed conductor at TWO places, once entering and once leaving. If
that is what welds are made of, then:

  * no single cut can separate them (remove one touch point, the arc still
    holds through the other) -- which is exactly what was measured; and
  * a cut at BOTH touch points would, and nobody has tried it.

This script tests that directly. For every welded node it runs the same causal
probe as locate_welds.py, then, when no single site separates the fused nets,
searches PAIRS of sites. Reported per weld: separable by one cut, by two, or
by neither, plus the pixel distance between the two sites of a successful
pair -- because a hop's two touch points sit roughly one stroke-crossing
apart, and a pair that only works at opposite ends of the drawing is a
different phenomenon with no local fix.

This is a measurement, not a method: it bounds what a two-site splitter could
buy before anyone builds one.

Usage:
    python scripts/two_cut_oracle.py --limit 60 --out-dir results/two_cut
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from locate_welds import align, nearest_pixel

from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import load_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.skeleton import intersection_sites_with_degree


def separates(mask8, sites, anchors, radius):
    """Does removing a disk at every listed site split the anchors apart?"""
    probe = mask8.copy()
    for (sx, sy) in sites:
        cv2.circle(probe, (sx, sy), radius, 0, -1)
    n_lab, lab = cv2.connectedComponents(probe, connectivity=8)
    if n_lab <= 2:
        return False
    got = {}
    for net, (ax, ay) in anchors.items():
        v = int(lab[ay, ax])
        if v == 0:                       # an anchor fell inside a cut
            return False
        got[net] = v
    return len(set(got.values())) >= 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--cut-radius", type=int, default=7)
    ap.add_argument("--max-sites", type=int, default=44,
                    help="cap candidate sites per node; pairs go as the square")
    ap.add_argument("--out-dir", default="results/two_cut")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
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
        node_map, pred, gt_comps = res["node_map"], res["components"], gt["components"]

        node_nets: dict[int, list] = {}
        for pi, gj in align(pred, dets, gt_comps):
            gnets = [t["net"] for t in gt_comps[gj]["terminals"]]
            gb = gt_comps[gj]["bbox"]
            for i, n in enumerate(pred[pi].get("nodes", [])):
                if n is None or i >= len(gnets) or gnets[i] is None:
                    continue
                node_nets.setdefault(int(n), []).append((gnets[i], (gb[0], gb[1])))
        welded = {n: v for n, v in node_nets.items()
                  if len({t[0] for t in v}) >= 2}

        sites_all = intersection_sites_with_degree((node_map >= 0).astype(np.uint8))

        for node, members in welded.items():
            mask = (node_map == node)
            anchors: dict[str, tuple] = {}
            for net, (bx, by) in members:
                if net in anchors:
                    continue
                p = nearest_pixel(mask, bx, by)
                if p:
                    anchors[net] = p
            if len(anchors) < 2:
                continue
            mask8 = mask.astype(np.uint8)
            cand = [(x, y) for (x, y, _d) in sites_all
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                    and mask[y, x]]

            one = next((s for s in cand
                        if separates(mask8, [s], anchors, args.cut_radius)), None)
            best_pair, pair_dist = None, None
            if one is None and len(cand) >= 2:
                use = cand[: args.max_sites]
                for i in range(len(use)):
                    for j in range(i + 1, len(use)):
                        if separates(mask8, [use[i], use[j]], anchors,
                                     args.cut_radius):
                            best_pair = (use[i], use[j])
                            pair_dist = float(np.hypot(use[i][0] - use[j][0],
                                                       use[i][1] - use[j][1]))
                            break
                    if best_pair:
                        break

            verdict = ("one_cut" if one else
                       "two_cut" if best_pair else "neither")
            tally[verdict] += 1
            tally["_welds"] += 1
            rows.append({"image": nm, "node": node,
                         "n_fused_nets": len(anchors),
                         "candidate_sites": len(cand),
                         "verdict": verdict,
                         "pair_distance_px": round(pair_dist, 1) if pair_dist else ""})
        if idx % 5 == 0:
            print(f"[{idx}/{len(names)}] welds={tally['_welds']} "
                  f"one={tally['one_cut']} two={tally['two_cut']} "
                  f"neither={tally['neither']}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_weld.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = tally["_welds"] or 1
    dists = [r["pair_distance_px"] for r in rows if r["pair_distance_px"] != ""]
    summary = {
        "images": len(names),
        "welds": tally["_welds"],
        "one_cut": tally["one_cut"],
        "two_cut": tally["two_cut"],
        "neither": tally["neither"],
        "one_cut_share": round(tally["one_cut"] / n, 4),
        "two_cut_share": round(tally["two_cut"] / n, 4),
        "cumulative_share_after_two_cuts":
            round((tally["one_cut"] + tally["two_cut"]) / n, 4),
        "median_pair_distance_px":
            round(float(np.median(dists)), 1) if dists else None,
        "interpretation":
            "two_cut counts welds that NO single site separates but some PAIR "
            "does. A large share with a small pair distance is the signature "
            "of a hand-drawn hop whose arc touches the crossed conductor "
            "twice, and would mean single-site splitting was the wrong "
            "operation rather than crossings being unrecoverable.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
