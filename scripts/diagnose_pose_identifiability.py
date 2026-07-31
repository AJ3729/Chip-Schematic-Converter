#!/usr/bin/env python3
"""Can wire evidence tell a MOSFET's drain from its source, even in principle?

With duplicate-node collapse fixed, snapping's set-level error in oracle
mode C is 0.1% but 24.2% of components still hold the right nets in the
WRONG ORDER (``scripts/diagnose_snapping.py``). Permutations are invisible
to every benchmark number — ``canonicalize_terminals`` sorts them away —
but they are exactly what the port-template contribution claims to fix,
and a swapped drain/source or anode/cathode emits SPICE that does not
describe the drawn circuit.

Before trying to fix pose selection, establish whether it is fixable.
``ports.match_ports`` scores a pose by how well its predicted pin
positions match the observed boundary crossings. That is a purely
GEOMETRIC criterion, so two poses that place pins at the same LOCATIONS
and differ only in which pin is called what are indistinguishable to it —
no scoring change, no threshold, no better matcher can separate them.
Inspecting the templates suggests this is exactly the situation: MOSFET-N
pose0 puts (Drain, Gate, Source) at left/bottom/right and pose4 puts them
at right/bottom/left. Same three sites, opposite labels.

This script measures it rather than assuming it. For every component in
mode C, where connectivity is perfect:

  1. score every pose the way ``match_ports`` does;
  2. find which poses would yield the CORRECT terminal order and which
     would not;
  3. report the score GAP between the best correct pose and the best
     incorrect one.

A gap near zero means the two are tied on geometry and the choice is a
coin flip that better geometry cannot win — the signal has to come from
symbol appearance. A clearly positive gap for the correct pose would mean
the selector is leaving usable evidence on the table, which is a fixable
bug.

Reported per class, since the answer differs by symbol: a Diode's two
poses are geometrically distinct in a way a MOSFET's are not.

Usage:
    python scripts/diagnose_pose_identifiability.py --limit 190
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.classes import canonical_class, class_terminals, is_ground
from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import load_gt
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.oracle_render import render_gt_node_map
from schematic2netlist.ports import load_templates, predicted_sites
from schematic2netlist.snapping import _boundary_run_sites

from oracle import gt_detections  # noqa: E402


def score_pose(cls, det, pose, templates, node_pts, n_ports, diag):
    """Replicate match_ports' assignment for ONE pose.

    Returns (mean_true_distance, nodes_in_port_order) or None.
    """
    sites = predicted_sites(cls, det, pose, templates)
    if not sites or len(sites) != n_ports:
        return None
    nodes_uniq = sorted(node_pts)
    repeats = max(1, -(-n_ports // len(nodes_uniq)))
    slots = [(nid, k) for k in range(repeats) for nid in nodes_uniq]
    true_d = np.zeros((n_ports, len(slots)))
    cost = np.zeros((n_ports, len(slots)))
    for i, (sx, sy) in enumerate(sites):
        for j, (nid, k) in enumerate(slots):
            d = min(float(np.hypot(sx - rx, sy - ry))
                    for rx, ry in node_pts[nid])
            true_d[i, j] = d
            cost[i, j] = d + k * diag
    rows, cols = linear_sum_assignment(cost)
    mean_d = float(true_d[rows, cols].sum()) / max(len(rows), 1)
    nodes = [None] * n_ports
    for i, j in zip(rows, cols):
        nodes[i] = slots[j][0]
    return mean_d, nodes


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=190)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out-dir", default="results/pose_identifiability")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    gt_dir = args.gt_dir or cfg["benchmark"]["gt_dir"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    templates = load_templates()

    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)
    s = cfg["snapping"]

    rows = []
    by_class = defaultdict(lambda: {"gaps": [], "n": 0, "chosen_ok": 0,
                                    "correct_exists": 0, "tied": 0})
    overall = Counter()

    for i, nm in enumerate(names, 1):
        stem = nm.rsplit(".", 1)[0]
        gt = load_gt(f"{gt_dir}/{stem}.json")
        img = cv2.imread(f"{images_dir}/{nm}")
        if img is None:
            continue
        node_map, label_of_net, report = render_gt_node_map(gt, img.shape)
        if not report["ok"]:
            continue
        net_of = {v: k for k, v in label_of_net.items()}
        gdets = gt_detections(gt)

        for ci, gcomp in enumerate(gt["components"]):
            det = gdets[ci]
            cls = canonical_class(det["class"])
            if is_ground(det["class"]):
                continue
            tpl = templates.get(cls)
            if not tpl or not tpl.get("poses"):
                continue
            n_ports = tpl["n_ports"]
            gt_nets = [t["net"] for t in gcomp["terminals"]]
            if len(gt_nets) != n_ports or any(n is None for n in gt_nets):
                continue
            # only a component whose pins carry >=2 DISTINCT nets can have a
            # meaningful order; a fully shorted one is order-free
            if len(set(gt_nets)) < 2:
                continue

            x1, y1, x2, y2 = bbox_xyxy(det)
            found = []
            for r in range(s["expand_step"], s["max_expand"] + 1,
                           s["expand_step"]):
                f = _boundary_run_sites(node_map, x1 - r, y1 - r,
                                        x2 + r, y2 + r)
                if len(f) > len(found):
                    found = f
                if len(f) >= n_ports:
                    break
            if not found:
                continue
            node_pts = defaultdict(list)
            for nid, rx, ry in found:
                node_pts[int(nid)].append((rx, ry))
            diag = float(np.hypot(det["width"], det["height"])) or 1.0

            scored = []
            for pose in tpl["poses"]:
                r_ = score_pose(cls, det, pose, templates, node_pts,
                                n_ports, diag)
                if r_ is None:
                    continue
                mean_d, nodes = r_
                nets = [net_of.get(n) if n is not None else None
                        for n in nodes]
                scored.append((mean_d, pose, nets == gt_nets))
            if not scored:
                continue
            scored.sort(key=lambda t: t[0])
            chosen_d, chosen_pose, chosen_ok = scored[0]
            correct = [t for t in scored if t[2]]
            wrong = [t for t in scored if not t[2]]
            rec = by_class[cls]
            rec["n"] += 1
            rec["chosen_ok"] += int(chosen_ok)
            overall["n"] += 1
            overall["chosen_ok"] += int(chosen_ok)
            if not correct:
                # no pose in the template can produce the GT order at all
                overall["no_correct_pose"] += 1
                rows.append({"image": nm, "comp_id": ci, "class": cls,
                             "chosen_pose": chosen_pose,
                             "chosen_ok": int(chosen_ok),
                             "gap_frac": "", "verdict": "no_correct_pose"})
                continue
            rec["correct_exists"] += 1
            if not wrong:
                # every pose yields the GT order — a symmetric class with no
                # port names, where order is free. There is no gap to report
                # and averaging one in as -inf would poison the statistic.
                rec["no_wrong_pose"] = rec.get("no_wrong_pose", 0) + 1
                overall["no_wrong_pose"] += 1
                rows.append({"image": nm, "comp_id": ci, "class": cls,
                             "chosen_pose": chosen_pose, "chosen_ok": 1,
                             "gap_frac": "", "verdict": "no_wrong_pose"})
                continue
            best_c, best_w = correct[0][0], wrong[0][0]
            # positive gap => the correct pose fits WORSE than some wrong
            # one, so geometry actively points the wrong way
            gap = (best_c - best_w) / diag
            rec["gaps"].append(gap)
            tied = abs(gap) < 0.02
            rec["tied"] += int(tied)
            overall["tied"] += int(tied)
            rows.append({"image": nm, "comp_id": ci, "class": cls,
                         "chosen_pose": chosen_pose,
                         "chosen_ok": int(chosen_ok),
                         "gap_frac": round(gap, 4),
                         "best_correct_frac": round(best_c / diag, 4),
                         "best_wrong_frac": round(best_w / diag, 4)
                         if wrong else "",
                         "verdict": ("tied" if tied else
                                     "correct_better" if gap < 0 else
                                     "wrong_better")})
        if i % 20 == 0:
            print(f"[{i}/{len(names)}] components={overall['n']}", flush=True)

    print(f"\n=== POSE IDENTIFIABILITY FROM WIRE GEOMETRY (oracle mode C) ===")
    print(f"components with >=2 distinct GT nets and a usable template: "
          f"{overall['n']}")
    print(f"chosen pose gave the correct terminal ORDER: "
          f"{overall['chosen_ok']}/{overall['n']} = "
          f"{overall['chosen_ok']/max(overall['n'],1):.4f}")
    print(f"no pose in the template could produce the GT order: "
          f"{overall['no_correct_pose']}")
    print(f"\nGap = (best correct pose's fit - best incorrect pose's fit) / "
          f"box diagonal.\nNegative means geometry favours the correct pose; "
          f"~0 means the two are\nindistinguishable; positive means geometry "
          f"actively points the wrong way.\n")
    print(f"  {'class':22s} {'n':>5s} {'order ok':>9s} {'no corr':>8s} "
          f"{'free':>5s} {'contested':>10s} {'tied':>7s} {'median gap':>11s}")
    print(f"  {'':22s} {'':>5s} {'':>9s} {'pose':>8s} "
          f"{'order':>5s} {'n':>10s} {'':>7s} {'':>11s}")
    ranked = sorted(by_class.items(), key=lambda kv: -kv[1]["n"])
    for cls, rec in ranked:
        if rec["n"] < 15:
            continue
        g = rec["gaps"]
        nofree = rec.get("no_wrong_pose", 0)
        nocorr = rec["n"] - rec["correct_exists"]
        med = f"{st.median(g):11.4f}" if g else f"{'--':>11s}"
        print(f"  {cls:22s} {rec['n']:5d} "
              f"{rec['chosen_ok']/rec['n']:9.1%} "
              f"{nocorr:8d} {nofree:5d} {len(g):10d} "
              f"{rec['tied']/max(len(g),1):7.1%} {med}")
    print(f"\n  'free order'  = every pose gives the GT order (symmetric "
          f"class, no port names)\n"
          f"  'no corr pose'= NO pose in the template can produce the GT "
          f"order at all\n"
          f"  'contested n' = the population the gap is computed over")

    allg = [g for rec in by_class.values() for g in rec["gaps"]]
    if allg:
        print(f"\noverall: {sum(1 for g in allg if abs(g) < 0.02)/len(allg):.1%}"
              f" of components have the correct and incorrect poses tied to "
              f"within\n0.02 box diagonals — for those, no geometric scoring "
              f"rule can pick the right\none, and the evidence must come from "
              f"symbol appearance instead.")

    with (out / "per_component.csv").open("w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "n_components": overall["n"],
        "chosen_order_correct": round(
            overall["chosen_ok"] / max(overall["n"], 1), 4),
        "no_correct_pose_exists": overall["no_correct_pose"],
        "tied_within_0.02_diag": round(
            sum(1 for g in allg if abs(g) < 0.02) / max(len(allg), 1), 4),
        "by_class": {
            cls: {"n": r["n"],
                  "order_correct": round(r["chosen_ok"] / max(r["n"], 1), 4),
                  "tied_rate": round(r["tied"] / max(len(r["gaps"]), 1), 4),
                  "median_gap_frac": round(st.median(r["gaps"]), 4)
                  if r["gaps"] else None}
            for cls, r in by_class.items() if r["n"] >= 15},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}/per_component.csv + summary.json")


if __name__ == "__main__":
    main()
