#!/usr/bin/env python3
"""Repair connectivity where an ELECTRICAL constraint proves it is wrong.

Image evidence for the crossing decision is exhausted: six approaches on 4822
real causally-labelled sites top out at 0.6589 AUC and 0.70 precision, and two
oracles cap the payoff (perfect GT crossover boxes give strict success 0.3263
against 0.3526; perfect per-box decisions give 0.0 headroom). So this repairs
only where a circuit-level prior, not a pixel, says the answer must be wrong.

Two priors from the verified GT are near-absolute:

  a component with all pins on ONE net    GT rate 0.60%   detector precision
                                          0.9500 (160 found, 8 genuine)
  a net with only ONE terminal            GT rate 0.00%   precision 1.0 by
                                          construction (0 of 1509 GT nets)

Zero currently-strict images contain either, so both are lethal to strict
success and acting on them cannot break a working image. Their honest weakness
is coverage: only ~25 self-shorts and ~16 one-terminal nets fall on images in
the reachable 0.5-0.9 precision band, and 112 of the 160 self-shorts sit in the
hopeless <0.3 bucket. This is a precise, low-yield repair and is measured as
such.

Repairs, each the minimal operation the constraint implies:

  SELF-SHORT   the pins are bridged through the component's own body, since
               build_non_wire_mask blanks the bounding box but a loose box
               leaves body ink outside it. Erase an inner band of the box from
               the wire mask -- enough to break the body path, not the leads.
               Concentrated exactly where the physics predicts: Capacitor
               12.4%, I-DC 17.1%, V-DC 15.5%, MOSFET-N 15.6%, but Resistor
               only 0.7%, because a zigzag separates its own leads while
               parallel plates and a source circle leave a straight path.

  ONE-TERMINAL a net with one pin is a fragment of a split net. Bridge it to
               the nearest other node, but only along its own outgoing stroke
               direction and only within a bounded gap -- the same
               collinearity safety the global stitcher uses, applied at a
               location a constraint identified rather than everywhere.

Repairs are applied, nodes rebuilt, and the result re-checked, up to
``--passes`` times, because fixing one violation can expose another.

Usage:
    python scripts/apply_constraint_repairs.py --limit 190
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class, class_role, is_ground
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import (
    net_level_metrics,
    per_component_connected_accuracy,
    terminal_pair_metrics,
)
from schematic2netlist.nodes import build_wire_nodes, build_wire_nodes_crossover_aware
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.snapping import build_component_pin_nets
from schematic2netlist.splits import add_split_arg, load_split


def rebuild(wires, dets, cfg):
    ncfg = cfg["nodes"]
    method = ncfg.get("method") or (
        "crossover" if ncfg.get("handle_crossovers") else "cc")
    xbox = [d for d in dets if canonical_class(d["class"]) == "Wire Crossover"]
    if method == "crossover":
        node_map, _n = build_wire_nodes_crossover_aware(
            wires, xbox, connectivity=ncfg["connectivity"],
            relink=ncfg.get("relink", "band"))
    elif method == "vector":
        from schematic2netlist.vector_nodes import build_wire_nodes_vector
        node_map, _n, _i = build_wire_nodes_vector(
            wires, xbox, connectivity=ncfg["connectivity"],
            **(ncfg.get("vector", {}) or {}))
    else:
        node_map, _n = build_wire_nodes(wires, connectivity=ncfg["connectivity"])
    comps = build_component_pin_nets(dets, node_map, cfg)
    for c in comps:
        c["node_names"] = [None if x is None else f"n{x}" for x in c["nodes"]]
    return node_map, comps


def as_pred(comps, dets):
    return [{"id": c["id"], "class": c["class"],
             "nets": list(c.get("node_names", [])),
             "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                      dets[c["id"]]["width"], dets[c["id"]]["height"]]}
            for c in comps]


def score(pred, gt_comps):
    p, g, stats = align_components(pred, gt_comps)
    pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)
    tp = terminal_pair_metrics(pc, gc)
    nf = net_level_metrics(pc, gc)
    return {"tp_f1": tp["f1"], "tp_precision": tp["precision"],
            "tp_recall": tp["recall"], "net_f1": nf["f1"],
            "percomp": per_component_connected_accuracy(pc, gc),
            "strict": int(stats["unmatched_gt"] == 0 and tp["f1"] == 1.0
                          and nf["f1"] == 1.0)}


def find_violations(comps, dets):
    """(self-shorted component ids, one-terminal net names)."""
    shorts = []
    for c in comps:
        det = dets[c["id"]]
        if is_ground(det["class"]) or class_role(det["class"]) == "none":
            continue
        nets = [n for n in c.get("node_names", []) if n is not None]
        if len(nets) >= 2 and len(set(nets)) < len(nets):
            shorts.append(c["id"])
    cnt = Counter()
    for c in comps:
        for n in c.get("node_names", []):
            if n is not None:
                cnt[n] += 1
    ones = [n for n, k in cnt.items() if k == 1]
    return shorts, ones


def erase_body(wires, det, frac):
    """Erase an inner band of a component box to break a body bridge."""
    h, w = wires.shape
    bw, bh = det["width"] * frac, det["height"] * frac
    x1 = int(max(0, det["x"] - bw / 2)); x2 = int(min(w, det["x"] + bw / 2))
    y1 = int(max(0, det["y"] - bh / 2)); y2 = int(min(h, det["y"] + bh / 2))
    if x2 <= x1 or y2 <= y1:
        return False
    if not (wires[y1:y2, x1:x2] > 0).any():
        return False
    wires[y1:y2, x1:x2] = 0
    return True


def bridge_fragment(wires, node_map, nid, max_gap, dir_tol_deg):
    """Connect a one-terminal node to the nearest other node, collinearly."""
    mine = (node_map == nid)
    if not mine.any():
        return False
    pts = np.argwhere(mine)                      # (y, x)
    other = (node_map >= 0) & (~mine)
    if not other.any():
        return False
    opts = np.argwhere(other)
    s_mine = pts[:: max(1, len(pts) // 300)]
    s_oth = opts[:: max(1, len(opts) // 600)]
    d2 = ((s_mine[:, None, :] - s_oth[None, :, :]) ** 2).sum(-1)
    k = int(d2.argmin())
    ia, ib = k // len(s_oth), k % len(s_oth)
    gap = float(np.sqrt(d2[ia, ib]))
    if gap > max_gap or gap < 1:
        return False
    pa, pb = s_mine[ia], s_oth[ib]
    seg = (pb - pa).astype(float)
    seg /= (np.linalg.norm(seg) or 1.0)
    # the fragment's own direction, from its pixels near the endpoint
    near = pts[np.abs(pts - pa).sum(1) < 15]
    if len(near) >= 4:
        c = near - near.mean(0)
        ev = np.linalg.eigh(c.T @ c)[1][:, -1]
        ev = ev / (np.linalg.norm(ev) or 1.0)
        if abs(float(seg @ ev)) < np.cos(np.radians(dir_tol_deg)):
            return False                          # not a continuation
    cv2.line(wires, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), 255, 2)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=190)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--body-frac", type=float, default=0.55,
                    help="inner fraction of a self-shorted box to erase")
    ap.add_argument("--max-gap", type=int, default=80)
    ap.add_argument("--dir-tol-deg", type=float, default=45.0)
    ap.add_argument("--no-shorts", action="store_true")
    ap.add_argument("--no-ones", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/constraint_repairs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out, cfg, seed, extra={
        "body_frac": args.body_frac, "max_gap": args.max_gap,
        "passes": args.passes, "shorts": not args.no_shorts,
        "ones": not args.no_ones})

    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)
    rows, acts = [], Counter()

    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        gcomps = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            f"{cfg['detect']['cache_dir']}/{stem}.json",
            min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(images_dir / nm, cfg, detections=dets)
        base = score(as_pred(res["components"], res["detections"]), gcomps)

        wires = res["clean_wires"].copy()
        node_map, comps = res["node_map"], res["components"]
        n_act = 0
        for _p in range(args.passes):
            shorts, ones = find_violations(comps, res["detections"])
            changed = False
            if not args.no_shorts:
                for cid in shorts:
                    if erase_body(wires, res["detections"][cid],
                                  args.body_frac):
                        acts["erased_self_short_body"] += 1
                        n_act += 1
                        changed = True
            if not args.no_ones:
                name_to_id = {}
                for c in comps:
                    for n, nn in zip(c.get("nodes", []),
                                     c.get("node_names", [])):
                        if n is not None and nn is not None:
                            name_to_id[nn] = int(n)
                for nn in ones:
                    nid = name_to_id.get(nn)
                    if nid is None:
                        continue
                    if bridge_fragment(wires, node_map, nid, args.max_gap,
                                       args.dir_tol_deg):
                        acts["bridged_one_terminal_net"] += 1
                        n_act += 1
                        changed = True
            if not changed:
                break
            node_map, comps = rebuild(wires, res["detections"], cfg)

        after = (score(as_pred(comps, res["detections"]), gcomps)
                 if n_act else dict(base))
        rows.append({"image": nm, "n_actions": n_act,
                     **{f"base_{k}": v for k, v in base.items()},
                     **{f"fix_{k}": v for k, v in after.items()}})
        if i % 20 == 0:
            print(f"  [{i}/{len(names)}] strict "
                  f"{sum(r['base_strict'] for r in rows)} -> "
                  f"{sum(r['fix_strict'] for r in rows)}  "
                  f"actions={sum(r['n_actions'] for r in rows)}", flush=True)

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / max(n, 1)
    print(f"\n=== CONSTRAINT-TRIGGERED CONNECTIVITY REPAIR ({n} images) ===\n")
    print(f"  {'metric':22s} {'baseline':>9s} {'repaired':>9s} {'delta':>9s}")
    for k in ("tp_f1", "tp_precision", "tp_recall", "net_f1", "percomp",
              "strict"):
        a, b = mean(f"base_{k}"), mean(f"fix_{k}")
        print(f"  {k:22s} {a:9.4f} {b:9.4f} {b-a:+9.4f}")
    print(f"\n  actions taken: {dict(acts)}")
    print(f"  images touched: {sum(1 for r in rows if r['n_actions'])}")
    print(f"  strict gained: {sum(1 for r in rows if r['fix_strict'] > r['base_strict'])}"
          f", lost: {sum(1 for r in rows if r['fix_strict'] < r['base_strict'])}")
    print(f"  tp_f1 improved: {sum(1 for r in rows if r['fix_tp_f1'] > r['base_tp_f1'])}"
          f", worsened: {sum(1 for r in rows if r['fix_tp_f1'] < r['base_tp_f1'])}")

    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": n,
        "baseline": {k: round(mean(f"base_{k}"), 4) for k in
                     ("tp_f1", "tp_precision", "tp_recall", "net_f1",
                      "percomp", "strict")},
        "repaired": {k: round(mean(f"fix_{k}"), 4) for k in
                     ("tp_f1", "tp_precision", "tp_recall", "net_f1",
                      "percomp", "strict")},
        "actions": dict(acts),
        "images_touched": sum(1 for r in rows if r["n_actions"]),
        "strict_gained": sum(1 for r in rows
                             if r["fix_strict"] > r["base_strict"]),
        "strict_lost": sum(1 for r in rows
                           if r["fix_strict"] < r["base_strict"]),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
