#!/usr/bin/env python3
"""ORACLE: if every welded pair of nets were separated by cutting the one
correct wire segment, what would strict success be?

This bounds an entire programme of work before any of it is built, the way
the crossover oracle did — and that one saved a GPU night by showing perfect
crossover boxes make strict success WORSE (0.3526 -> 0.3263).

The measurements that motivate it:

  - 33.7% of predicted nodes carry exactly TWO ground-truth nets. Pairwise
    fusion, not mega-nodes, is the dominant connectivity failure.
  - 93.2% of those pairwise-welded nodes are ONE connected ink blob, so they
    are genuine ink fusions rather than artefacts of the notch's logical
    relinking (which does dominate the 7.9% of nodes carrying 3+ nets).
  - On the skeleton graph of such a node, max-flow between the two nets'
    terminals is 1 for 75% of cases: a SINGLE arm bridges them.

A single bridging arm is a different object from a crossing, and that
difference explains why crossing work has failed. At a clean 4-way crossing
the two nets meet at a site and separating them needs two arm cuts, so
min-cut 1 means these welds are not crossings at all — they are one spurious
segment: a stray mark, unmasked text, two wires drawn touching, or ink
through a component body.

The oracle uses ground truth ONLY to choose which arm to cut. It then
removes that arm from the wire mask and re-runs the real node assembly,
snapping and scoring. So it answers "is finding the bridge worth doing",
without assuming any particular way of finding it.

Reported against the 0.3526 baseline. A large gain makes bridge detection
the priority; a small one closes it, as the crossover oracle closed
crossings.

Usage:
    python scripts/oracle_bridge_cut.py --limit 190
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from measure_weld_mincut import build_arm_graph, nearest_arm

from schematic2netlist.benchmark import align_components, canonicalize_terminals
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
from schematic2netlist.nodes import (
    build_wire_nodes,
    build_wire_nodes_crossover_aware,
)
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.snapping import build_component_pin_nets
from schematic2netlist.splits import add_split_arg, load_split


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
    strict = (stats["unmatched_gt"] == 0 and tp["f1"] == 1.0
              and nf["f1"] == 1.0)
    return {"tp_f1": tp["f1"], "tp_precision": tp["precision"],
            "tp_recall": tp["recall"], "net_f1": nf["f1"],
            "percomp": per_component_connected_accuracy(pc, gc),
            "strict": int(strict), "unmatched_gt": stats["unmatched_gt"]}


def rebuild_nodes(wires, dets, cfg):
    """Re-run node assembly + snapping on a modified wire mask.

    Mirrors ``pipeline.run_pipeline``'s dispatch exactly, including passing
    only the Wire Crossover boxes and the individual keyword arguments --
    these builders take explicit parameters, not the config dict.
    """
    from schematic2netlist.classes import canonical_class
    ncfg = cfg["nodes"]
    method = ncfg.get("method")
    if method is None:
        method = "crossover" if ncfg.get("handle_crossovers") else "cc"
    xbox = [d for d in dets
            if canonical_class(d["class"]) == "Wire Crossover"]
    if method == "crossover":
        node_map, n = build_wire_nodes_crossover_aware(
            wires, xbox, connectivity=ncfg["connectivity"],
            relink=ncfg.get("relink", "band"))
    elif method == "vector":
        from schematic2netlist.vector_nodes import build_wire_nodes_vector
        vcfg = ncfg.get("vector", {}) or {}
        node_map, n, _info = build_wire_nodes_vector(
            wires, xbox, connectivity=ncfg["connectivity"], **vcfg)
    else:
        node_map, n = build_wire_nodes(
            wires, connectivity=ncfg["connectivity"])
    comps = build_component_pin_nets(dets, node_map, cfg)
    for c in comps:
        c["node_names"] = [None if x is None else f"n{x}" for x in c["nodes"]]
    return node_map, comps


def cut_arms_for_welds(node_map, wires, pred_of, gt_of, name_to_id, by,
                       stats: Counter):
    """Return a mask of arm pixels whose removal separates welded pairs.

    Ground truth is used ONLY here, to pick the arm.
    """
    load = defaultdict(lambda: defaultdict(list))
    for t, pn in pred_of.items():
        gn = gt_of.get(t)
        if pn is not None and gn is not None:
            load[pn][gn].append(t)

    cut = np.zeros_like(wires, dtype=np.uint8)
    for pn, nets in load.items():
        if len(nets) < 2:
            continue
        nid = name_to_id.get(pn)
        if nid is None:
            continue
        m = (node_map == nid).astype(np.uint8)
        if cv2.connectedComponents(m, connectivity=8)[0] - 1 != 1:
            stats["skipped_node_physically_disconnected"] += 1
            continue
        G, arm_lab, _s = build_arm_graph(m)
        if G is None:
            stats["skipped_graph_too_small"] += 1
            continue
        # take the two most-populated nets; a 3+ node is handled pairwise
        ordered = sorted(nets.items(), key=lambda kv: -len(kv[1]))[:2]
        (a_net, a_terms), (b_net, b_terms) = ordered
        SRC, SNK = ("SRC",), ("SNK",)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue

        def anchor(terms, is_src):
            hit = 0
            for t in terms:
                gb = by.get(t[0], {}).get("bbox")
                if gb is None:
                    continue
                k = int(np.argmin((xs - gb[0]) ** 2 + (ys - gb[1]) ** 2))
                a = nearest_arm(arm_lab, int(xs[k]), int(ys[k]))
                if a is None:
                    continue
                hit += 1
                if is_src:
                    G.add_edge(SRC, (("arm", a), "in"), capacity=float("inf"))
                else:
                    G.add_edge((("arm", a), "out"), SNK, capacity=float("inf"))
            return hit

        if not anchor(a_terms, True) or not anchor(b_terms, False):
            stats["skipped_anchor_failed"] += 1
            continue
        try:
            value, part = nx.minimum_cut(G, SRC, SNK)
        except Exception:
            stats["skipped_flow_failed"] += 1
            continue
        if not np.isfinite(value) or value <= 0:
            stats["skipped_no_finite_cut"] += 1
            continue
        # saturated arm in->out edges crossing the partition ARE the cut
        left, _right = part
        n_cut = 0
        for u in left:
            if not (isinstance(u, tuple) and len(u) == 2
                    and isinstance(u[0], tuple) and u[0][0] == "arm"
                    and u[1] == "in"):
                continue
            if (u[0], "out") in _right:
                cut[arm_lab == u[0][1]] = 255
                n_cut += 1
        stats[f"cut_{min(n_cut, 4)}_arms"] += 1
    return cut


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=190)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/oracle_bridge_cut")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out, cfg, seed)

    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)
    stats = Counter()
    rows = []

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

        node_map, wires = res["node_map"], res["clean_wires"]
        p, g, _ = align_components(
            as_pred(res["components"], res["detections"]), gcomps)
        pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)
        name_to_id = {}
        for c in res["components"]:
            for n, nn in zip(c.get("nodes", []), c.get("node_names", [])):
                if n is not None and nn is not None:
                    name_to_id[nn] = int(n)
        pred_of, gt_of = {}, {}
        for c in pc:
            for k, n in enumerate(c["nets"]):
                pred_of[(c["id"], k)] = n
        for c in gc:
            for k, net in enumerate(c["nets"]):
                gt_of[(c["id"], k)] = net

        cut = cut_arms_for_welds(node_map, wires, pred_of, gt_of,
                                 name_to_id, by, stats)
        if cut.any():
            fixed = wires.copy()
            fixed[cut > 0] = 0
            _nm2, comps2 = rebuild_nodes(fixed, res["detections"], cfg)
            after = score(as_pred(comps2, res["detections"]), gcomps)
        else:
            after = dict(base)
        rows.append({"image": nm, "n_cut_px": int((cut > 0).sum()),
                     **{f"base_{k}": v for k, v in base.items()},
                     **{f"cut_{k}": v for k, v in after.items()}})
        if i % 20 == 0:
            print(f"  [{i}/{len(names)}] strict {sum(r['base_strict'] for r in rows)}"
                  f" -> {sum(r['cut_strict'] for r in rows)}", flush=True)

    n = len(rows)
    def mean(k):
        return sum(r[k] for r in rows) / max(n, 1)
    print(f"\n=== ORACLE: cut the ONE correct bridging arm per welded pair ===")
    print(f"{n} images\n")
    print(f"  {'metric':22s} {'baseline':>9s} {'bridge-cut':>11s} {'delta':>9s}")
    for k in ("tp_f1", "tp_precision", "tp_recall", "net_f1", "percomp",
              "strict"):
        a, b = mean(f"base_{k}"), mean(f"cut_{k}")
        print(f"  {k:22s} {a:9.4f} {b:11.4f} {b-a:+9.4f}")
    print(f"\n  images improved: "
          f"{sum(1 for r in rows if r['cut_tp_f1'] > r['base_tp_f1'])}, "
          f"worsened: {sum(1 for r in rows if r['cut_tp_f1'] < r['base_tp_f1'])}")
    print(f"  strict gained: "
          f"{sum(1 for r in rows if r['cut_strict'] > r['base_strict'])}, "
          f"lost: {sum(1 for r in rows if r['cut_strict'] < r['base_strict'])}")
    print(f"\n  weld handling:")
    for k, v in sorted(stats.items()):
        print(f"    {k:42s} {v:4d}")

    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": n,
        "baseline": {k: round(mean(f"base_{k}"), 4) for k in
                     ("tp_f1", "tp_precision", "tp_recall", "net_f1",
                      "percomp", "strict")},
        "bridge_cut": {k: round(mean(f"cut_{k}"), 4) for k in
                       ("tp_f1", "tp_precision", "tp_recall", "net_f1",
                        "percomp", "strict")},
        "weld_handling": dict(stats),
        "note": ("Ground truth chooses WHICH arm to cut; the cut mask then "
                 "goes through the real node assembly, snapping and scoring. "
                 "This bounds bridge detection without assuming how it would "
                 "be done."),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
