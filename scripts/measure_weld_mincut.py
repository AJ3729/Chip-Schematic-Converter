#!/usr/bin/env python3
"""How many cuts does it take to separate a welded pair of nets?

Two measurements disagree and the plan depends on which is right.
``scripts/diagnose_connectivity.py`` says 33.7% of predicted nodes carry
exactly TWO ground-truth nets, which is a pairwise fusion and should need one
separation. But the single-disk cure test in ``scripts/diagnose_defects.py``
finds that 91.8% of welds have NO single site whose removal separates them.
Both cannot be casually true: either the cure test is too strict (a 7 px disk
failing to sever a thick stroke, say) or the fused nets are joined along
SEVERAL independent paths, in which case cutting one changes nothing.

The distinction decides whether local repair is viable at all. One path means
a single correct decision fixes the net. Three paths means every local
mechanism -- notch, classifier, constraint -- is attacking one third of a
problem and will appear to do nothing, which is exactly the pattern the
oracle already showed when perfect crossover boxes made strict success worse.

This measures it exactly instead of sampling disks. The welded node's
skeleton becomes a graph: branch sites are vertices, the arms between them
are unit-capacity edges. Terminals of one GT net feed a super-source and
terminals of the other feed a super-sink. Max-flow from source to sink then
EQUALS the minimum number of arms that must be cut (Menger's theorem), and
it is also the number of edge-disjoint fusion paths.

Reported as a distribution, because a mean over "1 cut" and "7 cuts" cases
would hide the only thing that matters.

Usage:
    python scripts/measure_weld_mincut.py --limit 40
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
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.skeleton import thin

_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def build_arm_graph(node_mask: np.ndarray, min_sep: int = 9):
    """Skeleton of one predicted node as a graph of sites joined by arms.

    Returns ``(G, arm_label, site_of)`` where ``G`` has a vertex per branch
    site plus a vertex per arm, and unit-capacity edges joining an arm to
    every site it touches. Routing flow through an ARM vertex (rather than
    along a site-to-site edge) is what makes max-flow count ARM cuts, which
    is the physical operation available: an arm is a length of wire that can
    be severed anywhere along it.
    """
    skel = thin(node_mask).astype(np.uint8)
    if not skel.any():
        return None, None, None
    neigh = ndimage.convolve(skel.astype(np.int32), _NEIGH, mode="constant")
    branch = ((skel > 0) & (neigh >= 3)).astype(np.uint8)
    grown = cv2.dilate(branch, np.ones((min_sep, min_sep), np.uint8)) \
        if branch.any() else np.zeros_like(skel)
    n_site, site_lab = cv2.connectedComponents(grown, connectivity=8)
    cut = skel.copy()
    cut[grown > 0] = 0
    n_arm, arm_lab = cv2.connectedComponents(cut, connectivity=8)
    if n_arm <= 1:
        return None, None, None

    G = nx.DiGraph()
    # which sites does each arm touch?
    dil = cv2.dilate((arm_lab > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    touch = defaultdict(set)
    ys, xs = np.nonzero((dil > 0) & (grown > 0))
    for y, x in zip(ys, xs):
        s = int(site_lab[y, x])
        if s == 0:
            continue
        # any arm pixel in the 3x3 around this site pixel
        y0, y1 = max(0, y - 1), y + 2
        x0, x1 = max(0, x - 1), x + 2
        for a in np.unique(arm_lab[y0:y1, x0:x1]):
            if a > 0:
                touch[int(a)].add(s)

    for a, sites in touch.items():
        an = ("arm", a)
        for s in sites:
            sn = ("site", s)
            # arm as a capacitated vertex: in -> out, capacity 1
            G.add_edge(sn, (an, "in"), capacity=float("inf"))
            G.add_edge((an, "out"), sn, capacity=float("inf"))
        G.add_edge((an, "in"), (an, "out"), capacity=1.0)
    return G, arm_lab, site_lab


def protect_arm(G, a):
    """Make an arm uncuttable.

    An anchor arm is the stub carrying a component's own terminal into the
    node, so it is part of the net by definition and severing it is not a
    repair. Left at capacity 1 it becomes the trivial minimum cut -- the
    source attaches to it with infinite capacity, so max-flow reports 1 for
    almost every weld and the "75% separable by one cut" reading was an
    artefact of that, not a property of the welds. Cutting those arms in the
    oracle changed nothing at all, which is how it surfaced.
    """
    key = (("arm", a), "in"), (("arm", a), "out")
    if G.has_edge(*key):
        G[key[0]][key[1]]["capacity"] = float("inf")


def nearest_arm(arm_lab, x, y, r=25):
    """Arm id nearest to a point, searching outward."""
    H, W = arm_lab.shape
    for rr in range(1, r):
        y0, y1 = max(0, y - rr), min(H, y + rr + 1)
        x0, x1 = max(0, x - rr), min(W, x + rr + 1)
        sub = arm_lab[y0:y1, x0:x1]
        v = np.unique(sub[sub > 0])
        if v.size:
            return int(v[0])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/weld_mincut")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bench = {r["image"]: r for r in csv.DictReader(
        open("results/benchmark_1024/seed0/per_image.csv"))}
    strict = lambda r: r["strict_success"] in ("True", "1", "true")
    tgt = sorted(im for im, r in bench.items()
                 if int(r["unmatched_gt"]) == 0 and not strict(r))[: args.limit]
    images_dir = resolve_and_check(None, tgt, cfg)

    cuts = Counter()
    rows = []
    for i, nm in enumerate(tgt, 1):
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
        node_map = res["node_map"]
        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [res["detections"][c["id"]]["x"],
                          res["detections"][c["id"]]["y"],
                          res["detections"][c["id"]]["width"],
                          res["detections"][c["id"]]["height"]]}
                for c in res["components"]]
        p, g, _ = align_components(pred, gcomps)
        pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)
        name_to_id = {}
        for c in res["components"]:
            for n, nn in zip(c.get("nodes", []), c.get("node_names", [])):
                if n is not None and nn is not None:
                    name_to_id[nn] = int(n)
        pred_of = {}
        for c in pc:
            for k, n in enumerate(c["nets"]):
                pred_of[(c["id"], k)] = n
        gt_of = {}
        for c in gc:
            for k, net in enumerate(c["nets"]):
                gt_of[(c["id"], k)] = net

        # predicted node -> {gt net: [terminal, ...]}
        load = defaultdict(lambda: defaultdict(list))
        for t, pn in pred_of.items():
            gn = gt_of.get(t)
            if pn is not None and gn is not None:
                load[pn][gn].append(t)

        for pn, nets in load.items():
            if len(nets) != 2:            # measure the pairwise case
                continue
            nid = name_to_id.get(pn)
            if nid is None:
                continue
            m = (node_map == nid).astype(np.uint8)
            G, arm_lab, _site = build_arm_graph(m)
            if G is None:
                cuts["graph_too_small"] += 1
                continue
            (a_net, a_terms), (b_net, b_terms) = list(nets.items())
            SRC, SNK = ("SRC",), ("SNK",)
            ok = True
            for terms, endpoint, cap_dir in ((a_terms, SRC, "src"),
                                             (b_terms, SNK, "snk")):
                hit = 0
                for t in terms:
                    gb = by.get(t[0], {}).get("bbox")
                    if gb is None:
                        continue
                    ys, xs = np.nonzero(m)
                    if ys.size == 0:
                        continue
                    k = int(np.argmin((xs - gb[0])**2 + (ys - gb[1])**2))
                    a = nearest_arm(arm_lab, int(xs[k]), int(ys[k]))
                    if a is None:
                        continue
                    hit += 1
                    protect_arm(G, a)   # the net's own stub is not a cut
                    if cap_dir == "src":
                        G.add_edge(SRC, ((("arm", a)), "in"),
                                   capacity=float("inf"))
                    else:
                        G.add_edge((("arm", a), "out"), SNK,
                                   capacity=float("inf"))
                if hit == 0:
                    ok = False
            if not ok or SRC not in G or SNK not in G:
                cuts["anchor_failed"] += 1
                continue
            try:
                flow = nx.maximum_flow_value(G, SRC, SNK)
            except Exception:
                cuts["flow_failed"] += 1
                continue
            k = int(round(flow))
            if not np.isfinite(flow):
                cuts["infinite (anchors share an arm)"] += 1
                continue
            cuts[k] += 1
            rows.append({"image": nm, "pred_node": pn,
                         "gt_net_a": a_net, "gt_net_b": b_net,
                         "min_arm_cuts": k})
        if i % 10 == 0:
            print(f"  [{i}/{len(tgt)}] pairwise welds measured={len(rows)}",
                  flush=True)

    print(f"\n=== MINIMUM ARM CUTS TO SEPARATE A PAIRWISE WELD ===")
    print(f"({len(rows)} welded node/net-pairs over {len(tgt)} images)\n")
    numeric = {k: v for k, v in cuts.items() if isinstance(k, int)}
    tot = sum(numeric.values()) or 1
    for k in sorted(numeric):
        print(f"  {k:2d} cut(s)  {numeric[k]:4d}  {numeric[k]/tot:6.1%}")
    for k, v in cuts.items():
        if not isinstance(k, int):
            print(f"  {k:34s} {v:4d}")
    one = numeric.get(1, 0)
    two = one + numeric.get(2, 0)
    print(f"\n  separable by ONE cut : {one}/{tot} = {one/tot:.1%}")
    print(f"  separable by <=2 cuts: {two}/{tot} = {two/tot:.1%}")
    print(f"\n  A single cut sufficing means one correct decision fixes the")
    print(f"  net. Needing three or more means every local mechanism is")
    print(f"  attacking a fraction of the problem and will read as no-op.")

    if rows:
        with (out / "pairwise_welds.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": len(tgt), "n_pairwise_welds": len(rows),
        "min_arm_cuts_distribution": {str(k): v for k, v in
                                      sorted(numeric.items())},
        "non_numeric": {k: v for k, v in cuts.items()
                        if not isinstance(k, int)},
        "separable_by_one_cut": round(one / tot, 4),
        "separable_by_two_cuts": round(two / tot, 4),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
