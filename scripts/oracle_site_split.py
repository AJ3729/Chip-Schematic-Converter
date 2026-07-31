#!/usr/bin/env python3
"""What would PERFECT crossing decisions at ARBITRARY sites be worth?

This is the crossing oracle that had never actually been run, and the
distinction matters. The existing GT-crossover oracle injects the dataset's
``Wire Crossover`` boxes, but those annotate DRAWN HOP SYMBOLS -- the little
semicircle a draughtsman puts where wires pass without touching. It changes only
16 of 190 images. A plain X crossing, where two wires simply cross with no
symbol, is not annotated anywhere, so "perfect crossover boxes are worth less
than nothing" was never a statement about crossings in general.

Everything measured points at those unannotated X crossings as the residual:

  - precision does NOT recover as bridging is removed (0.7419 at bridge_span 3
    against 0.7497 at 7), so the welds are in the raw ink rather than
    manufactured by morphology;
  - GT-guided arm CUTTING is null -- 280 cuts, strict success unchanged --
    which is what you expect at an X, where the correct operation is to notch
    and RE-PAIR opposite arms, not to sever one;
  - the notch-and-relink machinery already exists but only fires where a
    crossover box was detected.

So: take the skeleton's own intersection sites, and let GROUND TRUTH decide at
each one whether to split. Greedy, because the decisions interact -- splitting
one site changes which nets exist and therefore whether the next split helps.
Each candidate site is tried as a synthesised crossover box, nodes are rebuilt
through the pipeline's own dispatch, and the split is kept only if the image's
terminal-pair F1 improves.

The result is an UPPER BOUND on any per-site classifier, however good. If it is
large, the crossing problem is a learning problem and the 0.659-AUC classifier
is worth improving. If it is small, crossings are genuinely closed and the
remaining error is not recoverable from the drawing at all.

Usage:
    python scripts/oracle_site_split.py --limit 60
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

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist import skeleton as sk
from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import (net_level_metrics,
                                       per_component_connected_accuracy,
                                       terminal_pair_metrics)
from schematic2netlist.nodes import (build_wire_nodes,
                                     build_wire_nodes_crossover_aware)
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.snapping import build_component_pin_nets


def rebuild(wires, dets, cfg, extra_boxes):
    """Nodes + components, through the SAME dispatch the pipeline uses."""
    ncfg = cfg["nodes"]
    method = ncfg.get("method") or (
        "crossover" if ncfg.get("handle_crossovers") else "cc")
    xb = [d for d in dets
          if canonical_class(d["class"]) == "Wire Crossover"] + list(extra_boxes)
    if method == "crossover":
        node_map, _n = build_wire_nodes_crossover_aware(
            wires, xb, connectivity=ncfg["connectivity"],
            relink=ncfg.get("relink", "band"))
    else:
        node_map, _n = build_wire_nodes(
            wires, connectivity=ncfg["connectivity"])
    comps = build_component_pin_nets(dets, node_map, cfg)
    for c in comps:
        c["node_names"] = [None if x is None else f"n{x}" for x in c["nodes"]]
    return node_map, comps


def score(comps, dets, gcomps):
    pred = [{"id": c["id"], "class": c["class"],
             "nets": list(c.get("node_names", [])),
             "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                      dets[c["id"]]["width"], dets[c["id"]]["height"]]}
            for c in comps]
    p, g, stats = align_components(pred, gcomps)
    pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)
    tp = terminal_pair_metrics(pc, gc)
    nf = net_level_metrics(pc, gc)
    return {"tp_f1": tp["f1"], "net_f1": nf["f1"],
            "percomp": per_component_connected_accuracy(pc, gc),
            "strict": int(stats["unmatched_gt"] == 0 and tp["f1"] == 1.0
                          and nf["f1"] == 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--config", default=None)
    ap.add_argument("--site-box", type=int, default=None,
                    help="px box synthesised at a site; defaults to "
                         "nodes.junction_site_box")
    ap.add_argument("--min-degree", type=int, default=3,
                    help="3 reaches T sites as well as X sites")
    ap.add_argument("--max-sites", type=int, default=40,
                    help="cap per image; greedy is O(sites^2) rebuilds")
    ap.add_argument("--out-dir", default="results/oracle_site_split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    box = args.site_box or cfg["nodes"].get("junction_site_box", 30)
    idir = Path(cfg["preprocess"]["images_dir"])
    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    names = names[: args.limit]

    rows = []
    acts = Counter()
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gp = Path(cfg["benchmark"]["gt_dir"]) / f"{stem}.json"
        dp = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
        ip = idir / nm
        if not (gp.exists() and dp.exists() and ip.exists()):
            continue
        gt = load_gt(str(gp))
        gcomps = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            str(dp), min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(str(ip), cfg, detections=dets)
        wires = res["clean_wires"]

        base = score(res["components"], res["detections"], gcomps)

        thin = sk.thin(wires)
        try:
            sites = sk.intersection_sites_with_degree(thin, min_sep=9)
        except Exception:
            sites = [(x, y, 4) for x, y in sk.intersection_sites(thin, min_sep=9)]
        cand = [(int(s[0]), int(s[1])) for s in sites
                if (len(s) < 3 or int(s[2]) >= args.min_degree)]
        cand = cand[: args.max_sites]

        chosen, cur = [], dict(base)
        improved = True
        while improved:
            improved = False
            best = None
            for (x, y) in cand:
                if (x, y) in chosen:
                    continue
                trial = chosen + [(x, y)]
                boxes = [{"class": "Wire Crossover", "confidence": 1.0,
                          "x": float(px), "y": float(py),
                          "width": float(box), "height": float(box)}
                         for px, py in trial]
                _nm2, comps2 = rebuild(wires, res["detections"], cfg, boxes)
                s = score(comps2, res["detections"], gcomps)
                if s["tp_f1"] > cur["tp_f1"] + 1e-9 and (
                        best is None or s["tp_f1"] > best[1]["tp_f1"]):
                    best = ((x, y), s)
            if best is not None:
                chosen.append(best[0])
                cur = best[1]
                improved = True
                acts["site_split_accepted"] += 1
        acts["candidate_sites"] += len(cand)
        rows.append({"image": nm, "n_sites": len(cand), "n_split": len(chosen),
                     **{f"base_{k}": v for k, v in base.items()},
                     **{f"orc_{k}": v for k, v in cur.items()}})
        if i % 10 == 0:
            print(f"  [{i}/{len(names)}] strict "
                  f"{sum(r['base_strict'] for r in rows)} -> "
                  f"{sum(r['orc_strict'] for r in rows)}  "
                  f"splits={sum(r['n_split'] for r in rows)}", flush=True)

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / max(n, 1)
    print(f"\n=== ORACLE: PERFECT SPLIT DECISIONS AT ARBITRARY SITES "
          f"({n} images) ===\n")
    print(f"  {'metric':12s} {'baseline':>10s} {'oracle':>10s} {'delta':>10s}")
    for k in ("tp_f1", "net_f1", "percomp", "strict"):
        b, o = mean(f"base_{k}"), mean(f"orc_{k}")
        print(f"  {k:12s} {b:10.4f} {o:10.4f} {o-b:+10.4f}")
    gained = sum(1 for r in rows if r["orc_strict"] > r["base_strict"])
    lost = sum(1 for r in rows if r["orc_strict"] < r["base_strict"])
    print(f"\n  strict gained {gained}, lost {lost}")
    print(f"  sites considered {acts['candidate_sites']}, "
          f"splits accepted {acts['site_split_accepted']}")
    print(f"\n  This is an UPPER BOUND on any per-site crossing classifier.")
    print(f"  A large number means the 0.659-AUC classifier is worth improving;")
    print(f"  a small one means crossings are closed and the residual error is")
    print(f"  not recoverable from the drawing.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "per_image.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": n, "actions": dict(acts),
        **{f"base_{k}": round(mean(f"base_{k}"), 4)
           for k in ("tp_f1", "net_f1", "percomp", "strict")},
        **{f"orc_{k}": round(mean(f"orc_{k}"), 4)
           for k in ("tp_f1", "net_f1", "percomp", "strict")},
        "strict_gained": gained, "strict_lost": lost,
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
