#!/usr/bin/env python3
"""Sweep config parameters on the two axes that actually matter: welds and splits.

Every uniform parameter in this pipeline has been reported as "already at a
local optimum", but those sweeps optimised terminal-pair F1 -- a single number
that mixes two opposed failures. Welding two nets adds spurious pairs and costs
precision; splitting one costs recall. A parameter that trades a weld for a
split can leave F1 unchanged while moving the thing strict success depends on,
and F1 cannot show that.

This measures both directly, per config, on the same images:

  weld rate   fraction of predicted nodes carrying two or more GT nets
  split rate  fraction of GT nets whose terminals land on several nodes

It also skips SPICE and nGED, so it costs roughly a quarter of a benchmark run
and can afford to sweep. Anything promising here still has to be confirmed by
the real benchmark -- this finds candidates, it does not adopt them.

Configs are given as ``label=path.yaml`` so the comparison is explicit about
what it ran.

Usage:
    python scripts/sweep_weld_split.py --limit 60 \
        base=configs/default.yaml bs9=scratchpad/plan_cfg/bs9.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.pipeline import run_pipeline


def measure(cfg, names):
    idir = Path(cfg["preprocess"]["images_dir"])
    cdir = Path(cfg["detect"]["cache_dir"])
    gdir = Path(cfg["benchmark"]["gt_dir"])
    load = Counter()
    n_nodes = n_nets = n_split = n_img = 0
    n_lost = 0
    for nm in names:
        st = Path(nm).stem
        gp, dp, ip = gdir / f"{st}.json", cdir / f"{st}.json", idir / nm
        if not (gp.exists() and dp.exists() and ip.exists()):
            continue
        gt = load_gt(str(gp))
        gc_ = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gc_:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            str(dp), min_confidence=cfg["detect"].get("confidence"))
        r = run_pipeline(str(ip), cfg, detections=dets)
        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [r["detections"][c["id"]]["x"],
                          r["detections"][c["id"]]["y"],
                          r["detections"][c["id"]]["width"],
                          r["detections"][c["id"]]["height"]]}
                for c in r["components"]]
        p, g, _ = align_components(pred, gc_)
        pc, gcn = canonicalize_terminals(p), canonicalize_terminals(g)
        pof, gof = {}, {}
        for c in pc:
            for k, x in enumerate(c["nets"]):
                pof[(c["id"], k)] = x
        for c in gcn:
            for k, x in enumerate(c["nets"]):
                gof[(c["id"], k)] = x
        # weld axis: distinct GT nets per predicted node
        nn = defaultdict(set)
        for t, pn in pof.items():
            gn = gof.get(t)
            if pn is not None and gn is not None:
                nn[pn].add(gn)
        for pn, s in nn.items():
            load[min(len(s), 3)] += 1
            n_nodes += 1
        # split axis: distinct predicted nodes per GT net
        gn_nodes = defaultdict(set)
        for t, gn in gof.items():
            if gn is None:
                continue
            pn = pof.get(t)
            if pn is None:
                n_lost += 1
            else:
                gn_nodes[gn].add(pn)
        for gn, s in gn_nodes.items():
            n_nets += 1
            if len(s) > 1:
                n_split += 1
        n_img += 1
    return {
        "n_images": n_img, "n_nodes": n_nodes, "n_gt_nets": n_nets,
        "weld_rate": round((n_nodes - load[1]) / max(n_nodes, 1), 4),
        "pairwise_weld_rate": round(load[2] / max(n_nodes, 1), 4),
        "mega_weld_rate": round(load[3] / max(n_nodes, 1), 4),
        "split_rate": round(n_split / max(n_nets, 1), 4),
        "lost_terminals": n_lost,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="label=path.yaml")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="results/sweeps/weld_split.csv")
    args = ap.parse_args()

    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    names = names[: args.limit]
    rows = []
    print(f"{'config':26s} {'imgs':>5s} {'nodes':>6s} {'WELD':>7s} "
          f"{'pair':>7s} {'mega':>6s} {'SPLIT':>7s} {'lost':>5s}")
    for spec in args.configs:
        label, _, path = spec.partition("=")
        cfg = load_config(path if path and path != "configs/default.yaml"
                          else None)
        m = measure(cfg, names)
        m["config"] = label
        m["path"] = path
        rows.append(m)
        print(f"{label:26s} {m['n_images']:5d} {m['n_nodes']:6d} "
              f"{m['weld_rate']:7.4f} {m['pairwise_weld_rate']:7.4f} "
              f"{m['mega_weld_rate']:6.4f} {m['split_rate']:7.4f} "
              f"{m['lost_terminals']:5d}", flush=True)

    base = rows[0]
    print(f"\ndeltas vs {base['config']} (negative is better on both axes):")
    for r in rows[1:]:
        dw = r["weld_rate"] - base["weld_rate"]
        ds = r["split_rate"] - base["split_rate"]
        verdict = ("BOTH BETTER" if dw < 0 and ds < 0 else
                   "BOTH WORSE" if dw > 0 and ds > 0 else
                   "traded (weld for split)" if dw < 0 else
                   "traded (split for weld)")
        print(f"  {r['config']:26s} weld {dw:+.4f}  split {ds:+.4f}   {verdict}")
    print(f"\n  A config that is BOTH BETTER is a genuine candidate; a trade")
    print(f"  needs the real benchmark to say whether strict success gains.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
