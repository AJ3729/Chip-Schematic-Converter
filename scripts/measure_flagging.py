#!/usr/bin/env python3
"""Can the pipeline tell you WHICH nets it probably got wrong? (C5)

This is the experiment the "usable tool" claim rests on. A netlist that
simulates but describes a different circuit is not a partial success —
it is a confident wrong answer, and worse than failing loudly. The only
honest way to ship imperfect topology recovery is for the tool to know
where it is unsure, so the user checks two flagged nets instead of
re-tracing fourteen components.

That is a measurable property, not a framing choice. Here we score
per-net risk signals against verified ground truth and ask: **of the
nets that are actually wrong, how many does the tool flag?**

Risk signals, all read off decisions the pipeline already makes — no
new model:

  intersections   stroke intersections lying on this net. Each one is a
                  connect-or-cross decision made by assumption; more of
                  them means more ways to be wrong.
  snap_radius     the largest boundary-expansion radius any terminal
                  needed to reach this net. Reaching far means weak
                  evidence.
  unsnapped       a component touching this net has a terminal that
                  never connected — the ledger already flags these.
  size            terminal count; an unusually large net is a candidate
                  weld.

A net counts as CORRECT when its set of (component, terminal) slots
matches a ground-truth net exactly. Anything else — merged, split,
partial — is wrong.

The headline is the detection curve: checking the k riskiest nets per
image, what fraction of wrong nets have you found? If that curve is
steep the tool is auditable and low strict success is survivable. If it
is flat, flagging does not work and the honest move is to say so.

Usage:
    python scripts/measure_flagging.py --limit 60
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import load_gt
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.skeleton import intersection_sites

import sys

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import gt_components, pred_components  # noqa: E402


def net_slots(components: list[dict]) -> dict:
    """net name -> frozenset of (component id, terminal index)."""
    out = defaultdict(set)
    for c in components:
        for k, net in enumerate(c["nets"]):
            if net is not None:
                out[net].add((c["id"], k))
    return {k: frozenset(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--images-dir", default=None,
                    help="preprocessed frames; defaults to "
                         "preprocess.images_dir from the config")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out-dir", default="results/flagging")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_dir = Path(args.gt_dir or cfg["benchmark"]["gt_dir"])
    det_dir = Path(cfg["detect"]["cache_dir"])
    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()][: args.limit]
    images_dir = resolve_and_check(args.images_dir, names, cfg)

    rows = []
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gp, dp = gt_dir / f"{stem}.json", det_dir / f"{stem}.json"
        if not gp.exists() or not dp.exists():
            continue
        gt = load_gt(gp)
        if not gt.get("verified"):
            continue
        print(f"[{i}/{len(names)}] {nm}", flush=True)

        dets = load_cached_detections(dp)
        result = run_pipeline(images_dir / nm, cfg, detections=dets)

        pred = pred_components(result)
        gtc = gt_components(gt)
        aligned, gt_a, _stats = align_components(pred, gtc)
        # Terminal INDEX is arbitrary for symmetric parts, so raw
        # (component, index) slots are not comparable across pred and GT.
        # The benchmark canonicalizes order by connectivity signature
        # before scoring; skipping that step reports every net as wrong.
        aligned = canonicalize_terminals(aligned)
        gt_a = canonicalize_terminals(gt_a)

        pred_nets = net_slots(aligned)
        gt_nets = net_slots(gt_a)
        gt_sets = set(gt_nets.values())

        # risk signals -------------------------------------------------
        comps = result["components"]
        node_of_slot = {}
        for c in comps:
            for k, node in enumerate(c.get("nodes", [])):
                node_of_slot[(c["id"], k)] = node
        # map original component ids to aligned ids
        orig_to_aligned = {p["id"]: a["id"] for p, a in zip(pred, aligned)}

        # intersections per predicted node
        img = cv2.imread(str(images_dir / nm))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # reuse the mask the pipeline produced by re-deriving it cheaply
        from schematic2netlist.classes import canonical_class
        from schematic2netlist.nodes import (
            build_wire_nodes, build_wire_nodes_crossover_aware)
        from schematic2netlist.textmask import detect_text_mask
        from schematic2netlist.wires import (
            build_non_wire_mask, extract_wires, stitch_wire_islands,
            stitchable_mask)
        tm = detect_text_mask(gray, cfg) if cfg["textmask"]["enabled"] else None
        _c, wires = extract_wires(gray, build_non_wire_mask(gray, dets, cfg, tm), cfg)
        if cfg["wires"].get("stitch_masked_gaps"):
            wires = stitch_wire_islands(
                wires, stitchable_mask(gray.shape, dets, cfg, tm), cfg)
        sites = intersection_sites(wires)
        # rebuild the same node map the pipeline used, so site -> net
        # attribution is exact rather than approximated
        if (cfg["nodes"].get("method") or
                ("crossover" if cfg["nodes"].get("handle_crossovers") else "cc")) == "cc":
            node_map, _n = build_wire_nodes(wires, connectivity=cfg["nodes"]["connectivity"])
        else:
            xo = [d for d in dets if canonical_class(d["class"]) == "Wire Crossover"]
            node_map, _n = build_wire_nodes_crossover_aware(
                wires, xo, connectivity=cfg["nodes"]["connectivity"])

        # Attribute each intersection to the net whose PIXELS it sits on,
        # by reading the node map directly. An earlier version used the
        # nearest component centroid, which is not the same question and
        # scattered sites onto whichever component happened to be close.
        sites_per_net = defaultdict(int)
        if sites and node_map is not None:
            name_of_node = result.get("node_name_map") or {}
            H, W = node_map.shape
            for (x, y) in sites:
                yy, xx = min(max(y, 0), H - 1), min(max(x, 0), W - 1)
                nid = int(node_map[yy, xx])
                if nid < 0:
                    # site centre can land a pixel off the stroke; look around
                    y0, y1 = max(0, yy - 2), min(H, yy + 3)
                    x0, x1 = max(0, xx - 2), min(W, xx + 3)
                    win = node_map[y0:y1, x0:x1]
                    vals = win[win >= 0]
                    if vals.size == 0:
                        continue
                    nid = int(np.bincount(vals).argmax())
                net = name_of_node.get(nid)
                if net is not None:
                    sites_per_net[net] += 1

        for net, slots in pred_nets.items():
            correct = slots in gt_sets
            n_unsnapped = sum(
                1 for c in comps
                for k, node in enumerate(c.get("nodes", []))
                if node is None
                and orig_to_aligned.get(c["id"]) is not None
                and any(s[0] == orig_to_aligned[c["id"]] for s in slots)
            )
            rows.append({
                "image": nm,
                "net": net,
                "correct": int(correct),
                "size": len(slots),
                "intersections": sites_per_net.get(net, 0),
                "unsnapped_nearby": n_unsnapped,
            })

    if not rows:
        raise SystemExit("no nets scored")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_net.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_total = len(rows)
    n_wrong = sum(1 for r in rows if not r["correct"])
    print(f"\nnets scored: {n_total}   wrong: {n_wrong} ({n_wrong/n_total:.1%})")

    # detection curve: rank by each signal, how many wrong nets are in
    # the top-k? A signal that beats the "check the biggest nets" and
    # random baselines is carrying real information.
    def curve(key, reverse=True):
        by_img = defaultdict(list)
        for r in rows:
            by_img[r["image"]].append(r)
        found = {k: 0 for k in (1, 2, 3)}
        for _im, rs in by_img.items():
            ranked = sorted(rs, key=lambda r: r[key], reverse=reverse)
            for k in found:
                found[k] += sum(1 for r in ranked[:k] if not r["correct"])
        return {k: (v / n_wrong if n_wrong else 0.0) for k, v in found.items()}

    summary = {"n_nets": n_total, "n_wrong": n_wrong,
               "wrong_rate": round(n_wrong / n_total, 4), "curves": {}}
    print(f"\n{'signal':22s} {'top-1':>8s} {'top-2':>8s} {'top-3':>8s}"
          "   (fraction of WRONG nets caught)")
    for key in ("intersections", "size", "unsnapped_nearby"):
        c = curve(key)
        summary["curves"][key] = {str(k): round(v, 4) for k, v in c.items()}
        print(f"{key:22s} {c[1]:8.3f} {c[2]:8.3f} {c[3]:8.3f}")
    # random-order baseline for reference
    import random
    random.seed(0)
    for r in rows:
        r["_rand"] = random.random()
    c = curve("_rand")
    summary["curves"]["random_baseline"] = {str(k): round(v, 4) for k, v in c.items()}
    print(f"{'random (baseline)':22s} {c[1]:8.3f} {c[2]:8.3f} {c[3]:8.3f}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json + per_net.csv")


if __name__ == "__main__":
    main()
