#!/usr/bin/env python3
"""Fast wire-extraction sweep against verified GT (C2 development tool).

Scores terminal-pair F1 / net F1 / per-component connectivity on a
subset, skipping the expensive graph-edit-distance so a config can be
tuned in seconds rather than half an hour. Use scripts/benchmark.py for
the full metric cascade on the chosen config.

Optimizing the fragmentation ratio alone is a trap: welding two distinct
nets into one also lowers it while making the answer worse. This scores
the metrics that actually matter.

Usage:
    python scripts/sweep_wires.py --limit 60
"""

from __future__ import annotations

import argparse
import statistics as st
import sys

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.config import load_config, set_by_dotted_key
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import (
    net_level_metrics,
    per_component_connected_accuracy,
    terminal_pair_metrics,
)
from schematic2netlist.pipeline import run_pipeline

sys.path.insert(0, "scripts")


def score(cfg, names, gt_dir, images_dir="data/cleaned"):
    tp, nf, pc, frag = [], [], [], []
    for nm in names:
        stem = nm.rsplit(".", 1)[0]
        gt = load_gt(f"{gt_dir}/{stem}.json")
        gcomps = gt_to_components(gt)
        by_id = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by_id[c["id"]]["bbox"]

        res = run_pipeline(
            f"{images_dir}/{nm}", cfg,
            detections=load_cached_detections(f"data/detections/{stem}.json"),
        )
        dets = res["detections"]
        pcomps = [{
            "id": c["id"], "class": c["class"],
            "nets": list(c.get("node_names", [])),
            "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                     dets[c["id"]]["width"], dets[c["id"]]["height"]],
        } for c in res["components"]]

        p_a, g_a, _ = align_components(pcomps, gcomps)
        p_a = canonicalize_terminals(p_a)
        g_a = canonicalize_terminals(g_a)
        tp.append(terminal_pair_metrics(p_a, g_a)["f1"])
        nf.append(net_level_metrics(p_a, g_a)["f1"])
        pc.append(per_component_connected_accuracy(p_a, g_a))
        nets = len({t["net"] for c in gt["components"]
                    for t in c["terminals"] if t["net"]})
        if nets:
            frag.append(res["num_wire_nodes"] / nets)
    return (st.mean(tp), st.mean(nf), st.mean(pc), st.median(frag))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--gt-dir", default=None,
                    help="overrides benchmark.gt_dir from the config")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    args.gt_dir = args.gt_dir or base["benchmark"]["gt_dir"]
    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()][: args.limit]

    configs = [("canny (baseline)", set_by_dotted_key(base, "wires.method", "canny"))]
    for span in (0, 5, 9, 13, 17, 21):
        c = set_by_dotted_key(base, "wires.method", "ink")
        c = set_by_dotted_key(c, "wires.bridge_span", span)
        configs.append((f"ink span={span}", c))

    print(f"scoring {len(names)} images per config "
          f"(GT: {args.gt_dir})\n")
    print(f"{'config':22s} {'term-pair F1':>13s} {'net F1':>8s} "
          f"{'per-comp':>9s} {'frag':>6s}")
    best = None
    for label, cfg in configs:
        t, n, p, f = score(cfg, names, args.gt_dir)
        star = ""
        if best is None or t > best[1]:
            best, star = (label, t), "  <-- best term-pair F1"
        print(f"{label:22s} {t:13.4f} {n:8.4f} {p:9.4f} {f:6.2f}{star}")
    print(f"\nbest: {best[0]}")


if __name__ == "__main__":
    main()
