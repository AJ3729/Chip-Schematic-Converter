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
from schematic2netlist.splits import add_split_arg, load_split

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

        # cache_dir from the config, not a hardcoded path: sweeping a
        # config whose detections live elsewhere (e.g. the 1024-px frame,
        # data/detections_1024) silently scored 512-px boxes against
        # 1024-px images otherwise.
        res = run_pipeline(
            f"{images_dir}/{nm}", cfg,
            detections=load_cached_detections(
                f"{cfg['detect']['cache_dir']}/{stem}.json",
                min_confidence=cfg["detect"].get("confidence")),
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


def parse_value(raw: str):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--gt-dir", default=None,
                    help="overrides benchmark.gt_dir from the config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--images-dir", default="data/cleaned",
                    help="preprocessed frames matching the config's "
                         "preprocess.target_size")
    ap.add_argument("--axis", action="append", default=None, metavar="KEY=V1,V2",
                    help="sweep a dotted config key one axis at a time, e.g. "
                         "--axis wires.stitch_max_gap=40,60,90 . Repeatable. "
                         "Omit to run the default bridge-span sweep.")
    ap.add_argument("--out", default=None, help="write results as CSV")
    args = ap.parse_args()

    base = load_config(args.config)
    args.gt_dir = args.gt_dir or base["benchmark"]["gt_dir"]
    names = load_split(args.split, args.splits_dir)[: args.limit]

    if args.axis:
        # One axis at a time from the current default, so each row is
        # attributable to a single knob. A full grid hides interactions
        # behind a wall of rows and is rarely what you want first.
        configs = [("DEFAULT (baseline)", base)]
        for spec in args.axis:
            key, _, raw = spec.partition("=")
            for v in (parse_value(x) for x in raw.split(",")):
                configs.append((f"{key}={v}", set_by_dotted_key(base, key, v)))
    else:
        configs = [("canny (baseline)", set_by_dotted_key(base, "wires.method", "canny"))]
        for span in (0, 5, 9, 13, 17, 21):
            c = set_by_dotted_key(base, "wires.method", "ink")
            c = set_by_dotted_key(c, "wires.bridge_span", span)
            configs.append((f"ink span={span}", c))

    print(f"scoring {len(names)} images per config "
          f"(GT: {args.gt_dir})\n")
    print(f"{'config':34s} {'term-pair F1':>13s} {'net F1':>8s} "
          f"{'per-comp':>9s} {'frag':>6s}")
    best, rows = None, []
    for label, cfg in configs:
        t, n, p, f = score(cfg, names, args.gt_dir, images_dir=args.images_dir)
        rows.append({"config": label, "terminal_pair_f1": round(t, 4),
                     "net_f1": round(n, 4), "per_component_acc": round(p, 4),
                     "fragmentation": round(f, 3)})
        star = ""
        if best is None or t > best[1]:
            best, star = (label, t), "  <-- best term-pair F1"
        print(f"{label:34s} {t:13.4f} {n:8.4f} {p:9.4f} {f:6.2f}{star}", flush=True)
    print(f"\nbest: {best[0]}")

    if args.out:
        import csv
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
