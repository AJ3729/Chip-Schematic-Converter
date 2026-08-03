#!/usr/bin/env python3
"""Ceiling for deciding WHICH detected crossover boxes to honour.

Reframes the crossing problem after five measurements pointed the same
way. Asking "is this intersection a crossing?" at every site failed every
time it was tried (learned classifier -0.110; perfect GT boxes -0.026
strict; vector geometric pairing; every threshold sweep preferring fewer
splits). But crossover-aware net assembly still beats plain connected
components by +0.074, and making the notch placement-invariant
(relink=snap) LOSES 0.0236 terminal-pair F1 and 0.0368 strict.

Those only reconcile one way: the box-centred notch severs some boxes and
misses others, and the misses are frequently CORRECT — at many detected
Wire Crossover boxes the two conductors really are on one net. The
mechanism's aggregate benefit therefore comes from severing roughly the
right FRACTION of boxes, by luck rather than by judgement.

So the decision worth learning is not per-intersection but per-box, and
there are only ~1.2 detected boxes per image instead of ~20 sites. This
script measures the ceiling of getting that decision right: for each
image it enumerates every subset of its crossover boxes, notches exactly
that subset, scores the result, and keeps the best. The gap between that
and the shipped default is the headroom available to a box-level
classifier.

Enumeration is 2^n per image; images with more than ``--max-boxes`` boxes
are skipped and counted rather than sampled, so the reported mean is over
a stated subset.

Usage:
    python scripts/oracle_box_decisions.py --limit 60
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import (
    net_level_metrics, per_component_connected_accuracy, terminal_pair_metrics)
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.splits import add_split_arg, load_split


def score(res, gcomps):
    dets = res["detections"]
    pred = [{
        "id": c["id"], "class": c["class"],
        "nets": list(c.get("node_names", [])),
        "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                 dets[c["id"]]["width"], dets[c["id"]]["height"]],
    } for c in res["components"]]
    p, g, _ = align_components(pred, gcomps)
    p, g = canonicalize_terminals(p), canonicalize_terminals(g)
    return (terminal_pair_metrics(p, g)["f1"],
            net_level_metrics(p, g)["f1"],
            per_component_connected_accuracy(p, g))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_split_arg(ap, "val")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-boxes", type=int, default=5,
                    help="skip images with more boxes than this (2^n cost)")
    ap.add_argument("--out-dir", default="results/oracle_box_decisions")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)

    rows, skipped = [], 0
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        gcomps = gt_to_components(gt)
        by_id = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by_id[c["id"]]["bbox"]
        dets = load_cached_detections(
            f"{cfg['detect']['cache_dir']}/{stem}.json",
            min_confidence=cfg["detect"].get("confidence"))
        xidx = [k for k, d in enumerate(dets)
                if canonical_class(d["class"]) == "Wire Crossover"]
        if len(xidx) > args.max_boxes:
            skipped += 1
            continue

        # baseline: honour every detected box (the shipped default)
        base = score(run_pipeline(images_dir / nm, cfg, detections=dets),
                     gcomps)

        best = base
        best_keep = tuple(xidx)
        for r in range(len(xidx) + 1):
            for keep in itertools.combinations(xidx, r):
                if set(keep) == set(xidx):
                    continue                     # that is the baseline
                trimmed = [d for k, d in enumerate(dets)
                           if k not in xidx or k in keep]
                s = score(run_pipeline(images_dir / nm, cfg,
                                       detections=trimmed), gcomps)
                if s[0] > best[0]:
                    best, best_keep = s, keep
        rows.append({
            "image": nm, "n_boxes": len(xidx),
            "n_honoured_best": len(best_keep),
            "tp_f1_default": round(base[0], 4),
            "tp_f1_best": round(best[0], 4),
            "tp_f1_gain": round(best[0] - base[0], 4),
            "net_f1_default": round(base[1], 4),
            "net_f1_best": round(best[1], 4),
            "percomp_default": round(base[2], 4),
            "percomp_best": round(best[2], 4),
        })
        if i % 10 == 0:
            print(f"[{i}/{len(names)}] scored={len(rows)} skipped={skipped}",
                  flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    with_boxes = [r for r in rows if r["n_boxes"] > 0]
    summary = {
        "n_scored": len(rows),
        "n_skipped_too_many_boxes": skipped,
        "n_images_with_boxes": len(with_boxes),
        "tp_f1_default": round(m("tp_f1_default"), 4),
        "tp_f1_perfect_box_decisions": round(m("tp_f1_best"), 4),
        "tp_f1_headroom": round(m("tp_f1_best") - m("tp_f1_default"), 4),
        "net_f1_default": round(m("net_f1_default"), 4),
        "net_f1_perfect": round(m("net_f1_best"), 4),
        "percomp_default": round(m("percomp_default"), 4),
        "percomp_perfect": round(m("percomp_best"), 4),
        "images_improved": sum(1 for r in rows if r["tp_f1_gain"] > 0),
        "mean_boxes_per_image": round(m("n_boxes"), 2),
        "interpretation": (
            "tp_f1_headroom is the ceiling for a per-box honour/ignore "
            "classifier, holding everything else fixed. Compare it with the "
            "+0.074 that crossover-aware currently gains over plain CC."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
