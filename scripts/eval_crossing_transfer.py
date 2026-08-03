#!/usr/bin/env python3
"""Does a render-trained crossing classifier transfer to REAL wire masks?

This is the measurement that decides whether the classifier is worth
integrating, and it is deliberately separate from validation accuracy on
the synthetic set. The CGHD classifier scored 0.97 balanced accuracy
in-domain and 0.72 AUC on our masks; high in-domain accuracy is necessary
but nowhere near sufficient.

Ground truth on real images comes from the verified GT netlists rather
than from any annotation of crossings. For each site the pipeline reports
on a real wire mask, the arms around it are attributed to GT nets via the
node map and Hungarian-aligned components:

  - arms carrying terminals of TWO OR MORE distinct GT nets  -> must SPLIT
  - arms carrying terminals of exactly ONE GT net            -> must UNION

Sites whose surrounding ink carries no identifiable terminal are skipped
and counted, because GT cannot adjudicate them.

Reports AUC (threshold-free), plus precision/recall at a sweep of
thresholds so the operating point can be chosen against the asymmetry
that matters: a wrong split severs a net and corrupts every component on
it, while a missed split leaves a weld that was already there.

Usage:
    python scripts/eval_crossing_transfer.py \
        --weights experiments/junction/synth128_gpu/best.pt --limit 60
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import load_gt
from schematic2netlist.junction_model import crossing_probabilities
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.skeleton import intersection_sites_with_degree
from schematic2netlist.splits import add_split_arg, load_split


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def auc_score(pos, neg) -> float:
    import scipy.stats as ss
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = ss.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_split_arg(ap, "val")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--context", type=float, default=3.0)
    ap.add_argument("--cut-radius", type=int, default=7,
                    help="disk radius removed when testing whether this site "
                         "is the cut point separating two GT nets")
    ap.add_argument("--out-dir", default="results/crossing_transfer")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)

    pos, neg, rows = [], [], []
    skipped = Counter()
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        dets = load_cached_detections(
            f"{cfg['detect']['cache_dir']}/{stem}.json",
            min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(images_dir / nm, cfg, detections=dets)
        node_map, wires = res["node_map"], res["clean_wires"]
        gt_comps = gt["components"]
        pred = res["components"]

        # node id -> set of GT nets whose terminals snapped to it
        cost = np.ones((len(pred), len(gt_comps)))
        for pi, c in enumerate(pred):
            d = dets[c["id"]]
            pb = (d["x"], d["y"], d["width"], d["height"])
            for gj, g in enumerate(gt_comps):
                if canonical_class(c["class"]) == canonical_class(g["class"]):
                    cost[pi, gj] = 1.0 - iou(pb, g["bbox"])
        # node id -> {gt_net: [component box centre, ...]}, so a net can be
        # ANCHORED to pixels of the node for the cut test below
        node_nets: dict[int, dict] = {}
        if len(pred) and len(gt_comps):
            ri, ci = linear_sum_assignment(cost)
            for pi, gj in zip(ri, ci):
                if cost[pi, gj] > 0.7:
                    continue
                gnets = [t["net"] for t in gt_comps[gj]["terminals"]]
                gb = gt_comps[gj]["bbox"]
                for k, n in enumerate(pred[pi].get("nodes", [])):
                    if n is None or k >= len(gnets) or gnets[k] is None:
                        continue
                    node_nets.setdefault(int(n), {}).setdefault(
                        gnets[k], []).append((gb[0], gb[1]))

        sites = intersection_sites_with_degree((wires > 0).astype(np.uint8))
        if not sites:
            continue
        probs = crossing_probabilities(
            wires, [(x, y) for x, y, _ in sites], args.weights,
            context=args.context)

        H, W = node_map.shape
        for (x, y, deg), p in zip(sites, probs):
            nid = int(node_map[y, x]) if node_map[y, x] >= 0 else -1
            if nid < 0:
                ys, xs = np.nonzero(node_map[max(0, y-3):y+4,
                                             max(0, x-3):x+4] >= 0)
                if ys.size == 0:
                    skipped["site_off_any_node"] += 1
                    continue
                nid = int(node_map[max(0, y-3)+ys[0], max(0, x-3)+xs[0]])
            nets = node_nets.get(nid, {})
            if not nets:
                skipped["node_carries_no_gt_terminal"] += 1
                continue
            if len(nets) < 2:
                label = 0            # node is not welded: nothing to split
            else:
                # CAUSAL test. Asking "does this node carry >=2 nets?" labels
                # every site on a welded node as must-split, including plain
                # junctions --- a welded node spans much of the drawing, so
                # that measures coincidence. Instead cut a disk at THIS site
                # and ask whether the fused nets actually separate. Only then
                # is this site the weld's cut point.
                mask = (node_map == nid).astype(np.uint8)
                probe = mask.copy()
                cv2.circle(probe, (x, y), args.cut_radius, 0, -1)
                n_lab, lab = cv2.connectedComponents(probe, connectivity=8)
                anchors = {}
                for net, centres in nets.items():
                    bx, by = centres[0]
                    ys, xs = np.nonzero(mask)
                    if ys.size == 0:
                        continue
                    k = int(np.argmin((xs - bx) ** 2 + (ys - by) ** 2))
                    anchors[net] = int(lab[ys[k], xs[k]])
                comp_ids = [v for v in anchors.values() if v != 0]
                label = 1 if len(set(comp_ids)) >= 2 else 0
            (pos if label else neg).append(float(p))
            rows.append({"image": nm, "x": x, "y": y, "degree": deg,
                         "label": label, "prob": round(float(p), 4),
                         "n_gt_nets_on_node": len(nets)})
        if i % 10 == 0:
            print(f"[{i}/{len(names)}] sites scored={len(rows)}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_site.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    P, N = np.array(pos), np.array(neg)
    sweep = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tp = int((P >= thr).sum()); fn = int((P < thr).sum())
        fp = int((N >= thr).sum()); tn = int((N < thr).sum())
        sweep.append({
            "threshold": thr,
            "split_recall": round(tp/max(tp+fn, 1), 4),
            "split_precision": round(tp/max(tp+fp, 1), 4),
            "union_recall": round(tn/max(tn+fp, 1), 4),
            "balanced_acc": round(0.5*(tp/max(tp+fn,1) + tn/max(tn+fp,1)), 4),
            "wrong_splits": fp, "missed_splits": fn,
        })
    summary = {
        "weights": args.weights, "n_images": len(names),
        "n_sites_scored": len(rows),
        "n_must_split": len(pos), "n_must_union": len(neg),
        "skipped": dict(skipped),
        "AUC": round(auc_score(pos, neg), 4),
        "threshold_sweep": sweep,
        "reference": ("CGHD classifier reached AUC 0.72 on real masks with "
                      "0.97 in-domain accuracy; >0.85 here would be a real "
                      "improvement, ~0.72 means the render domain did not "
                      "close the gap."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
