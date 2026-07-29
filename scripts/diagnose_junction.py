#!/usr/bin/env python3
"""Diagnose why the CGHD-trained junction/crossover classifier transfers
poorly into the pipeline (C2 investigation).

The classifier reaches 0.97 balanced accuracy on drafter-disjoint CGHD
validation, yet integrating it as ``nodes.method: learned`` does not help
the pipeline. This script isolates the reason with four measurements,
each on the 1024-px default frames unless noted:

1. TRAINING vs INFERENCE patch statistics (ink density). Establishes that
   the two distributions differ and by how much.

2. SCALE robustness. Takes labelled CGHD val patches and simulates the
   inference scale by zooming, measuring how much accuracy that ALONE
   costs. Answer: little — the model is scale-robust, so scale is not the
   culprit.

3. REAL-PATCH discrimination (AUC), using the detector's own Wire
   Crossover boxes as a weak positive label and other degree-4 sites as
   the negative. This is where the collapse shows: in-domain 0.97 falls
   to ~0.72 on our masks.

4. The THINNING intervention. Our 1024 masks are 3-5 px thick after
   morphology; training strokes are ~1-2 px. Skeletonizing the mask
   before cropping recovers AUC toward 0.80.

Weak-label caveat: detector crossovers are noisy positives and "other
degree-4 sites" contains undetected true crossings, so the absolute AUC
is a lower bound on the achievable separation. The COMPARISON across
variants is what the argument rests on, and label noise is common to all.

Usage:
    python scripts/diagnose_junction.py --limit 40
"""

from __future__ import annotations

import argparse
import glob
import random

import numpy as np


def build_mask(nm, cfg):
    from schematic2netlist.detect import load_cached_detections
    from schematic2netlist.preprocess import preprocess_image_meta
    from schematic2netlist.textmask import detect_text_mask
    from schematic2netlist.wires import (
        build_non_wire_mask, extract_wires, stitch_wire_islands, stitchable_mask)

    stem = nm[:-4]
    dets = load_cached_detections(
        f"{cfg['detect']['cache_dir']}/{stem}.json",
        min_confidence=cfg["detect"].get("confidence"))
    out = preprocess_image_meta(f"{cfg['preprocess']['images_dir']}/{nm}", cfg)
    gray = out[0] if isinstance(out, tuple) else out
    tm = detect_text_mask(gray, cfg) if cfg["textmask"]["enabled"] else None
    _, cw = extract_wires(gray, build_non_wire_mask(gray, dets, cfg, tm), cfg)
    if cfg["wires"].get("stitch_masked_gaps"):
        cw = stitch_wire_islands(cw, stitchable_mask(gray.shape, dets, cfg, tm), cfg)
    return cw, dets


def auc(pos, neg):
    import scipy.stats as ss
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = ss.rankdata(np.concatenate([pos, neg]))
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    import cv2
    import torch

    from schematic2netlist.classes import canonical_class
    from schematic2netlist.config import load_config
    from schematic2netlist.junction_model import load_model
    from schematic2netlist.nodes import bbox_xyxy
    from schematic2netlist.skeleton import (
        crop_site, intersection_sites_with_degree, thin)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--train-dir", default="data/junctions_full")
    args = ap.parse_args()

    cfg = load_config(args.config)
    weights = cfg["nodes"]["junction_weights"]
    model, size = load_model(weights)

    def probs(mask, sites, half):
        if not sites:
            return np.zeros(0)
        P = np.stack([crop_site(mask, x, y, half, size)
                      for x, y in sites]).astype(np.float32) / 255.0
        with torch.no_grad():
            return torch.softmax(
                model(torch.from_numpy(P).unsqueeze(1)), 1)[:, 1].numpy()

    # (1) training patch ink density
    random.seed(0)
    def density(paths):
        d = [(cv2.imread(p, 0) > 127).mean()
             for p in paths if cv2.imread(p, 0) is not None]
        return np.array(d)
    trx = glob.glob(f"{args.train_dir}/train/crossover/*.png")
    trj = glob.glob(f"{args.train_dir}/train/junction/*.png")
    if trx and trj:
        dx = density(random.sample(trx, min(300, len(trx))))
        dj = density(random.sample(trj, min(300, len(trj))))
        print(f"[1] TRAINING ink density: crossover {dx.mean():.3f}, "
              f"junction {dj.mean():.3f}")

    # (3)+(4) real-patch AUC, raw vs thinned
    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()][:args.limit]
    cache = []
    for nm in names:
        cw, dets = build_mask(nm, cfg)
        xb = [bbox_xyxy(d) for d in dets
              if canonical_class(d["class"]) == "Wire Crossover"]
        sites = [(x, y) for x, y, deg in intersection_sites_with_degree(cw)
                 if deg >= 4]
        if sites:
            cache.append((cw, (thin(cw) * 255).astype(np.uint8), sites, xb))

    inf_density = np.array([
        (crop_site(cw, x, y, 24, 64) > 127).mean()
        for cw, _, sites, _ in cache for x, y in sites])
    print(f"[2] INFERENCE ink density @half=24: {inf_density.mean():.3f} "
          f"(vs training ~0.11 — our masks are denser)")

    print("\n[3+4] real-patch crossing-vs-junction AUC "
          "(detector-crossover=pos, other deg-4=neg):")
    half = max(4, int(round(size * cfg['nodes'].get('junction_context', 3.0) / 8)))
    for tag, use_thin in [("raw mask (current)", False),
                          ("thinned mask", True)]:
        pos, neg = [], []
        for cw, sk, sites, xb in cache:
            pr = probs(sk if use_thin else cw, sites, half)
            for (x, y), p in zip(sites, pr):
                inx = any(x1 <= x <= x2 and y1 <= y <= y2
                          for x1, y1, x2, y2 in xb)
                (pos if inx else neg).append(p)
        print(f"    {tag:22s} AUC={auc(pos, neg):.3f}  "
              f"(n_pos={len(pos)}, n_neg={len(neg)})")


if __name__ == "__main__":
    main()
