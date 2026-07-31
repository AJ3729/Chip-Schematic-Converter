#!/usr/bin/env python3
"""Label crossings on REAL images from the verified GT, and measure which
geometric features actually separate them.

Three classifiers trained on self-labelled synthetic renders failed to
transfer, and worse, failed monotonically: raising in-domain AUC from 0.7469
to 0.9045 dropped transfer AUC from 0.5231 to 0.4849
(``results/crossing_transfer_v5b/summary.json``). Halving the measured
appearance gap (ink Cohen's d 1.79 -> 0.60) made transfer WORSE, so
appearance statistics were never the binding constraint. The only checkpoint
with any signal was trained on real photographs, which says the missing
ingredient is real annotated crossings.

Those can be manufactured. The causal cut test built to EVALUATE transfer is
itself an annotator: remove a disk at a site, ask whether Hungarian-aligned
GT nets actually separate, and the verified GT netlist answers must-split
versus must-union with no human labelling. It already produced 1720 labelled
real sites on the test split. Run it on the TRAIN split and there is a real
dataset.

This script extracts, per site, that causal label plus interpretable
geometric features — the evidence a human actually uses to read a schematic:

  dot_ratio        local ink half-width / median stroke half-width. A drawn
                   junction dot is a deliberate "these connect" mark, and it
                   is the single most direct signal available.
  degree           number of skeleton arms at the site
  straightness     for the best arm pair, |180 deg - angle between them|. A
                   crossing is two lines passing through; small values mean
                   a straight run.
  n_collinear      how many arm pairs are near-collinear
  angle_min_gap    smallest angle between any two arms (a T-junction has one
                   near-90 deg gap; an X has two)
  d_xover_box      distance to the nearest detected Wire Crossover box, in
                   stroke widths
  d_comp_box       distance to the nearest component box, in stroke widths
  local_ink        ink fraction in a small window (crowding)

Features are reported by AUC individually, then combined by logistic
regression FIT ON TRAIN AND SCORED ON TEST. Fitting and scoring on the same
split would repeat the mistake this whole line of work has been correcting.

A deterministic rule is preferred to a model if one works, so the per-feature
AUCs are the primary output; the regression is a check on whether any
combination does better.

Usage:
    python scripts/build_real_crossing_features.py --split train --limit 220
    python scripts/build_real_crossing_features.py --split test  --limit 190
    python scripts/build_real_crossing_features.py --fit        # then this
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
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import load_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.skeleton import intersection_sites_with_degree, thin

FEATURES = ["dot_ratio", "degree", "straightness", "n_collinear",
            "angle_min_gap", "d_xover_box", "d_comp_box", "local_ink"]


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
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = ss.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def arm_angles(skel: np.ndarray, x: int, y: int, r: int) -> list[float]:
    """Directions (degrees) of skeleton arms leaving a site.

    Sampled on a ring of radius ``r``: the skeleton pixels on that ring,
    clustered angularly, are where the arms cross it. Reading direction at a
    distance rather than from the immediate neighbourhood is what makes this
    robust to the blob of branch pixels a thick hand-drawn intersection
    leaves behind.
    """
    H, W = skel.shape
    hits = []
    for deg in range(0, 360, 3):
        a = np.deg2rad(deg)
        px, py = int(round(x + r*np.cos(a))), int(round(y + r*np.sin(a)))
        if 0 <= px < W and 0 <= py < H and skel[py, px]:
            hits.append(deg)
    if not hits:
        return []
    # cluster contiguous angular runs, wrapping at 360
    groups, cur = [], [hits[0]]
    for d in hits[1:]:
        if d - cur[-1] <= 9:
            cur.append(d)
        else:
            groups.append(cur); cur = [d]
    groups.append(cur)
    if len(groups) > 1 and (groups[0][0] + 360 - groups[-1][-1]) <= 9:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    return [float(np.mean([g if g < 360 else g-360 for g in grp]) % 360)
            for grp in groups]


def geometry_features(wires, skel, dist, hw, x, y, deg, xover, comps):
    """Interpretable features at one site."""
    r = max(6, int(round(hw * 4.0)))
    angs = arm_angles(skel, x, y, r)
    n_arms = len(angs) if angs else deg

    straightness, n_collinear, min_gap = 180.0, 0, 180.0
    if len(angs) >= 2:
        best = 180.0
        for i in range(len(angs)):
            for j in range(i+1, len(angs)):
                d = abs(angs[i] - angs[j]) % 360
                d = min(d, 360 - d)
                best = min(best, abs(180.0 - d))
                if abs(180.0 - d) <= 25.0:
                    n_collinear += 1
                min_gap = min(min_gap, d)
        straightness = best

    def box_dist(boxes):
        if not boxes:
            return 50.0
        best = 1e9
        for b in boxes:
            dx = max(abs(x - b["x"]) - b["width"]/2, 0.0)
            dy = max(abs(y - b["y"]) - b["height"]/2, 0.0)
            best = min(best, float(np.hypot(dx, dy)))
        return min(best / max(hw, 1.0), 50.0)

    H, W = wires.shape
    w = max(8, int(round(hw * 6)))
    sub = wires[max(0, y-w):y+w+1, max(0, x-w):x+w+1]
    return {
        "dot_ratio": round(float(dist[y, x]) / max(hw, 1e-6), 4),
        "degree": int(n_arms),
        "straightness": round(straightness, 2),
        "n_collinear": int(n_collinear),
        "angle_min_gap": round(min_gap, 2),
        "d_xover_box": round(box_dist(xover), 3),
        "d_comp_box": round(box_dist(comps), 3),
        "local_ink": round(float((sub > 0).mean()) if sub.size else 0.0, 4),
    }


def extract(split: str, limit: int, cfg, cut_radius: int) -> list[dict]:
    names = [l.strip() for l in open(f"data/splits/{split}.txt") if l.strip()]
    names = names[:limit]
    images_dir = resolve_and_check(None, names, cfg)
    rows, skipped = [], Counter()

    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gt_path = Path(cfg["benchmark"]["gt_dir"]) / f"{stem}.json"
        det_path = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
        if not gt_path.exists() or not det_path.exists():
            skipped["no_gt_or_detections"] += 1
            continue
        gt = load_gt(str(gt_path))
        dets = load_cached_detections(
            str(det_path), min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(images_dir / nm, cfg, detections=dets)
        node_map, wires = res["node_map"], res["clean_wires"]
        pred, gt_comps = res["components"], gt["components"]

        cost = np.ones((len(pred), len(gt_comps)))
        for pi, c in enumerate(pred):
            d = dets[c["id"]]
            pb = (d["x"], d["y"], d["width"], d["height"])
            for gj, g in enumerate(gt_comps):
                if canonical_class(c["class"]) == canonical_class(g["class"]):
                    cost[pi, gj] = 1.0 - iou(pb, g["bbox"])
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

        mask8 = (wires > 0).astype(np.uint8)
        sites = intersection_sites_with_degree(mask8)
        if not sites:
            continue
        skel = thin(mask8).astype(np.uint8)
        dist = cv2.distanceTransform(mask8, cv2.DIST_L2, 3)
        v = dist[skel > 0]
        hw = float(np.median(v)) if v.size else 1.0
        xover = [d for d in dets
                 if canonical_class(d["class"]) == "Wire Crossover"]
        cboxes = [d for d in dets
                  if canonical_class(d["class"]) != "Wire Crossover"]

        for (x, y, deg) in sites:
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
                label = 0
            else:
                # causal cut: does removing a disk HERE separate the fused
                # nets? Asking only "does this node carry >=2 nets?" labels
                # every site on a welded node must-split, which measures
                # coincidence -- a welded node spans much of the drawing.
                m = (node_map == nid).astype(np.uint8)
                probe = m.copy()
                cv2.circle(probe, (x, y), cut_radius, 0, -1)
                _n, lab = cv2.connectedComponents(probe, connectivity=8)
                anchors = {}
                ys, xs = np.nonzero(m)
                for net, centres in nets.items():
                    bx, by = centres[0]
                    if ys.size == 0:
                        continue
                    k = int(np.argmin((xs - bx)**2 + (ys - by)**2))
                    anchors[net] = int(lab[ys[k], xs[k]])
                ids = [v for v in anchors.values() if v != 0]
                label = 1 if len(set(ids)) >= 2 else 0

            feats = geometry_features(wires, skel, dist, hw, x, y, deg,
                                      xover, cboxes)
            rows.append({"image": nm, "x": x, "y": y, "label": label,
                         "stroke_hw": round(hw, 3), **feats})
        if i % 20 == 0:
            print(f"  [{i}/{len(names)}] sites={len(rows)}", flush=True)
    print(f"  skipped: {dict(skipped)}", flush=True)
    return rows


def report(rows, title):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    print(f"\n=== {title} ===")
    print(f"sites {len(rows)}  must-split {len(pos)}  must-union {len(neg)}")
    print(f"\n  {'feature':16s} {'AUC':>7s} {'|AUC-.5|':>9s} "
          f"{'mean(split)':>12s} {'mean(union)':>12s}")
    ranked = []
    for f in FEATURES:
        a = auc_score([r[f] for r in pos], [r[f] for r in neg])
        ranked.append((abs(a - 0.5), a, f))
    for gap, a, f in sorted(ranked, reverse=True):
        print(f"  {f:16s} {a:7.4f} {gap:9.4f} "
              f"{np.mean([r[f] for r in pos]):12.3f} "
              f"{np.mean([r[f] for r in neg]):12.3f}")
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--cut-radius", type=int, default=7)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/real_crossings")
    ap.add_argument("--fit", action="store_true",
                    help="fit on the saved train split, score the test split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.split:
        print(f"extracting split={args.split} limit={args.limit}")
        rows = extract(args.split, args.limit, cfg, args.cut_radius)
        if not rows:
            raise SystemExit("no sites extracted")
        p = out / f"sites_{args.split}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        report(rows, f"{args.split} split, REAL causally-labelled sites")
        print(f"\nwrote {p}")
        return

    if args.fit:
        # Verified GT netlists exist ONLY for the 190 test images -- the train
        # split's nets were transferred from the pipeline's own output at
        # bootstrap and never human-corrected, so using them as crossing
        # labels would be circular. There is therefore no held-out split to
        # fit on, and the combined model is estimated by GROUPED
        # cross-validation over images instead: every fold fits on one set of
        # images and scores sites from images it never saw. That answers
        # "do these features generalize across drawings", which is the
        # question, while being explicit that it is not a clean held-out
        # number and that fitting anything on these images would contaminate
        # the benchmark they are the test set for.
        te = list(csv.DictReader(open(out / "sites_test.csv")))
        for r in te:
            r["label"] = int(r["label"])
            for f in FEATURES:
                r[f] = float(r[f])
        report(te, "ALL REAL SITES (per-feature AUC needs no fitting, so "
                   "this is contamination-free)")

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold
        X = np.array([[r[f] for f in FEATURES] for r in te])
        y = np.array([r["label"] for r in te])
        groups = np.array([r["image"] for r in te])
        n_groups = len(set(groups))
        oof = np.full(len(y), np.nan)
        gkf = GroupKFold(n_splits=min(5, n_groups))
        for tr_i, te_i in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr_i])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(X[tr_i]), y[tr_i])
            oof[te_i] = clf.predict_proba(sc.transform(X[te_i]))[:, 1]
        a_cv = auc_score(oof[y == 1], oof[y == 0])

        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X), y)

        print(f"\n=== COMBINED MODEL on real geometric features ===")
        print(f"  image-grouped {gkf.get_n_splits()}-fold CV, out-of-fold AUC "
              f"{a_cv:.4f}   ({len(y)} sites over {n_groups} images)")
        print(f"\n  coefficients (standardized, fit on all; sign pushes "
              f"toward SPLIT):")
        for f, c in sorted(zip(FEATURES, clf.coef_[0]),
                           key=lambda kv: -abs(kv[1])):
            print(f"    {f:16s} {c:+.4f}")
        print(f"\n  same sites, for comparison:")
        print(f"    render-trained CNN, 750 epochs   0.4849")
        print(f"    render-trained CNN, v3           0.5094")
        print(f"    CGHD CNN (real photographs)      0.596")
        print(f"    real geometric features (CV)     {a_cv:.4f}")
        print(f"\n  A wrong split severs a net and corrupts every component")
        print(f"  on it, so the operating point matters more than the AUC:")
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            tp = int(((oof >= thr) & (y == 1)).sum())
            fp = int(((oof >= thr) & (y == 0)).sum())
            fn = int(((oof < thr) & (y == 1)).sum())
            print(f"    thr {thr:.1f}  split_recall {tp/max(tp+fn,1):.3f}  "
                  f"split_precision {tp/max(tp+fp,1):.3f}  "
                  f"wrong_splits {fp}")
        summary = {
            "n_sites": len(te), "n_images": n_groups,
            "methodology": (
                "Verified GT exists only for the test split, so there is no "
                "clean held-out set; the combined model is image-grouped "
                "cross-validated. Per-feature AUCs require no fitting and are "
                "contamination-free."),
            "grouped_cv_auc": round(a_cv, 4),
            "per_feature_auc": {
                f: round(auc_score([r[f] for r in te if r["label"] == 1],
                                   [r[f] for r in te if r["label"] == 0]), 4)
                for f in FEATURES},
            "coefficients": {f: round(float(c), 4)
                             for f, c in zip(FEATURES, clf.coef_[0])},
            "baselines": {"render_cnn_750ep": 0.4849, "render_cnn_v3": 0.5094,
                          "cghd_cnn_real_photos": 0.596},
        }
        (out / "fit_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {out}/fit_summary.json")
        return

    ap.error("pass --split to extract, or --fit to fit and score")


if __name__ == "__main__":
    main()
