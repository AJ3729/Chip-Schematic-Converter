#!/usr/bin/env python3
"""Candidates and labelled crops for a hop detector, working on RAW INK.

The geometric attempt failed for one reason: a hand-drawn hop grazes the wire it
arcs over, that contact makes a branch point, and skeletonisation splits segments
at branch points -- so the bump is destroyed by the very step used to find it.
Everything here therefore stays on the ink.

CANDIDATES come from the distance transform of the ink. Where two strokes cross,
their overlap is locally thicker: a perpendicular crossing of a stroke of
half-width h measures about h*sqrt(2). So local maxima of the distance transform
above a multiple of the measured half-width find crossings, junctions and hops
without thinning anything. Stroke half-width is measured per image from the
distance transform's own mode, so this does not depend on a fixed pixel size.

LABELS are causal and local. A candidate is POSITIVE when it lies close to a WELD
PATH -- the shortest route through a predicted node between terminals that ground
truth assigns to different nets. That is much tighter than "somewhere on a welded
node", which is what the previous attempt used: a welded node can span the page
while the actual join is a few dozen pixels, so the loose label buries the signal
in negatives that look nothing like hops.

Everything is self-labelled from the verified netlists; no hand annotation.

Usage:
    python scripts/build_hop_dataset.py --limit 190 --out data/hops
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from localize_welds import bfs_path

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.splits import add_split_arg, load_split


def stroke_half_width(dt: np.ndarray) -> float:
    """Modal distance-transform value over ink -- the stroke's half-width."""
    v = dt[dt > 0.5]
    if v.size == 0:
        return 1.5
    hist, edges = np.histogram(v, bins=40, range=(0.5, 8.0))
    return float(edges[int(hist.argmax())] + (edges[1] - edges[0]) / 2)


def ink_candidates(wires: np.ndarray, thick_mult: float = 1.28,
                   min_sep: int = 11, max_per_image: int = 4000):
    """Local maxima of the distance transform: crossings, junctions and hops.

    No thinning anywhere. A perpendicular crossing of a stroke of half-width h
    measures h*sqrt(2) = 1.41h, so a threshold a little below that catches
    crossings while rejecting plain wire.
    """
    ink = (wires > 0).astype(np.uint8)
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    hw = stroke_half_width(dt)
    thr = thick_mult * hw
    k = np.ones((min_sep, min_sep), np.uint8)
    localmax = (dt >= cv2.dilate(dt, k) - 1e-6) & (dt >= thr)
    ys, xs = np.nonzero(localmax)
    if len(ys) == 0:
        return [], hw
    order = np.argsort(-dt[ys, xs])
    kept = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all((y - a) ** 2 + (x - b) ** 2 > min_sep ** 2 for a, b, _ in kept):
            kept.append((y, x, float(dt[y, x])))
        if len(kept) >= max_per_image:
            break
    return kept, hw


def weld_paths(res, gcomps):
    """Shortest routes joining terminals that GT says are on different nets."""
    node_map, comps = res["node_map"], res["components"]
    pred = [{"id": c["id"], "class": c["class"],
             "nets": list(c.get("node_names", [])),
             "bbox": [res["detections"][c["id"]]["x"],
                      res["detections"][c["id"]]["y"],
                      res["detections"][c["id"]]["width"],
                      res["detections"][c["id"]]["height"]]}
            for c in comps]
    p, g, _ = align_components(pred, gcomps)
    pc, gcn = canonicalize_terminals(p), canonicalize_terminals(g)
    pof, gof = {}, {}
    for c in pc:
        for k, n in enumerate(c["nets"]):
            pof[(c["id"], k)] = n
    for c in gcn:
        for k, n in enumerate(c["nets"]):
            gof[(c["id"], k)] = n
    name_to_id = {}
    for c in comps:
        for n_, nn_ in zip(c.get("nodes", []), c.get("node_names", [])):
            if n_ is not None and nn_ is not None:
                name_to_id[nn_] = int(n_)
    byp = {c["id"]: c for c in comps}
    txy = {}
    for c in pc:
        s = byp.get(c["id"])
        if s is None:
            continue
        x1, y1, x2, y2 = bbox_xyxy(res["detections"][s["id"]])
        n = len(c["nets"])
        for k in range(n):
            txy[(c["id"], k)] = (int((y1 + y2) / 2),
                                 int(x1 + (k + 1) * (x2 - x1) / (n + 1)))
    onn = defaultdict(lambda: defaultdict(list))
    for t, pn in pof.items():
        gn = gof.get(t)
        if pn is not None and gn is not None and t in txy:
            onn[pn][gn].append(t)
    paths = []
    for pn, nets in onn.items():
        if len(nets) < 2:
            continue
        nid = name_to_id.get(pn)
        if nid is None:
            continue
        m = node_map == nid
        if not m.any():
            continue
        pts = np.argwhere(m)
        snap = lambda q: tuple(pts[np.argmin(((pts - np.array(q)) ** 2).sum(1))])
        keys = sorted(nets)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                SA = [snap(txy[t]) for t in nets[keys[a]] if t in txy]
                SB = [snap(txy[t]) for t in nets[keys[b]] if t in txy]
                if not SA or not SB:
                    continue
                pth = bfs_path(m, SA, SB)
                if pth and len(pth) >= 3:
                    paths.append(np.array(pth))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=190)
    ap.add_argument("--config", default=None)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--pos-radius", type=float, default=14.0,
                    help="px from a weld path for a candidate to count POSITIVE")
    ap.add_argument("--thick-mult", type=float, default=1.28)
    ap.add_argument("--out", default="data/hops")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    idir = Path(cfg["preprocess"]["images_dir"])
    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]

    X, Y, G, META = [], [], [], []
    n_img = 0
    covered = total_paths = 0
    half = args.crop // 2
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
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)

        cands, hw = ink_candidates(wires, thick_mult=args.thick_mult)
        paths = weld_paths(res, gcomps)
        total_paths += len(paths)

        # distance from every pixel to the nearest weld path
        pmask = np.zeros(wires.shape, np.uint8)
        for pth in paths:
            pmask[pth[:, 0], pth[:, 1]] = 255
        dist_to_path = (cv2.distanceTransform(255 - pmask, cv2.DIST_L2, 3)
                        if pmask.any() else
                        np.full(wires.shape, 1e6, np.float32))
        hit = set()
        for (y, x, d) in cands:
            lab = int(dist_to_path[y, x] <= args.pos_radius)
            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half
            if y0 < 0 or x0 < 0 or y1 > wires.shape[0] or x1 > wires.shape[1]:
                continue
            crop = np.stack([gray[y0:y1, x0:x1],
                             (wires[y0:y1, x0:x1] > 0).astype(np.uint8) * 255],
                            axis=0)
            X.append(crop.astype(np.uint8))
            Y.append(lab)
            G.append(n_img)
            META.append((nm, y, x, round(d, 2), round(hw, 2)))
            if lab:
                for pi, pth in enumerate(paths):
                    if np.min(np.hypot(pth[:, 0] - y, pth[:, 1] - x)) <= args.pos_radius:
                        hit.add(pi)
        covered += len(hit)
        n_img += 1
        if i % 20 == 0:
            print(f"  [{i}/{len(names)}] crops={len(X)} pos={sum(Y)}", flush=True)

    X = np.array(X, np.uint8)
    Y = np.array(Y, np.int64)
    G = np.array(G, np.int64)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "hops.npz", X=X, y=Y, g=G)
    (out / "meta.json").write_text(json.dumps({
        "n_images": n_img, "n_crops": int(len(X)), "n_positive": int(Y.sum()),
        "pos_rate": float(Y.mean()) if len(Y) else 0.0,
        "weld_paths": total_paths, "weld_paths_covered": covered,
        "coverage": round(covered / max(total_paths, 1), 4),
        "crop": args.crop, "pos_radius": args.pos_radius,
        "thick_mult": args.thick_mult,
    }, indent=2) + "\n")

    print(f"\n=== RAW-INK HOP DATASET ({n_img} images) ===\n")
    print(f"  crops {len(X)}  ({len(X)/max(n_img,1):.0f} per image)")
    print(f"  positive {int(Y.sum())} ({Y.mean():.2%})")
    print(f"\n  COVERAGE — weld paths with a candidate within "
          f"{args.pos_radius:.0f} px:")
    print(f"    {covered} / {total_paths} = {covered/max(total_paths,1):.1%}")
    print(f"\n  Coverage is the gate. The geometric attempt reached 37.1% and no")
    print(f"  classifier over it could have helped; if this is not clearly")
    print(f"  higher, training is not worth doing either.")
    print(f"\nwrote {out}/hops.npz")


if __name__ == "__main__":
    main()
