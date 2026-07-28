#!/usr/bin/env python3
"""Does gap-tolerant path tracing beat connected components? (tier-4 probe)

The oracle attributes the bulk of remaining end-to-end error to wire
connectivity, so the next build target is replacing global connected
components with pin-anchored path tracing: two boundary-crossing sites
belong to the same net if a cheap path runs between them through ink,
where crossing a small gap is expensive but possible.

Before building that, this script MEASURES whether the idea can work,
because the failure mode is symmetric and easy to miss: a cost model
loose enough to reconnect genuinely-broken rails is also loose enough
to short two nets that merely pass close together. Both are measured
here against verified GT.

For each (gap cost, cost threshold) it reports, over anchor pairs that
connected components places in DIFFERENT components — i.e. exactly the
pairs tracing would newly merge:

    merges_correct    pairs on the same GT net      (tracing wins)
    merges_wrong      pairs on different GT nets    (tracing shorts)
    precision         correct / (correct + wrong)

A configuration is only worth building if precision stays high while
correct merges are non-trivial. Anything else means CC is not the
bottleneck the oracle's aggregate suggests, and the effort belongs
elsewhere.

Usage:
    python scripts/explore_path_tracing.py --limit 40
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import load_gt
from schematic2netlist.nodes import bbox_xyxy, build_wire_nodes
from schematic2netlist.snapping import _boundary_run_sites
from schematic2netlist.textmask import detect_text_mask
from schematic2netlist.wires import (
    build_non_wire_mask,
    extract_wires,
    stitch_wire_islands,
    stitchable_mask,
)


def grid_graph(ink: np.ndarray, gap_cost: float) -> sp.csr_matrix:
    """4-connected pixel graph; edge weight is the mean endpoint cost,
    so travelling through ink is cheap and crossing a gap is not."""
    H, W = ink.shape
    node_cost = np.where(ink, 1.0, gap_cost).astype(np.float32)
    idx = np.arange(H * W).reshape(H, W)
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = idx[:H - dy, :W - dx].ravel()
        b = idx[dy:, dx:].ravel()
        w = 0.5 * (node_cost[:H - dy, :W - dx].ravel()
                   + node_cost[dy:, dx:].ravel())
        rows.append(a)
        cols.append(b)
        vals.append(w)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    return sp.coo_matrix(
        (np.concatenate([v, v]),
         (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(H * W, H * W),
    ).tocsr()


def anchors_for(dets, node_map, cfg) -> list[dict]:
    """Boundary-crossing sites: (component, cc id, pixel). These lie on
    ink by construction, which is what makes them usable path sources."""
    out = []
    step = cfg["snapping"]["expand_step"]
    max_expand = cfg["snapping"]["max_expand"]
    for i, d in enumerate(dets):
        if canonical_class(d["class"]) == "Wire Crossover":
            continue
        x1, y1, x2, y2 = bbox_xyxy(d)
        sites = []
        for r in range(step, max_expand + 1, step):
            sites = _boundary_run_sites(node_map, x1 - r, y1 - r, x2 + r, y2 + r)
            if sites:
                break
        for nid, px, py in sites:
            out.append({"comp": i, "cc": nid, "x": int(px), "y": int(py)})
    return out


def gt_net_of(gt: dict, comp_idx: int, dets: list, anchor_xy) -> str | None:
    """Best-effort GT net for an anchor: the net of the GT component
    whose box best overlaps this detection, at the terminal nearest the
    anchor. Returns None when no GT component corresponds."""
    d = dets[comp_idx]
    best, best_iou = None, 0.0
    dx1, dy1, dx2, dy2 = bbox_xyxy(d)
    for c in gt["components"]:
        cx, cy, w, h = c["bbox"]
        gx1, gy1, gx2, gy2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        ix = max(0.0, min(dx2, gx2) - max(dx1, gx1))
        iy = max(0.0, min(dy2, gy2) - max(dy1, gy1))
        inter = ix * iy
        union = (dx2 - dx1) * (dy2 - dy1) + w * h - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = c, iou
    if best is None or best_iou < 0.1:
        return None
    # nearest terminal by angle from the component centre
    cx, cy, w, h = best["bbox"]
    ax, ay = anchor_xy
    horiz = w >= h
    n_t = len(best["terminals"])
    if n_t == 0:
        return None
    if n_t == 1:
        return best["terminals"][0]["net"]
    first = (ax < cx) if horiz else (ay < cy)
    term = best["terminals"][0] if first else best["terminals"][1]
    return term["net"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--gt-dir", default=None,
                    help="overrides benchmark.gt_dir from the config")
    ap.add_argument("--out-dir", default="results/path_tracing_probe")
    ap.add_argument("--config", default=None)
    ap.add_argument("--gap-costs", default="4,6,8,10")
    ap.add_argument("--thresholds", default="60,120,200,320")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_dir = Path(args.gt_dir or cfg["benchmark"]["gt_dir"])
    gap_costs = [float(g) for g in args.gap_costs.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()][: args.limit]

    tally = {(g, t): {"correct": 0, "wrong": 0, "unknown": 0}
             for g in gap_costs for t in thresholds}
    n_images = 0

    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gt_path = gt_dir / f"{stem}.json"
        det_path = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
        if not gt_path.exists() or not det_path.exists():
            continue
        gt = load_gt(gt_path)
        if not gt.get("verified"):
            continue

        img = cv2.imread(str(Path(args.images_dir) / nm))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = load_cached_detections(det_path)
        text_mask = detect_text_mask(gray, cfg) if cfg["textmask"]["enabled"] else None
        nwm = build_non_wire_mask(gray, dets, cfg, text_mask)
        _cand, wires = extract_wires(gray, nwm, cfg)
        if cfg["wires"].get("stitch_masked_gaps"):
            wires = stitch_wire_islands(
                wires, stitchable_mask(gray.shape, dets, cfg, text_mask), cfg
            )
        node_map, _n = build_wire_nodes(wires, connectivity=8)
        anchors = anchors_for(dets, node_map, cfg)
        if len(anchors) < 2:
            continue
        n_images += 1
        print(f"[{i}/{len(names)}] {nm} anchors={len(anchors)}", flush=True)

        H, W = wires.shape
        ink = wires > 0
        src = [a["y"] * W + a["x"] for a in anchors]
        nets = [gt_net_of(gt, a["comp"], dets, (a["x"], a["y"])) for a in anchors]

        for g in gap_costs:
            D = dijkstra(grid_graph(ink, g), indices=src, limit=max(thresholds))
            sub = D[:, src]
            for k in range(len(anchors)):
                for m in range(k + 1, len(anchors)):
                    if anchors[k]["cc"] == anchors[m]["cc"]:
                        continue          # CC already joins these
                    cost = sub[k, m]
                    if not np.isfinite(cost):
                        continue
                    for t in thresholds:
                        if cost > t:
                            continue
                        a_net, b_net = nets[k], nets[m]
                        key = ("unknown" if a_net is None or b_net is None
                               else "correct" if a_net == b_net else "wrong")
                        tally[(g, t)][key] += 1

    rows = []
    for (g, t), c in sorted(tally.items()):
        decided = c["correct"] + c["wrong"]
        rows.append({
            "gap_cost": g, "threshold": t,
            "merges_correct": c["correct"], "merges_wrong": c["wrong"],
            "merges_unknown_gt": c["unknown"],
            "precision": round(c["correct"] / decided, 4) if decided else None,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"n_images": n_images, "sweep": rows}, indent=2) + "\n"
    )

    print(f"\nnew merges beyond connected components ({n_images} images)")
    print(f"  {'gap':>5s} {'thresh':>7s} {'correct':>8s} {'wrong':>7s} "
          f"{'unknown':>8s} {'precision':>10s}")
    for r in rows:
        p = "n/a" if r["precision"] is None else f"{r['precision']:.3f}"
        print(f"  {r['gap_cost']:5.1f} {r['threshold']:7.0f} "
              f"{r['merges_correct']:8d} {r['merges_wrong']:7d} "
              f"{r['merges_unknown_gt']:8d} {p:>10s}")
    print(f"\nwrote {out_dir}/sweep.csv")


if __name__ == "__main__":
    main()
