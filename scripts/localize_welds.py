#!/usr/bin/env python3
"""WHERE, physically, do two ground-truth nets get joined?

Every attempted fix so far has presumed a location and every one came back null:
notch-and-relink presumes the weld is at a detected crossover box (oracle
negative), arm cutting presumes it is at a skeleton branch (280 GT-guided cuts,
strict unchanged), and split-at-any-site presumes it is at an intersection (8 of
1245 sites accepted). Three nulls in a row is not three failed fixes, it is
evidence that the presumption is wrong -- so this stops guessing and measures it.

For each predicted node carrying two or more GT nets, take a terminal of net A
and a terminal of net B, and find the SHORTEST PATH between them through that
node's own pixels. That path is the conductor the pipeline believes exists. Then
describe it:

  where it runs        fraction of the path inside a component box, inside the
                       text mask, and out on open wire
  what it passes       whether it crosses a skeleton branch point, and how many
  how far              path length, and its length relative to the straight-line
                       distance (a long detour means the two nets are joined
                       somewhere remote, not locally)
  how thin             the minimum ink width along the path -- a genuine
                       conductor is stroke-width; a spurious join through a
                       near-touch is thinner

The distinction that matters for what to build next: a weld through a NARROW
bottleneck is a local imaging artifact and could in principle be cut. A weld that
runs the full width of a drawn rail at full stroke width is a real conductor in
the ink, and no amount of graph surgery will separate it -- the answer has to
come from somewhere other than the pixels.

Usage:
    python scripts/localize_welds.py --limit 60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.splits import add_split_arg, load_split
from schematic2netlist import skeleton as sk


def bfs_path(mask: np.ndarray, starts, goals) -> list | None:
    """Shortest 8-connected path through mask from any start to any goal."""
    H, W = mask.shape
    goal = set(goals)
    prev = {}
    dq = deque()
    for s in starts:
        if 0 <= s[0] < H and 0 <= s[1] < W and mask[s]:
            dq.append(s)
            prev[s] = None
    seen = set(prev)
    while dq:
        cur = dq.popleft()
        if cur in goal:
            out = []
            while cur is not None:
                out.append(cur)
                cur = prev[cur]
            return out[::-1]
        y, x = cur
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                n = (y + dy, x + dx)
                if (0 <= n[0] < H and 0 <= n[1] < W and mask[n]
                        and n not in seen):
                    seen.add(n)
                    prev[n] = cur
                    dq.append(n)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/weld_localization")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    idir = Path(cfg["preprocess"]["images_dir"])
    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]

    rows = []
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gp = Path(cfg["benchmark"]["gt_dir"]) / f"{stem}.json"
        dp = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
        ip = idir / nm
        if not (gp.exists() and dp.exists() and ip.exists()):
            continue
        gt = load_gt(str(gp))
        gc0 = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gc0:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            str(dp), min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(str(ip), cfg, detections=dets)
        node_map, wires = res["node_map"], res["clean_wires"]
        comps = res["components"]

        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [res["detections"][c["id"]]["x"],
                          res["detections"][c["id"]]["y"],
                          res["detections"][c["id"]]["width"],
                          res["detections"][c["id"]]["height"]]}
                for c in comps]
        p, g, _ = align_components(pred, gc0)
        pc, gcn = canonicalize_terminals(p), canonicalize_terminals(g)
        pred_of, gt_of = {}, {}
        for c in pc:
            for k, n in enumerate(c["nets"]):
                pred_of[(c["id"], k)] = n
        for c in gcn:
            for k, n in enumerate(c["nets"]):
                gt_of[(c["id"], k)] = n

        # terminal pixel positions, and which predicted node / GT net each is on
        name_to_id = {}
        for c in comps:
            for n_, nn_ in zip(c.get("nodes", []), c.get("node_names", [])):
                if n_ is not None and nn_ is not None:
                    name_to_id[nn_] = int(n_)
        term_xy = {}
        by_pid = {c["id"]: c for c in comps}
        idmap = {c["id"]: c for c in pc}
        for c in pc:
            src = by_pid.get(c["id"])
            if src is None:
                continue
            det = res["detections"][src["id"]]
            x1, y1, x2, y2 = bbox_xyxy(det)
            n = len(c["nets"])
            for k in range(n):
                term_xy[(c["id"], k)] = (int((y1 + y2) / 2),
                                         int(x1 + (k + 1) * (x2 - x1) / (n + 1)))

        on_node = defaultdict(lambda: defaultdict(list))
        for t, pn in pred_of.items():
            gn = gt_of.get(t)
            if pn is not None and gn is not None and t in term_xy:
                on_node[pn][gn].append(t)

        # component-box and skeleton context for characterising a path
        boxes = np.zeros(wires.shape, np.uint8)
        for d in res["detections"]:
            if canonical_class(d["class"]) == "Wire Crossover":
                continue
            x1, y1, x2, y2 = bbox_xyxy(d)
            boxes[max(0, y1):y2, max(0, x1):x2] = 255
        thin = sk.thin(wires)
        try:
            sites = sk.intersection_sites_with_degree(thin, min_sep=9)
            site_pts = [(int(s[1]), int(s[0])) for s in sites]
        except Exception:
            site_pts = []
        dist_in = cv2.distanceTransform((wires > 0).astype(np.uint8),
                                        cv2.DIST_L2, 3)

        for pn, nets in on_node.items():
            if len(nets) < 2:
                continue
            nid = name_to_id.get(pn)
            if nid is None:
                continue
            m = node_map == nid
            keys = sorted(nets)
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    sa = [term_xy[t] for t in nets[keys[a]] if t in term_xy]
                    sb = [term_xy[t] for t in nets[keys[b]] if t in term_xy]
                    if not sa or not sb:
                        continue
                    snap = lambda pts: [tuple(np.argwhere(m)[
                        np.argmin(((np.argwhere(m) - np.array(q)) ** 2).sum(1))])
                        for q in pts] if m.any() else []
                    try:
                        SA, SB = snap(sa), snap(sb)
                    except Exception:
                        continue
                    path = bfs_path(m, SA, SB)
                    if not path or len(path) < 3:
                        continue
                    arr = np.array(path)
                    in_box = float(boxes[arr[:, 0], arr[:, 1]].mean() / 255.0)
                    widths = dist_in[arr[:, 0], arr[:, 1]] * 2.0
                    straight = float(np.hypot(*(arr[0] - arr[-1])))
                    nsite = 0
                    if site_pts:
                        sp = np.array(site_pts)
                        for pt in arr[:: max(1, len(arr) // 60)]:
                            if np.min(((sp - pt[::-1]) ** 2).sum(1)) < 144:
                                nsite += 1
                    rows.append({
                        "image": nm, "node": pn,
                        "net_a": keys[a], "net_b": keys[b],
                        "path_len": len(path),
                        "straight": round(straight, 1),
                        "detour": round(len(path) / max(straight, 1e-6), 2),
                        "frac_in_component_box": round(in_box, 3),
                        "min_width": round(float(widths.min()), 2),
                        "median_width": round(float(np.median(widths)), 2),
                        "branch_sites_on_path": nsite,
                    })
        if i % 10 == 0:
            print(f"  [{i}/{len(names)}] weld paths={len(rows)}", flush=True)

    if not rows:
        raise SystemExit("no welded node/net pairs found")

    W = lambda k: np.array([r[k] for r in rows], dtype=float)
    print(f"\n=== WHERE {len(rows)} WELD PATHS ACTUALLY RUN ===\n")
    print(f"  {'quantity':30s} {'median':>9s} {'mean':>9s}")
    for k in ("path_len", "straight", "detour", "frac_in_component_box",
              "min_width", "median_width", "branch_sites_on_path"):
        v = W(k)
        print(f"  {k:30s} {np.median(v):9.2f} {v.mean():9.2f}")

    thinnest = W("min_width")
    med = W("median_width")
    print(f"\n  BOTTLENECK ANALYSIS — is the join thinner than a real stroke?")
    print(f"    paths whose narrowest point is < 60% of their own median width: "
          f"{int((thinnest < 0.6 * med).sum())} / {len(rows)} "
          f"({(thinnest < 0.6 * med).mean():.1%})")
    print(f"    paths at essentially full stroke width throughout (>= 80%):     "
          f"{int((thinnest >= 0.8 * med).sum())} / {len(rows)} "
          f"({(thinnest >= 0.8 * med).mean():.1%})")
    box = W("frac_in_component_box")
    print(f"\n  ROUTE")
    print(f"    paths running >20% inside a component box: "
          f"{int((box > 0.2).sum())} ({(box > 0.2).mean():.1%})")
    print(f"    paths passing no skeleton branch point:    "
          f"{int((W('branch_sites_on_path') == 0).sum())} "
          f"({(W('branch_sites_on_path') == 0).mean():.1%})")
    print(f"\n  A weld at full stroke width, on open wire, with no bottleneck is a")
    print(f"  REAL conductor in the ink. Graph surgery cannot separate it and the")
    print(f"  three null oracles are explained. A thin bottleneck would instead")
    print(f"  mean the join is an imaging artifact and IS cuttable.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "paths.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_paths": len(rows),
        **{f"median_{k}": float(np.median(W(k)))
           for k in ("path_len", "detour", "frac_in_component_box",
                     "min_width", "median_width")},
        "frac_bottlenecked": float((thinnest < 0.6 * med).mean()),
        "frac_full_width": float((thinnest >= 0.8 * med).mean()),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
