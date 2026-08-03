#!/usr/bin/env python3
"""Find wire HOPS geometrically, and self-label them from the netlist ground truth.

A hop is the draughtsman's "crossing, not connected" mark: the wire makes a short
detour -- a semicircle or a square U -- over another wire. Hand-drawn, the detour
usually touches what it arcs over, so the ink connects and the pipeline welds two
nets that the drawing says are separate. 90% of measured welds have no Wire
Crossover box within 60 px, detected OR ground-truth, so the dataset does not
annotate these and they have to be found from the geometry.

WHY NOT INTERSECTION SITES. The earlier site-split oracle proposed splits at
skeleton branch points and got almost nothing (8 accepted of 1245). The weld
localisation says why: 84.3% of weld paths pass NO branch point. A hop arcing
over a wire and grazing it does not necessarily produce a clean degree-4
junction, so a branch-point detector looks in the wrong places. This looks for
the SHAPE instead.

The detector is two geometric conditions, both cheap and both meaningful:

  detour     trace each skeleton segment as a polyline and slide a window along
             it. Arc length over chord length is ~1.0 on a straight run, pi/2 =
             1.57 for a semicircular hop and 2.0 for a square U. A local maximum
             above threshold is a candidate.
  it hops    OVER something -- the straight chord the detour departs from must
             cross ink that is not part of this segment. That is what separates
             a hop from a corner, a component lead, or a wobble, and it is the
             condition that carries almost all of the precision.

Labels come from the netlist GT, not from a human: a candidate is POSITIVE when
the predicted node it sits on carries two or more distinct GT nets -- i.e. this
is somewhere a weld actually happened -- and NEGATIVE otherwise. That makes the
whole set self-labelling on any image with verified topology.

Usage:
    python scripts/hop_candidates.py --limit 60
    python scripts/hop_candidates.py --limit 190 --out-dir results/hops
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from schematic2netlist import skeleton as sk
from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.splits import add_split_arg, load_split

NEI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbours(sk_mask, y, x):
    H, W = sk_mask.shape
    out = []
    for dy, dx in NEI:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and sk_mask[ny, nx]:
            out.append((ny, nx))
    return out


def trace_segments(sk_mask):
    """Skeleton as a list of polylines between branch/end points.

    Branch pixels (3+ neighbours) are removed first, so what remains is a set of
    simple chains; each is then walked end to end. Ordering matters because arc
    length is meaningless on an unordered pixel set.
    """
    deg = np.zeros_like(sk_mask, dtype=np.uint8)
    ys, xs = np.nonzero(sk_mask)
    for y, x in zip(ys, xs):
        deg[y, x] = len(neighbours(sk_mask, y, x))
    simple = sk_mask & (deg <= 2)
    n, lab = cv2.connectedComponents(simple.astype(np.uint8), connectivity=8)
    segs = []
    for i in range(1, n):
        pix = np.argwhere(lab == i)
        if len(pix) < 12:
            continue
        pts = {(int(a), int(b)) for a, b in pix}
        ends = [p for p in pts if len([q for q in neighbours(simple, *p)]) == 1]
        start = ends[0] if ends else next(iter(pts))
        order, seen, cur = [], {start}, start
        while cur is not None:
            order.append(cur)
            nxt = None
            for q in neighbours(simple, *cur):
                if q not in seen:
                    nxt = q
                    break
            if nxt is not None:
                seen.add(nxt)
            cur = nxt
        if len(order) >= 12:
            segs.append(np.array(order))
    return segs


def hop_candidates(wires, cfg, wins=(20, 30, 42), min_detour=1.18, min_sep=16):
    """Local detours on skeleton segments whose chord crosses OTHER ink.

    "Other ink" is measured against the SEGMENT'S OWN PIXELS, not against
    connected-component labels. Comparing labels cannot work here and the reason
    is the whole phenomenon: a hand-drawn hop grazes the wire it arcs over, so
    the two are one connected skeleton component and every label test returns
    "same". Distance to the segment's own polyline is the test that survives
    that.

    Several window lengths are scanned because a hop's size varies with how big
    the draughtsman drew it; one fixed window sees only hops of one scale.
    """
    thin = sk.thin(wires) > 0
    segs = trace_segments(thin)
    ink = wires > 0
    H, W = ink.shape
    # which SEGMENT each skeleton pixel belongs to. Connected components are too
    # coarse: a hop grazes what it arcs over, so both wires are one component and
    # every "is this foreign ink" test returns "same". Segments are split at
    # branch points, so the crossed wire really is a different segment.
    segid = np.zeros((H, W), np.int32)
    for si, s in enumerate(segs, 1):
        segid[s[:, 0], s[:, 1]] = si
    cands = []
    for sidx, seg in enumerate(segs, 1):
        for win in wins:
            if len(seg) < win + 2:
                continue
            for i in range(0, len(seg) - win, 3):
                a, b = seg[i], seg[i + win]
                chord = float(np.hypot(*(a - b)))
                if chord < 5:
                    continue
                detour = win / chord
                if detour < min_detour:
                    continue
                steps = max(int(chord), 2)
                ts = np.linspace(0, 1, steps)
                cy = np.clip((a[0] + ts * (b[0] - a[0])).round().astype(int),
                             0, H - 1)
                cx = np.clip((a[1] + ts * (b[1] - a[1])).round().astype(int),
                             0, W - 1)
                # foreign = a skeleton pixel of ANOTHER segment within 3 px of
                # the chord. This is the "it hops over something" condition.
                other = 0
                for yy, xx in zip(cy, cx):
                    y0, y1 = max(0, yy - 3), min(H, yy + 4)
                    x0, x1 = max(0, xx - 3), min(W, xx + 4)
                    patch = segid[y0:y1, x0:x1]
                    if np.any((patch > 0) & (patch != sidx)):
                        other += 1
                mid = seg[i + win // 2]
                cands.append({"y": int(mid[0]), "x": int(mid[1]),
                              "detour": round(float(detour), 3),
                              "chord": round(chord, 1),
                              "win": int(win),
                              "chord_over_other_ink": other,
                              "hops_over": int(other >= 2)})
    cands.sort(key=lambda c: (-c["hops_over"], -c["detour"]))
    kept = []
    for c in cands:
        if all((c["y"] - k["y"]) ** 2 + (c["x"] - k["x"]) ** 2 > min_sep ** 2
               for k in kept):
            kept.append(c)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--config", default=None)
    ap.add_argument("--wins", type=int, nargs="*",
                    default=[20, 30, 42])
    ap.add_argument("--min-detour", type=float, default=1.18)
    ap.add_argument("--out-dir", default="results/hops")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    idir = Path(cfg["preprocess"]["images_dir"])
    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]

    rows = []
    n_img = 0
    weld_covered = weld_total = 0
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
        node_map, comps = res["node_map"], res["components"]

        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [res["detections"][c["id"]]["x"],
                          res["detections"][c["id"]]["y"],
                          res["detections"][c["id"]]["width"],
                          res["detections"][c["id"]]["height"]]}
                for c in comps]
        p, g, _ = align_components(pred, gc0)
        pc, gcn = canonicalize_terminals(p), canonicalize_terminals(g)
        pof, gof = {}, {}
        for c in pc:
            for k, n in enumerate(c["nets"]):
                pof[(c["id"], k)] = n
        for c in gcn:
            for k, n in enumerate(c["nets"]):
                gof[(c["id"], k)] = n
        nets_on = defaultdict(set)
        for t, pn in pof.items():
            gn = gof.get(t)
            if pn is not None and gn is not None:
                nets_on[pn].add(gn)
        name_to_id = {}
        for c in comps:
            for n_, nn_ in zip(c.get("nodes", []), c.get("node_names", [])):
                if n_ is not None and nn_ is not None:
                    name_to_id[nn_] = int(n_)
        welded_ids = {name_to_id[pn] for pn, s in nets_on.items()
                      if len(s) > 1 and pn in name_to_id}
        weld_total += len(welded_ids)

        cands = hop_candidates(res["clean_wires"], cfg,
                               wins=tuple(args.wins),
                               min_detour=args.min_detour)
        hit = set()
        for c in cands:
            nid = int(node_map[c["y"], c["x"]])
            c["node"] = nid
            c["label"] = int(nid in welded_ids)
            if c["label"]:
                hit.add(nid)
            c["image"] = nm
            rows.append(c)
        weld_covered += len(hit)
        n_img += 1
        if i % 10 == 0:
            print(f"  [{i}/{len(names)}] candidates={len(rows)}", flush=True)

    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    hops = [r for r in rows if r["hops_over"]]
    hp = [r for r in hops if r["label"]]
    print(f"\n=== GEOMETRIC HOP CANDIDATES ({n_img} images) ===\n")
    print(f"  candidates {len(rows)}  ({len(rows)/max(n_img,1):.1f} per image)")
    print(f"  on a welded node (POSITIVE): {len(pos)} ({len(pos)/max(len(rows),1):.1%})")
    print(f"\n  COVERAGE — welded nodes with at least one candidate on them:")
    print(f"    {weld_covered} / {weld_total} = "
          f"{weld_covered/max(weld_total,1):.1%}")
    print(f"\n  effect of requiring the chord to cross OTHER ink:")
    print(f"    {'subset':28s} {'n':>6s} {'positives':>10s} {'precision':>10s}")
    print(f"    {'all candidates':28s} {len(rows):6d} {len(pos):10d} "
          f"{len(pos)/max(len(rows),1):10.1%}")
    print(f"    {'hops_over only':28s} {len(hops):6d} {len(hp):10d} "
          f"{len(hp)/max(len(hops),1):10.1%}")
    for th in (1.3, 1.4, 1.5, 1.7):
        s = [r for r in hops if r["detour"] >= th]
        sp = [r for r in s if r["label"]]
        print(f"    {'hops_over + detour>=%.1f' % th:28s} {len(s):6d} "
              f"{len(sp):10d} {len(sp)/max(len(s),1):10.1%}")
    print(f"\n  Coverage bounds what a perfect classifier could fix; precision")
    print(f"  says how selective it must be. The site-split oracle needs ~99.4%")
    print(f"  specificity when candidates are branch points -- if these are")
    print(f"  cleaner, that is the whole point of looking for the SHAPE.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "candidates.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": n_img, "n_candidates": len(rows), "n_positive": len(pos),
        "weld_nodes_total": weld_total, "weld_nodes_covered": weld_covered,
        "coverage": round(weld_covered / max(weld_total, 1), 4),
        "precision_all": round(len(pos) / max(len(rows), 1), 4),
        "precision_hops_over": round(len(hp) / max(len(hops), 1), 4),
        "wins": args.wins, "min_detour": args.min_detour,
    }, indent=2) + "\n")
    print(f"\nwrote {out}/candidates.csv")


if __name__ == "__main__":
    main()
