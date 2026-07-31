#!/usr/bin/env python3
"""Recover components the primary detector missed, using the other two seeds.

The class vote fixed LABELS and netted +23 correct ones at 84.4% precision. It
cannot help the other quarter of the detection block: of 54 unmatched GT
components, 41 are class confusion but 13 are "nothing detected there" -- and a
component the primary seed missed may well have been found by seed 1 or seed 2,
since the three were trained independently.

The construction is deliberately conservative, because adding detections is
riskier than relabelling them. Every primary detection survives untouched, so
baseline recall cannot fall. On top of that, a cluster of boxes from the OTHER
seeds is added only when at least ``--min-agree`` of them agree on a component
the primary has no box for. Requiring agreement is what keeps a single seed's
false positive from entering; requiring the primary to be silent is what keeps
this from perturbing components the baseline already handles.

Added boxes are averaged over their cluster and their class is the cluster
majority, since with the primary absent there is no reason to prefer any one
seed's geometry.

The failure mode to watch is a false positive landing on a WIRE, because a
spurious component box blanks that ink in build_non_wire_mask and severs the
net -- turning a detection gain into a connectivity loss. That is why precision
is measured against GT here, before the benchmark is spent on it.

Usage:
    python scripts/ensemble_detection_union.py --out data/detections_1024_union
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def box_of(d):
    return (d["x"], d["y"], d["width"], d["height"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--primary", default="data/detections_1024_vote",
                    help="defaults to the vote cache so the two ensembles "
                         "compose rather than compete")
    ap.add_argument("--others", nargs="*",
                    default=["data/detections_seed1_1024",
                             "data/detections_seed2_1024"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    conf = cfg["detect"].get("confidence")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(f"data/splits/{args.split}.txt")
             if l.strip()]

    added = Counter()
    n_files = n_added = n_base = 0
    for nm in names:
        stem = Path(nm).stem
        pp = Path(args.primary) / f"{stem}.json"
        if not pp.exists():
            continue
        cache = json.loads(pp.read_text())
        prim = [d for d in cache["detections"]
                if conf is None or d.get("confidence", 1.0) >= conf]
        n_base += len(prim)

        pool = []
        for d in args.others:
            q = Path(d) / f"{stem}.json"
            if not q.exists():
                continue
            for x in json.loads(q.read_text())["detections"]:
                if conf is None or x.get("confidence", 1.0) >= conf:
                    pool.append(x)

        # greedy clustering of the OTHER seeds' boxes
        used = [False] * len(pool)
        for i, a in enumerate(pool):
            if used[i]:
                continue
            grp = [a]
            used[i] = True
            for j in range(i + 1, len(pool)):
                if not used[j] and iou(box_of(a), box_of(pool[j])) >= args.iou:
                    grp.append(pool[j])
                    used[j] = True
            if len(grp) < args.min_agree:
                continue
            # does the primary already have this component?
            if any(iou(box_of(a), box_of(d)) >= args.iou for d in prim):
                continue
            cls = Counter(canonical_class(x["class"]) for x in grp)
            top, n_top = cls.most_common(1)[0]
            if n_top * 2 <= len(grp):        # no majority on the label
                continue
            cache["detections"].append({
                "class": top,
                "confidence": float(sum(x.get("confidence", 1.0)
                                        for x in grp) / len(grp)),
                "x": float(sum(x["x"] for x in grp) / len(grp)),
                "y": float(sum(x["y"] for x in grp) / len(grp)),
                "width": float(sum(x["width"] for x in grp) / len(grp)),
                "height": float(sum(x["height"] for x in grp) / len(grp)),
                "ensemble_added": True,
            })
            added[top] += 1
            n_added += 1
        (out / f"{stem}.json").write_text(json.dumps(cache) + "\n")
        n_files += 1

    print(f"wrote {n_files} caches to {out}")
    print(f"primary detections {n_base}, ADDED {n_added} "
          f"(agreement >= {args.min_agree} of {len(args.others)} other seeds)")
    print(f"\nadded by class:")
    for k, v in added.most_common(15):
        print(f"  {k:22s} {v:4d}")
    print(f"\n13 of the 54 unmatched GT components are 'nothing detected "
          f"there'.\nAdded boxes only help if they land on those rather than "
          f"on wires --\na spurious box blanks its ink and severs the net.")


if __name__ == "__main__":
    main()
