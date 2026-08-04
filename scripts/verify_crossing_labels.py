#!/usr/bin/env python3
"""Are the synthetic crossing labels actually correct, and do they land
where the pipeline looks?

The labels are generated from the ROUTED CELL GEOMETRY, but the classifier
sees RENDERED INK, and the pipeline asks its question at skeleton branch
points. Three things can therefore go wrong silently:

1. **Label wrong.** Sites are found by binning net cells into a 6-px grid
   and calling a bin with >= 2 nets a crossing. Two nets passing within
   6 px but never touching would be binned together and labelled a
   crossing even though the rendered ink shows no intersection at all.
2. **Site misplaced.** Wobble, pen lifts, hop gaps and component erasure
   all change the ink after routing, so a labelled site may sit where
   there is no longer an intersection — or the real intersection may be
   several pixels away.
3. **Distribution mismatch.** If labelled sites do not coincide with the
   sites `intersection_sites_with_degree` actually reports, the model
   trains on one population and is asked about another. That is precisely
   how the CGHD classifier failed (0.97 in-domain, 0.72 on our masks).

This script re-derives ground truth independently of the label pipeline.
For one render it keeps a per-net label map, runs the pipeline's own site
detector on the rendered ink, and for each detected site reads which nets
are actually present in the surrounding ink. Two or more distinct nets
means the site must be SPLIT; one net means UNION. That verdict is
compared against the emitted dataset label for the nearest labelled site.

Reports: label agreement, how far emitted sites sit from pipeline sites,
what fraction of pipeline sites have no emitted label at all (unlabelled
population the model will still be asked about), and the degree mix.

Usage:
    python scripts/verify_crossing_labels.py --layouts 12
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import cv2
import numpy as np

import sys
from pathlib import Path
# scripts/ is a package, so importing a sibling needs the repo root on
# sys.path — otherwise this file runs only via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.build_crossing_dataset as B  # noqa: E402  (same-dir import)
from schematic2netlist.skeleton import intersection_sites_with_degree


def net_label_map(shape, routed, comps):
    """Paint each net's routed cells with its own id (background -1)."""
    h, w = shape
    m = np.full((h, w), -1, np.int32)
    for net, cells in routed.items():
        for (y, x, s) in cells:
            cv2.rectangle(m, (x, y), (x + s - 1, y + s - 1), int(net), -1)
    for c in comps:                     # component interiors are erased
        cx, cy, bw, bh = c["bbox"]
        x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        x2, y2 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
        m[y1:y2, x1:x2] = -1
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layouts", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--radius", type=int, default=14,
                    help="ink neighbourhood read around a pipeline site")
    ap.add_argument("--match-dist", type=int, default=16,
                    help="max px between an emitted site and a pipeline site")
    ap.add_argument("--touch-rate", type=float, default=0.35)
    ap.add_argument("--merge-q", type=float, default=0.4)
    ap.add_argument("--p-dot", type=float, default=0.5)
    ap.add_argument("--p-hop", type=float, default=0.12)
    args = ap.parse_args()

    layouts = B.load_layouts(
        B.COCO, "data/transforms_1024.json", "data/splits/train.txt")
    layouts = layouts[: args.layouts]

    agree = Counter()
    dists, unlabelled_deg, labelled_deg = [], Counter(), Counter()
    n_pipe_sites = n_emitted = 0

    for li, (stem, comps) in enumerate(layouts):
        for rd in range(args.rounds):
            rng = random.Random((0, "train", stem, rd).__hash__())
            shape = (1024, 1024)
            nets = B.synth_topology(comps, rng)
            routed = B.route_nets(comps, nets, shape, rng)
            if len(routed) < 2:
                continue
            B.add_touch_contacts(routed, rng, args.touch_rate)

            # replicate the generator's merge step exactly
            from collections import defaultdict
            pos_nets = defaultdict(set)
            for net, cells in routed.items():
                for (y, x, s) in cells:
                    pos_nets[(y // 6, x // 6)].add(net)
            pairs = sorted({tuple(sorted(ns)[:2])
                            for ns in pos_nets.values() if len(ns) >= 2})
            group = {n: n for n in routed}

            def find(n):
                while group[n] != n:
                    group[n] = group[group[n]]
                    n = group[n]
                return n

            for a, b in pairs:
                if rng.random() < args.merge_q and find(a) != find(b):
                    group[find(a)] = find(b)
            merged = defaultdict(set)
            for net, cells in routed.items():
                merged[find(net)] |= cells
            merged = dict(merged)

            cross, junc = B.find_sites(merged)
            ink = B.render_ink(shape, comps, merged, cross, junc, rng,
                               p_dot=args.p_dot, p_hop=args.p_hop)
            nmap = net_label_map(shape, merged, comps)

            emitted = ([(x, y, "crossover") for (x, y) in cross]
                       + [(x, y, "junction") for (x, y) in junc])
            n_emitted += len(emitted)

            pipe = intersection_sites_with_degree((ink > 0).astype(np.uint8))
            n_pipe_sites += len(pipe)

            for (px, py, deg) in pipe:
                # truth: how many distinct nets actually appear in the ink
                # around this site?
                y0, y1 = max(0, py - args.radius), min(1024, py + args.radius + 1)
                x0, x1 = max(0, px - args.radius), min(1024, px + args.radius + 1)
                sub_ink = ink[y0:y1, x0:x1] > 0
                sub_net = nmap[y0:y1, x0:x1]
                present = set(np.unique(sub_net[sub_ink])) - {-1}
                truth = "crossover" if len(present) >= 2 else "junction"

                # nearest emitted label
                if emitted:
                    ex, ey, elab = min(
                        emitted, key=lambda e: (e[0] - px) ** 2 + (e[1] - py) ** 2)
                    d = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
                else:
                    d, elab = 1e9, None
                if d <= args.match_dist:
                    dists.append(d)
                    labelled_deg[min(deg, 5)] += 1
                    agree[f"{elab}->{truth}"] += 1
                else:
                    unlabelled_deg[min(deg, 5)] += 1
        if (li + 1) % 4 == 0:
            print(f"  [{li+1}/{len(layouts)}]", flush=True)

    matched = sum(agree.values())
    correct = agree["crossover->crossover"] + agree["junction->junction"]
    print("\n=== LABEL CORRECTNESS (emitted label vs independently re-derived "
          "truth from the rendered ink) ===")
    for k in sorted(agree):
        print(f"  {k:26s} {agree[k]:6d}")
    if matched:
        print(f"\n  agreement: {correct}/{matched} = {correct/matched:.3f}")
    print(f"\n=== SITE ALIGNMENT ===")
    print(f"  pipeline sites found on renders : {n_pipe_sites}")
    print(f"  emitted labelled sites          : {n_emitted}")
    print(f"  pipeline sites WITH a label     : {matched} "
          f"({matched/max(n_pipe_sites,1):.1%})")
    print(f"  pipeline sites WITHOUT a label  : {sum(unlabelled_deg.values())} "
          f"({sum(unlabelled_deg.values())/max(n_pipe_sites,1):.1%})")
    if dists:
        print(f"  emitted-to-pipeline distance    : "
              f"mean {np.mean(dists):.1f}px  median {np.median(dists):.1f}px")
    print(f"  degree mix, labelled            : {dict(sorted(labelled_deg.items()))}")
    print(f"  degree mix, UNlabelled          : {dict(sorted(unlabelled_deg.items()))}")
    print("\nThe unlabelled fraction is the population the model is asked")
    print("about at inference but never trained on.")


if __name__ == "__main__":
    main()
