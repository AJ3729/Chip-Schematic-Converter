#!/usr/bin/env python3
"""Draw the verified GT netlist on top of the ORIGINAL photograph.

Adjudicating a weld needs two things side by side: what the drafter drew, and
what the annotation claims about it. Every other view in this repo shows one or
the other. This shows both — each GT component boxed on the original
photograph, labelled with its class and the net of each terminal, colour-coded
by net so a shared net is visible at a glance.

That is what makes a GT ERROR verdict possible: if two boxes the drawing
plainly joins with unbroken wire carry different net labels, the annotation is
wrong, and no amount of pixel work will recover it.

Usage:
    python scripts/render_gt_overlay.py --images circuit_1134.jpg,circuit_1256.jpg
    python scripts/render_gt_overlay.py --from-welds   # every image in the review
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.config import load_config
from schematic2netlist.gt import load_gt
from schematic2netlist.preprocess import unproject_point

# distinguishable at a glance, and stable per net name across images
PALETTE = [(60, 60, 220), (30, 160, 30), (220, 120, 20), (200, 30, 200),
           (20, 170, 200), (120, 80, 200), (0, 110, 110), (150, 90, 30),
           (200, 60, 120), (80, 140, 60), (40, 90, 220), (140, 40, 40)]


UNSET = (255, 0, 255)      # magenta: a terminal whose net was never set


def net_colour(net, order):
    if net is None:
        return UNSET
    if net in ("0", 0):
        return (0, 0, 0)
    return PALETTE[order.index(net) % len(PALETTE)] if net in order else (120,) * 3


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--images", default="")
    ap.add_argument("--from-file", default=None,
                    help="text file of image names, one per line")
    ap.add_argument("--gt-dir", default=None,
                    help="override benchmark.gt_dir (e.g. data/gt_val_1024)")
    ap.add_argument("--from-welds", action="store_true")
    ap.add_argument("--welds", default="results/weld_review/welds.csv")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--transforms", default="data/transforms_1024.json")
    ap.add_argument("--out-dir", default="results/gt_overlay")
    ap.add_argument("--width", type=int, default=1500)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tf = json.load(open(args.transforms))
    if args.from_welds:
        names = sorted({r["image"] for r in csv.DictReader(open(args.welds))})
    elif args.from_file:
        names = [l.strip() for l in open(args.from_file) if l.strip()]
    else:
        names = [s.strip() for s in args.images.split(",") if s.strip()]
    gt_dir = args.gt_dir or cfg["benchmark"]["gt_dir"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for nm in names:
        stem = Path(nm).stem
        gp = Path(gt_dir) / f"{stem}.json"
        if not gp.exists():
            print(f"  !! no GT for {nm}")
            continue
        gt = load_gt(str(gp))
        meta = tf[stem]
        img = cv2.imread(str(Path(args.raw_dir) / nm))
        if img is None:
            print(f"  !! no original for {nm}")
            continue

        order, seen = [], set()
        for c in gt["components"]:
            for t in c["terminals"]:
                if (t["net"] is not None and t["net"] not in seen
                        and t["net"] != "0"):
                    seen.add(t["net"])
                    order.append(t["net"])

        vis = img.copy()
        H, W = vis.shape[:2]
        thick = max(2, int(round(W / 700)))
        fs = W / 1900.0
        for c in gt["components"]:
            bx, by, bw, bh = c["bbox"]
            corners = [unproject_point(meta, bx - bw / 2, by - bh / 2),
                       unproject_point(meta, bx + bw / 2, by - bh / 2),
                       unproject_point(meta, bx - bw / 2, by + bh / 2),
                       unproject_point(meta, bx + bw / 2, by + bh / 2)]
            xs, ys = [p[0] for p in corners], [p[1] for p in corners]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            nets = [t["net"] for t in c["terminals"]]
            col = net_colour(nets[0], order)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)
            txt = f"{c['id']}:{c['class']}"
            cv2.putText(vis, txt, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 255, 255),
                        int(thick * 2.6), cv2.LINE_AA)
            cv2.putText(vis, txt, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, col,
                        max(1, thick - 1), cv2.LINE_AA)
            # Each terminal's net carries ITS OWN colour. Painting the whole
            # label in nets[0]'s colour made a component straddling two nets
            # look like it sat on one, which is the thing this view exists to
            # show. Halos are drawn in a first pass: interleaving them lets the
            # next token's halo eat the previous token's glyphs.
            ly = min(H - 4, y2 + int(26 * fs * 1.5))
            toks, lx = [], x1
            for k, net in enumerate(nets):
                tok = ("?" if net is None else net) + (
                    "" if k == len(nets) - 1 else "|")
                toks.append((tok, lx, net))
                lx += cv2.getTextSize(tok, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9,
                                      max(1, thick - 1))[0][0]
            for tok, tx, _n in toks:
                cv2.putText(vis, tok, (tx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            fs * 0.9, (255, 255, 255), int(thick * 2.6),
                            cv2.LINE_AA)
            for tok, tx, _n in toks:
                cv2.putText(vis, tok, (tx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            fs * 0.9, net_colour(_n, order),
                            max(1, thick - 1), cv2.LINE_AA)

        s = args.width / vis.shape[1]
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        key = np.full((34 * ((len(order) + 5) // 6 + 1), vis.shape[1], 3),
                      250, np.uint8)
        for i, n in enumerate(order + ["0"]):
            cx, cy = 12 + (i % 6) * (vis.shape[1] // 6), 24 + (i // 6) * 32
            cv2.putText(key, f"{n}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        net_colour(n, order), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}_gt.png"), np.vstack([vis, key]))
        print(f"  {nm}: {len(gt['components'])} components, "
              f"{len(order)+1} nets")

    print(f"\nwrote {len(names)} overlays -> {out}")


if __name__ == "__main__":
    main()
