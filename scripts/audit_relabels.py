#!/usr/bin/env python3
"""Price a relabelled detection cache against GT before spending a benchmark.

An ensemble that changes N labels has done something useful only if those
changes land on labels that were WRONG. Counting changes cannot tell the
difference, and neither can overall class accuracy when the errors are 2.4% of
detections: a vote that fixes 27 and breaks 4 and a vote that fixes 4 and
breaks 27 both read as "31 labels changed".

So each changed label is resolved against the best-overlapping GT component and
sorted into four outcomes:

    corrected     was wrong, now right      -- the point of the exercise
    broke         was right, now wrong      -- the cost
    still_wrong   was wrong, still wrong    -- churn, neither helps nor hurts
    no_gt         no GT box overlaps        -- a false positive being relabelled

Precision here is corrected / (corrected + broke), the number to compare against
the seed vote's 84.4%, and the net is corrected - broke. A negative or barely
positive net means do not benchmark it.

Usage:
    python scripts/audit_relabels.py data/detections_1024 data/detections_1024_tta
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.gt import load_gt


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou", type=float, default=0.3,
                    help="GT match threshold; 0.3 matches the benchmark's "
                         "component alignment, not a strict 0.5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    conf = cfg["detect"].get("confidence")
    gdir = Path(cfg["benchmark"]["gt_dir"])
    names = [l.strip() for l in open(f"data/splits/{args.split}.txt")
             if l.strip()]

    out = Counter()
    detail = Counter()
    per_image: dict[str, Counter] = {}
    n_det = 0
    for nm in names:
        stem = Path(nm).stem
        bp, apth, gp = (Path(args.before) / f"{stem}.json",
                        Path(args.after) / f"{stem}.json",
                        gdir / f"{stem}.json")
        if not (bp.exists() and apth.exists() and gp.exists()):
            continue
        bd = json.loads(bp.read_text())["detections"]
        ad = json.loads(apth.read_text())["detections"]
        if len(bd) != len(ad):
            print(f"[warn] {stem}: {len(bd)} vs {len(ad)} detections -- the "
                  f"caches differ in geometry, not only labels; skipping")
            continue
        gt = load_gt(str(gp))
        gboxes = [(canonical_class(c["class"]), c["bbox"])
                  for c in gt["components"]]

        for b, a in zip(bd, ad):
            n_det += 1
            if conf is not None and b.get("confidence", 1.0) < conf:
                continue
            cb, ca = canonical_class(b["class"]), canonical_class(a["class"])
            if cb == ca:
                continue
            box = (b["x"], b["y"], b["width"], b["height"])
            best, gcls = 0.0, None
            for gc, gb in gboxes:
                o = iou(box, tuple(gb))
                if o > best:
                    best, gcls = o, gc
            if gcls is None or best < args.iou:
                kind = "no_gt"
            elif cb != gcls and ca == gcls:
                kind = "corrected"
            elif cb == gcls and ca != gcls:
                kind = "broke"
            else:
                kind = "still_wrong"
            out[kind] += 1
            detail[f"{kind}: {cb} -> {ca}" + ("" if kind == "no_gt"
                                              else f" (gt {gcls})")] += 1
            per_image.setdefault(nm, Counter())[kind] += 1

    corrected, broke = out["corrected"], out["broke"]
    total = sum(out.values())
    print(f"=== {args.before}  ->  {args.after} ===")
    print(f"{n_det} detections compared, {total} labels changed\n")
    for k in ("corrected", "broke", "still_wrong", "no_gt"):
        print(f"  {k:12s} {out[k]:4d}")
    denom = corrected + broke
    print(f"\n  precision  {corrected}/{denom} = "
          f"{corrected/denom:.1%}" if denom else "\n  precision  n/a")
    print(f"  NET        {corrected - broke:+d} correct labels")

    print(f"\n  changes by kind:")
    for k, v in detail.most_common(24):
        print(f"    {k:56s} {v:3d}")

    hurt = [nm for nm, c in per_image.items() if c["broke"] and not c["corrected"]]
    helped = [nm for nm, c in per_image.items()
              if c["corrected"] and not c["broke"]]
    print(f"\n  images with only corrections: {len(helped)}")
    print(f"  images with only breaks:      {len(hurt)}"
          + (f"  {hurt[:6]}" if hurt else ""))
    print(f"\n  Strict success is a product over every component in an image, so")
    print(f"  an image with ONLY breaks can lose strict while the net stays")
    print(f"  positive. That list is the risk, not the net.")


if __name__ == "__main__":
    main()
