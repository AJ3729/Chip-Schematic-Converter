#!/usr/bin/env python3
"""Test the crossing-darkness hypothesis where the evidence still EXISTS.

Where two pen strokes cross, the ink is laid twice and those pixels are
darker. That is the one physical cue not yet tested, and the natural place to
look after everything else closed: perfect crossover boxes make strict
success worse, per-site binary geometry caps at 0.70 precision, the junction
dot is at chance (AUC 0.5017), every uniform parameter sits at a local
optimum, and the dominant weld is two nets on ONE continuous conductor with
no branch point between them -- a configuration no reasoning over the mask
can undo, because the information is already gone.

Measuring darkness on the pipeline's own frames returned nothing (dark_peak
AUC 0.4998, and 0.5009 restricted to degree>=4 where an X must double the
ink). But those frames carry no dynamic range to measure: 93.5% of pixels are
exactly 255 and the ink is crushed to near-0, median grey about 8. That is a
NULL MEASUREMENT ON DESTROYED EVIDENCE, not a refutation.

The original photographs do retain it -- Otsu threshold 172, ink median grey
86 with standard deviation 29.3 and a 5th-to-95th percentile span of 53 to
154. So this re-measures the same 4822 causally-labelled sites after mapping
each one back to original-image coordinates with
``preprocess.unproject_point``, and reads darkness there.

If the cue is real, preserving it through preprocessing is an upstream,
information-preserving change and the first live avenue in a while. If it is
absent in the originals too, the crossing question is closed on physical
grounds rather than for lack of a good enough model, which is a much stronger
statement for the paper.

Controls, because a positive result here is easy to fake:

  stroke_dark_local  the local stroke darkness itself. If this separates the
                     classes, the signal is pen pressure or lighting, not
                     crossings.
  within-image       the same comparison restricted inside each drawing,
                     which removes any between-drawing confound.
  degree>=4          where two strokes genuinely cross, so the doubling must
                     appear if it appears anywhere.

Usage:
    python scripts/measure_ink_darkness_original.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.preprocess import unproject_point

ORIG = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
        "Component Symbol and Text Label Data/Circuit Diagram Images")
FEATS = ["dark_at_site", "dark_peak", "dark_excess_area", "stroke_dark_local"]


def auc_score(pos, neg) -> float:
    import scipy.stats as ss
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = ss.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default="results/real_crossings/sites_test.csv")
    ap.add_argument("--transforms", default="data/transforms_1024.json")
    ap.add_argument("--orig-dir", default=ORIG)
    ap.add_argument("--out-dir", default="results/ink_darkness_original")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf = json.loads(Path(args.transforms).read_text())

    sites = list(csv.DictReader(open(args.sites)))
    by_img = defaultdict(list)
    for r in sites:
        by_img[r["image"]].append(r)
    print(f"{len(sites)} labelled sites over {len(by_img)} images")

    rows = []
    missing = 0
    for i, nm in enumerate(sorted(by_img), 1):
        stem = Path(nm).stem
        meta = tf.get(stem)
        p = Path(args.orig_dir) / nm
        if meta is None or not p.exists():
            missing += 1
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing += 1
            continue
        H, W = img.shape
        _thr, ink = cv2.threshold(img, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # scale: how many original px per cleaned px
        inv_scale = 1.0 / max(meta["scale"], 1e-9)

        for r in by_img[nm]:
            ox, oy = unproject_point(meta, float(r["x"]), float(r["y"]))
            ox, oy = int(round(ox)), int(round(oy))
            if not (0 <= ox < W and 0 <= oy < H):
                continue
            # radius scaled from the cleaned-frame stroke half-width
            hw = float(r.get("stroke_hw", 2.0) or 2.0) * inv_scale
            rs = max(3, int(round(hw * 2.0)))
            rl = max(rs * 4, 12)            # local neighbourhood for baseline
            def win(rad):
                y0, y1 = max(0, oy-rad), min(H, oy+rad+1)
                x0, x1 = max(0, ox-rad), min(W, ox+rad+1)
                return img[y0:y1, x0:x1], ink[y0:y1, x0:x1]
            ps, ks = win(rs)
            pl, kl = win(rl)
            vs, vl = ps[ks > 0], pl[kl > 0]
            if vs.size == 0 or vl.size < 20:
                continue
            # LOCAL baseline: the stroke darkness right around this site, so
            # pen pressure and lighting cancel instead of confounding
            base = float(np.median(vl))
            sd = float(np.std(vl)) or 1.0
            rows.append({
                "image": nm, "x": int(r["x"]), "y": int(r["y"]),
                "label": int(r["label"]), "degree": int(r["degree"]),
                "dark_at_site": round((base - float(np.mean(vs))) / sd, 4),
                "dark_peak": round((base - float(np.min(vs))) / sd, 4),
                "dark_excess_area": int((vs < base - sd).sum()),
                "stroke_dark_local": round(base, 2),
            })
        if i % 40 == 0:
            print(f"  [{i}/{len(by_img)}] {len(rows)} sites", flush=True)

    if missing:
        print(f"  ({missing} images had no original or no transform)")
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    print(f"\n=== INK DARKNESS ON THE ORIGINAL PHOTOGRAPHS ===")
    print(f"sites {len(rows)}  must-split {len(pos)}  must-union {len(neg)}\n")
    print(f"  {'feature':20s} {'AUC':>7s} {'|AUC-.5|':>9s} "
          f"{'mean(split)':>12s} {'mean(union)':>12s}")
    res = {}
    for f in FEATS:
        a = auc_score([r[f] for r in pos], [r[f] for r in neg])
        res[f] = round(a, 4)
        print(f"  {f:20s} {a:7.4f} {abs(a-0.5):9.4f} "
              f"{np.mean([r[f] for r in pos]):12.3f} "
              f"{np.mean([r[f] for r in neg]):12.3f}")

    print(f"\n  degree>=4 only (two strokes genuinely cross, so the doubling")
    print(f"  must show here if it shows anywhere):")
    for f in ("dark_at_site", "dark_peak"):
        p4 = [r[f] for r in pos if r["degree"] >= 4]
        n4 = [r[f] for r in neg if r["degree"] >= 4]
        if p4 and n4:
            print(f"    {f:20s} AUC {auc_score(p4, n4):.4f}  "
                  f"({len(p4)} split / {len(n4)} union)")

    print(f"\n  WITHIN-IMAGE (removes any between-drawing confound):")
    per = defaultdict(list)
    for r in rows:
        per[r["image"]].append(r)
    for f in ("dark_at_site", "dark_peak"):
        wins = tot = 0
        for nm, rs in per.items():
            p = [x[f] for x in rs if x["label"] == 1]
            n = [x[f] for x in rs if x["label"] == 0]
            if not p or not n:
                continue
            tot += 1
            wins += int(np.mean(p) > np.mean(n))
        if tot:
            print(f"    {f:20s} split darker in {wins}/{tot} = {wins/tot:.1%} "
                  f"of images  (50% = nothing)")

    if rows:
        with (out / "sites_darkness_original.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_sites": len(rows), "n_must_split": len(pos), "n_must_union": len(neg),
        "per_feature_auc": res,
        "source": "original photographs, sites unprojected via transforms_1024",
        "note": ("The pipeline's own frames carry no dynamic range (93.5% of "
                 "pixels exactly 255, ink median grey ~8), so the earlier null "
                 "there measured destroyed evidence. This measures where the "
                 "evidence still exists."),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
