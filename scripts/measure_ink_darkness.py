#!/usr/bin/env python3
"""Does stroke DARKNESS distinguish a crossing from a junction?

Everything measured so far says the mask does not contain the answer. The
dominant weld is two ground-truth nets sitting on ONE continuous wire run
with no branch point between them, so no cut along the skeleton separates
them; every uniform parameter (component pad, bridge span, notch, stitch) is
already at a local optimum; perfect crossover boxes make strict success
worse; and per-site evidence from binary geometry caps at 0.70 precision,
with the junction dot at exactly chance (AUC 0.5017 over 4822 real sites).

That points upstream rather than downstream. Binarization is where the
information goes: it converts a grey pen stroke into a flat mask, and one
physical cue does not survive it. Where two strokes CROSS, the ink is laid
down twice, so those pixels are darker than either stroke alone. Where a
stroke ENDS on another (a T-junction) the overlap is smaller, and where a
deliberate junction dot is drawn the darkening is broad rather than
localized to a crossing point. The distance transform used by ``dot_ratio``
measures stroke WIDTH on the binary mask and is blind to all of it -- which
may be exactly why it scored at chance.

This reuses the 4822 causally-labelled real sites already extracted by
``scripts/build_real_crossing_features.py`` (labels derived from verified GT
netlists by the causal cut test, so no human annotation) and adds greyscale
features read from the preprocessed frame the pipeline actually consumes:

  dark_at_site      how much darker the site is than nearby stroke, in
                    local standard deviations
  dark_ratio        site darkness / median stroke darkness
  dark_peak         darkest single pixel at the site, same normalization
  dark_area         how many pixels near the site are darker than the
                    stroke baseline (broad dot versus point crossing)
  stroke_dark       the local stroke baseline itself, as a control -- if
                    this separates the classes, the "signal" is really
                    pen-pressure confounding, not crossing evidence

Per-feature AUC needs no fitting, so it is contamination-free. The control
feature matters: a difference driven by ``stroke_dark`` would mean darker
drawings simply have more crossings, which is a dataset artefact rather than
usable local evidence.

Usage:
    python scripts/measure_ink_darkness.py
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

from schematic2netlist.config import load_config
from schematic2netlist.frames import resolve_and_check

FEATS = ["dark_at_site", "dark_ratio", "dark_peak", "dark_area", "stroke_dark"]


def auc_score(pos, neg) -> float:
    import scipy.stats as ss
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = ss.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default="results/real_crossings/sites_test.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/ink_darkness")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sites = list(csv.DictReader(open(args.sites)))
    by_img = defaultdict(list)
    for r in sites:
        by_img[r["image"]].append(r)
    names = sorted(by_img)
    images_dir = resolve_and_check(None, names, cfg)
    print(f"{len(sites)} labelled sites over {len(names)} images")

    rows = []
    for i, nm in enumerate(names, 1):
        img = cv2.imread(str(images_dir / nm), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        # stroke baseline: the ink itself, not the paper. Otsu on the frame
        # separates them, and the MEDIAN of ink pixels is the typical stroke
        # darkness this drawing was made with.
        thr, ink = cv2.threshold(img, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_vals = img[ink > 0]
        if ink_vals.size < 50:
            continue
        stroke_med = float(np.median(ink_vals))
        stroke_sd = float(np.std(ink_vals)) or 1.0

        for r in by_img[nm]:
            x, y = int(r["x"]), int(r["y"])
            hw = float(r.get("stroke_hw", 2.0) or 2.0)
            w = max(3, int(round(hw * 2.0)))
            y0, y1 = max(0, y - w), min(H, y + w + 1)
            x0, x1 = max(0, x - w), min(W, x + w + 1)
            patch = img[y0:y1, x0:x1]
            pink = ink[y0:y1, x0:x1]
            vals = patch[pink > 0]
            if vals.size == 0:
                continue
            site_dark = float(np.mean(vals))
            peak = float(np.min(vals))
            # darker = LOWER grey value, so invert the sign so that a
            # positive number always means "darker than the local stroke"
            rows.append({
                "image": nm, "x": x, "y": y, "label": int(r["label"]),
                "degree": int(r["degree"]),
                "dark_at_site": round((stroke_med - site_dark) / stroke_sd, 4),
                "dark_ratio": round(stroke_med / max(site_dark, 1e-6), 4),
                "dark_peak": round((stroke_med - peak) / stroke_sd, 4),
                "dark_area": int((vals < stroke_med - stroke_sd).sum()),
                "stroke_dark": round(stroke_med, 2),
            })
        if i % 40 == 0:
            print(f"  [{i}/{len(names)}] {len(rows)} sites", flush=True)

    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    print(f"\n=== INK DARKNESS AT CAUSALLY-LABELLED REAL SITES ===")
    print(f"sites {len(rows)}  must-split {len(pos)}  must-union {len(neg)}\n")
    print(f"  {'feature':16s} {'AUC':>7s} {'|AUC-.5|':>9s} "
          f"{'mean(split)':>12s} {'mean(union)':>12s}")
    res = {}
    for f in FEATS:
        a = auc_score([r[f] for r in pos], [r[f] for r in neg])
        res[f] = round(a, 4)
        print(f"  {f:16s} {a:7.4f} {abs(a-0.5):9.4f} "
              f"{np.mean([r[f] for r in pos]):12.3f} "
              f"{np.mean([r[f] for r in neg]):12.3f}")

    print(f"\n  stroke_dark is the CONTROL. If it separates the classes as "
          f"well as\n  the site features do, the apparent signal is "
          f"pen-pressure confounding\n  between drawings, not local crossing "
          f"evidence.")

    # restrict to degree-4 sites, where a true crossing doubles the ink
    d4p = [r[f] for f in ("dark_peak",) for r in pos if r["degree"] >= 4]
    d4n = [r[f] for f in ("dark_peak",) for r in neg if r["degree"] >= 4]
    if d4p and d4n:
        print(f"\n  degree>=4 only (where an X should double the ink): "
              f"dark_peak AUC {auc_score(d4p, d4n):.4f} "
              f"({len(d4p)} split / {len(d4n)} union)")

    # within-image ranking removes the between-drawing confound entirely
    print(f"\n  WITHIN-IMAGE comparison (removes the pen-pressure confound):")
    for f in ("dark_at_site", "dark_peak", "dark_ratio"):
        wins = ties = tot = 0
        for nm, rs in defaultdict(list, {k: [r for r in rows if r["image"] == k]
                                         for k in {r["image"] for r in rows}}
                                  ).items():
            p = [r[f] for r in rs if r["label"] == 1]
            n = [r[f] for r in rs if r["label"] == 0]
            if not p or not n:
                continue
            tot += 1
            if np.mean(p) > np.mean(n):
                wins += 1
            elif np.mean(p) == np.mean(n):
                ties += 1
        if tot:
            print(f"    {f:16s} split darker in {wins}/{tot} images "
                  f"= {wins/tot:.1%}  (50% = no signal)")

    if rows:
        with (out / "sites_darkness.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_sites": len(rows), "n_must_split": len(pos), "n_must_union": len(neg),
        "per_feature_auc": res,
        "note": ("Labels come from the causal cut test on verified GT "
                 "netlists. stroke_dark is a control for pen-pressure "
                 "confounding between drawings."),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
