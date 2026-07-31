#!/usr/bin/env python3
"""Is the detection cache aligned with the frames it is being used with?

The trap, measured: regenerating frames while keeping the committed detection
cache costs 0.027 terminal-pair F1 and 0.032 strict success -- worse than several
real regressions -- and NOTHING complains. Both artifacts look present and fresh,
audit_data_freshness.py reports "ok" for each, and the benchmark runs to
completion. The boxes are simply in the wrong place by a few pixels.

It is easy to hit. Frames are regenerated to sweep a preprocess parameter, or
pulled from a different generation, and the cache is left alone because
regenerating it means running the detector. That is exactly how this session
produced a bogus "the speck filter has a sharp optimum" result: every arm that
regenerated frames scored ~0.744 and the single arm using the committed frames
scored 0.767, so the sweep measured provenance rather than the parameter.

The check needs no detector, but it does need the right signal. Ink COVERAGE
inside the boxes was tried first and is useless here: a few pixels of frame shift
still leaves every box sitting on ink, and the aligned and mismatched pairs scored
6.059 and 6.069 -- indistinguishable.

What does separate them is CENTRING. A box the detector drew on these frames has
its ink centroid at the box centre; a box drawn on a different generation of them
is offset by however much the frames moved. Measured over 2815 detections:

    committed frames + committed cache      mean |offset| 2.943 px
    regenerated frames + regenerated cache  mean |offset| 2.954 px
    regenerated frames + COMMITTED cache    mean |offset| 4.074 px

The two correctly-paired stacks agree to 0.011 px while the mismatch is 1.13 px
adrift, so a threshold between them is safe and the quantity means something
physical rather than being tuned to this dataset.

Usage:
    python scripts/check_cache_alignment.py
    python scripts/check_cache_alignment.py --images-dir data/cleaned_1024_repro \\
        --cache-dir data/detections_1024        # the mismatch, to see it fire
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.config import load_config


def ink_of(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-offset", type=float, default=3.2,
                    help="mean px between a box centre and its ink centroid; "
                         "correctly paired stacks measure ~2.95, a mismatched "
                         "one ~3.63; 3.2 sits midway")
    args = ap.parse_args()

    cfg = load_config(args.config)
    idir = Path(args.images_dir or cfg["preprocess"]["images_dir"])
    cdir = Path(args.cache_dir or cfg["detect"]["cache_dir"])
    conf = cfg["detect"].get("confidence")
    names = [l.strip() for l in open(ROOT / f"data/splits/{args.split}.txt")
             if l.strip()]
    if args.limit:
        names = names[: args.limit]

    rows = []
    for nm in names:
        stem = Path(nm).stem
        ip, cp = idir / nm, cdir / f"{stem}.json"
        if not (ip.exists() and cp.exists()):
            continue
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = ink_of(g)
        frame_density = float(ink.mean())
        dets = [d for d in json.loads(cp.read_text())["detections"]
                if conf is None or d.get("confidence", 1.0) >= conf]
        if not dets:
            continue
        offsets = []
        out_of_bounds = 0
        H, W = ink.shape
        for d in dets:
            x1 = int(round(d["x"] - d["width"] / 2))
            y1 = int(round(d["y"] - d["height"] / 2))
            x2 = int(round(d["x"] + d["width"] / 2))
            y2 = int(round(d["y"] + d["height"] / 2))
            if x1 < -2 or y1 < -2 or x2 > W + 2 or y2 > H + 2:
                out_of_bounds += 1
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            sub = ink[y1:y2, x1:x2]
            if sub.size == 0 or not sub.any():
                continue
            ys, xs = np.nonzero(sub)
            offsets.append(float(np.hypot(x1 + xs.mean() - d["x"],
                                          y1 + ys.mean() - d["y"])))
        if not offsets:
            continue
        rows.append({"image": nm, "n_dets": len(dets),
                     "offset": float(np.mean(offsets)),
                     "frame_density": frame_density,
                     "out_of_bounds": out_of_bounds})

    if not rows:
        raise SystemExit("no comparable image/cache pairs found")

    offs = np.array([r["offset"] for r in rows])
    oob = sum(r["out_of_bounds"] for r in rows)
    print(f"frames  {idir}")
    print(f"cache   {cdir}")
    print(f"{len(rows)} images, {sum(r['n_dets'] for r in rows)} detections\n")
    print(f"  mean box-centre to ink-centroid offset  {offs.mean():.3f} px")
    print(f"  median / p90                            "
          f"{np.median(offs):.3f} / {np.percentile(offs, 90):.3f} px")
    print(f"  boxes outside frame                     {oob}")

    worst = sorted(rows, key=lambda r: -r["offset"])[:6]
    print(f"\n  worst-centred images:")
    for r in worst:
        print(f"    {r['image']:22s} offset {r['offset']:6.3f} px  "
              f"({r['n_dets']} dets)")

    print()
    if offs.mean() > args.max_offset:
        print(f"  FAIL — mean offset {offs.mean():.3f} px > {args.max_offset}. The "
              f"boxes are not centred on the ink in THESE frames.\n"
              f"  Almost certainly the cache was computed on a different "
              f"generation of them. Regenerate it:\n"
              f"    python scripts/detect_batch.py --images-dir {idir}")
        sys.exit(1)
    print(f"  PASS — mean offset {offs.mean():.3f} px <= {args.max_offset}; the "
          f"cache was computed on these frames.")


if __name__ == "__main__":
    main()
