#!/usr/bin/env python3
"""The condition that bridges the corruption sweep and the cross-corpus result.

WHY NOT JUST PAD THE SCANS. Preprocessing crops to the bounding box of ink, so
a white margin is cropped straight back off. Measured: a 50% margin on every
side shrinks the rendered circuit by only 7%, and that 7% comes from
crop_pad_frac (2% of the larger dimension) rather than from the margin. Reaching
CGHD's component scale that way needs padding of roughly 4.6x the content width
per side -- a 97%-white image, which is not a photograph anyone would take.

So relative component size is set directly, in the frame where it is defined.
The cleaned 1024 image is rescaled by f about the centre onto a white canvas and
the ground truth is rescaled by the SAME f, which keeps every box exactly where
its component is. Nothing is approximated: the transform is a similarity with a
known factor applied to both sides.

WHY THIS IS THE RIGHT VARIABLE. Component area in the 1024 frame, measured here:

    Digitize-HCD test   p5 1325   p25 2179   median 3194   min 640
    CGHD                p5  168   p25  633   median 1606   min  18

43.5% of CGHD components fall below Digitize-HCD's 5th percentile, reproducing
the figure the manuscript reports. Median area differs by 0.503, i.e. a LINEAR
factor of 0.709. The factors below step from mild to past CGHD's p25, so the
sweep crosses the corpus it is meant to bridge to instead of stopping short.

Usage:
    python scripts/robustness/scale_bridge.py --factors 0.85,0.71,0.54,0.36
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data/robustness"
RES = ROOT / "results/robustness"


def scale_image(img: np.ndarray, f: float, size: int = 1024) -> np.ndarray:
    """Shrink about the centre onto a white canvas of the same size."""
    n = max(8, int(round(size * f)))
    small = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8) if img.ndim == 3 else \
        np.full((size, size), 255, np.uint8)
    o = (size - n) // 2
    canvas[o:o + n, o:o + n] = small
    return canvas


def scale_gt(gt: dict, f: float, size: int = 1024) -> dict:
    """Same similarity transform, applied to the boxes."""
    c = size / 2.0
    out = json.loads(json.dumps(gt))
    for comp in out.get("components", []):
        bb = comp.get("bbox")
        if not bb:
            continue
        x, y, w, h = bb
        comp["bbox"] = [(x - c) * f + c, (y - c) * f + c, w * f, h * f]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factors", default="0.85,0.71,0.54,0.36")
    ap.add_argument("--src-clean", default="data/robustness/cleaned/clean")
    ap.add_argument("--gt-dir", default="data/gt_test_1024")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    stems = [Path(l.strip()).stem for l in
             (ROOT / f"data/splits/{a.split}.txt").read_text().split() if l.strip()]

    for f in [float(x) for x in a.factors.split(",")]:
        cond = f"scale_f{int(round(f * 100)):02d}"
        clean_dir = DATA / "cleaned" / cond
        gt_dir = DATA / "gt_scaled" / cond
        det_dir = DATA / "detections" / cond
        out_dir = RES / cond
        for d in (clean_dir, gt_dir, det_dir, out_dir):
            d.mkdir(parents=True, exist_ok=True)

        areas = []
        for stem in stems:
            sp = ROOT / a.src_clean / f"{stem}.jpg"
            gp = ROOT / a.gt_dir / f"{stem}.json"
            if not sp.exists() or not gp.exists():
                continue
            img = cv2.imread(str(sp))
            cv2.imwrite(str(clean_dir / f"{stem}.jpg"), scale_image(img, f),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            g = scale_gt(json.loads(gp.read_text()), f)
            (gt_dir / f"{stem}.json").write_text(json.dumps(g))
            areas += [c["bbox"][2] * c["bbox"][3]
                      for c in g.get("components", []) if c.get("bbox")]
        areas.sort()
        med = areas[len(areas) // 2] if areas else 0

        cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        cfg["detect"]["cache_dir"] = str(det_dir.relative_to(ROOT))
        cfg["preprocess"]["images_dir"] = str(clean_dir.relative_to(ROOT))
        cfg["benchmark"]["gt_dir"] = str(gt_dir.relative_to(ROOT))
        cp = ROOT / "configs/robustness" / f"{cond}.yaml"
        cp.write_text(yaml.safe_dump(cfg, sort_keys=False))

        print(f"[{cond}] f={f}  median component area {med:.0f} px^2", flush=True)
        t = time.time()
        for label, cmd in (
            ("detect", [sys.executable, "scripts/detect_batch.py",
                        "--images-dir", str(clean_dir.relative_to(ROOT)),
                        "--config", str(cp.relative_to(ROOT))]),
            ("benchmark", [sys.executable, "scripts/benchmark.py",
                           "--split", a.split,
                           "--images-dir", str(clean_dir.relative_to(ROOT)),
                           "--gt-dir", str(gt_dir.relative_to(ROOT)),
                           "--out-dir", str(out_dir.relative_to(ROOT)),
                           "--config", str(cp.relative_to(ROOT))])):
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  !! {label} failed\n" + (r.stderr or r.stdout)[-600:], flush=True)
                break
        else:
            s = json.loads((out_dir / "summary.json").read_text())
            (out_dir / "scale_meta.json").write_text(json.dumps(
                {"factor": f, "median_component_area_px2": med,
                 "cghd_median_area_px2": 1606,
                 "hcd_clean_median_area_px2": 3194}, indent=1) + "\n")
            print(f"  strict={s['topology']['strict_success']['mean']:.4f} "
                  f"({time.time() - t:.0f}s)", flush=True)
        # the scaled frames are derived and large; the GT and results are not
        shutil.rmtree(clean_dir, ignore_errors=True)
    print("SCALE BRIDGE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
