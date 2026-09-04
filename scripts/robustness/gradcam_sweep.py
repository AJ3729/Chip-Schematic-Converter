#!/usr/bin/env python3
"""Grad-CAM for every condition, over the SAME images, so they can be compared.

The image list is fixed across conditions deliberately. A heatmap is only
interesting here next to its own clean counterpart -- "the detector stopped
looking at the transistor once the blur got to sigma 13" is a claim about one
drawing under two conditions, and it cannot be made if each condition sampled
different drawings.

Also writes a per-condition contact sheet: clean beside corrupted, for the same
circuit, so the degradation is visible without opening two folders.

Usage:
    python scripts/robustness/gradcam_sweep.py --n 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import corruptions as C  # noqa: E402

CAM = ROOT / "results/robustness/gradcam"


def sheet(cond: str, stems: list[str], kind: str) -> None:
    """clean | corrupted, stacked, for a handful of circuits."""
    rows = []
    for stem in stems[:6]:
        a = CAM / kind / "clean" / f"{stem}_cam.jpg"
        b = CAM / kind / cond / f"{stem}_cam.jpg"
        if not (a.exists() and b.exists()):
            continue
        ia, ib = cv2.imread(str(a)), cv2.imread(str(b))
        if ia is None or ib is None:
            continue
        h = 300
        ia = cv2.resize(ia, (int(ia.shape[1] * h / ia.shape[0]), h))
        ib = cv2.resize(ib, (int(ib.shape[1] * h / ib.shape[0]), h))
        w = max(ia.shape[1], ib.shape[1])
        pad = lambda im: cv2.copyMakeBorder(  # noqa: E731
            im, 0, 0, 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        row = np.hstack([pad(ia), np.full((h, 8, 3), 255, np.uint8), pad(ib)])
        cv2.putText(row, f"{stem}   clean | {cond}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(row)
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for r in rows]
    out = CAM / "contact_sheets" / kind
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{cond}.jpg"), np.vstack(rows),
                [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--method", default="gradcam")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    stems = [Path(l.strip()).stem for l in
             (ROOT / "data/splits/test.txt").read_text().split() if l.strip()][:a.n]
    conds = [c for c, _, _ in C.conditions()]
    if a.only:
        conds = [c.strip() for c in a.only.split(",") if c.strip()]

    for i, cond in enumerate(conds, 1):
        if not (ROOT / "data/robustness/transforms" / f"{cond}.json").exists():
            print(f"  [{i}/{len(conds)}] {cond}: skipped (condition not run yet)")
            continue
        r = subprocess.run(
            [sys.executable, "scripts/robustness/gradcam_overlay.py",
             "--condition", cond, "--method", a.method, "--limit", str(a.n)],
            cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"  [{i}/{len(conds)}] {cond:18} {'ok' if ok else 'FAILED'}", flush=True)
        if not ok:
            print("     " + (r.stderr or "").strip().splitlines()[-1][:120])
            continue
        if cond != "clean":
            for kind in ("cleaned", "raw"):
                sheet(cond, stems, kind)
    print(f"\noverlays  -> {CAM.relative_to(ROOT)}/{{cleaned,raw}}/<condition>/")
    print(f"contact sheets -> {CAM.relative_to(ROOT)}/contact_sheets/{{cleaned,raw}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
