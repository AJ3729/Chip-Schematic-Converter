#!/usr/bin/env python3
"""What corruption actually reached the pipeline, as opposed to what was asked for.

The nominal parameter is not the delivered one, and on this corpus the gap is
large enough to matter. Two mechanisms, both specific to scanned line art:

  CLIPPING   these scans average 249.7 of 255 -- almost pure white. Additive
             Gaussian noise on a pixel at 253 is clipped at 255, so over the
             background, which is most of the page, roughly half the noise is
             destroyed before it is ever written out. sigma 16 becomes 10.3.

  RE-ENCODE  the corrupted copies were written as JPEG at quality 95, which
             removes about another 29% of what clipping left. sigma 10.3
             becomes 7.3.

Impulse noise is untouched by both: it is already saturated, so clipping is a
no-op, and JPEG preserves it. Its delivered rate reads as half the nominal for a
real reason rather than a lossy one -- salt landing on white background changes
nothing, so only pepper-on-white and salt-on-ink are visible.

Reporting nominal sigma in the manuscript would overstate the severity of the
Gaussian arms by more than 2x. This writes the measured values so the text can
quote what happened.

Usage:
    python scripts/robustness/measure_delivered.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import corruptions as C  # noqa: E402

NOMINAL = {"gauss_noise": {1: 8, 2: 16, 3: 32}}
CHANGED_THRESH = 60          # a pixel a reader would call altered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="results/robustness/delivered_corruption.json")
    a = ap.parse_args()

    stems = [Path(l.strip()).stem for l in
             (ROOT / "data/splits/test.txt").read_text().split() if l.strip()][:a.n]

    out: dict = {"what_this_is": "delivered vs nominal corruption, measured on disk",
                 "n_images": 0, "changed_threshold": CHANGED_THRESH,
                 "conditions": {}}
    n_used = 0
    for cond, fam, sev in C.conditions():
        if cond == "clean":
            continue
        d = ROOT / "data/robustness/raw" / cond
        if not d.is_dir():
            continue
        sig_mem, sig_disk, chg_mem, chg_disk = [], [], [], []
        for s in stems:
            p = d / f"{s}.jpg"
            src = ROOT / "data/raw" / f"{s}.jpg"
            if not p.exists() or not src.exists():
                continue
            a0 = cv2.imread(str(src))
            b = cv2.imread(str(p))
            if a0 is None or b is None or a0.shape != b.shape:
                continue
            mem = C.apply(cond, a0, s).astype(np.float32)
            af, bf = a0.astype(np.float32), b.astype(np.float32)
            sig_mem.append(float((mem - af).std()))
            sig_disk.append(float((bf - af).std()))
            chg_mem.append(float((np.abs(mem - af) > CHANGED_THRESH).any(axis=2).mean()))
            chg_disk.append(float((np.abs(bf - af) > CHANGED_THRESH).any(axis=2).mean()))
        if not sig_disk:
            continue
        n_used = max(n_used, len(sig_disk))
        rec = {
            "family": fam, "severity": sev,
            "sigma_in_memory": round(float(np.mean(sig_mem)), 3),
            "sigma_on_disk": round(float(np.mean(sig_disk)), 3),
            "changed_frac_in_memory": round(float(np.mean(chg_mem)), 5),
            "changed_frac_on_disk": round(float(np.mean(chg_disk)), 5),
        }
        if fam == "photometric" and cond.rsplit("_s", 1)[0] in NOMINAL:
            nom = NOMINAL[cond.rsplit("_s", 1)[0]][sev]
            rec["nominal_sigma"] = nom
            rec["clipping_loss_frac"] = round(1 - rec["sigma_in_memory"] / nom, 3)
            rec["jpeg_loss_frac"] = round(
                1 - rec["sigma_on_disk"] / max(rec["sigma_in_memory"], 1e-9), 3)
        out["conditions"][cond] = rec
    out["n_images"] = n_used

    g = {k: v for k, v in out["conditions"].items() if k.startswith("gauss_noise")}
    out["gaussian_summary"] = {
        "nominal": [g[f"gauss_noise_s{i}"]["nominal_sigma"] for i in (1, 2, 3)],
        "delivered": [g[f"gauss_noise_s{i}"]["sigma_on_disk"] for i in (1, 2, 3)],
        "raw_mean_intensity": 249.7,
    }
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"measured on {n_used} images\n")
    print(f"{'condition':16}{'nominal':>9}{'in-mem':>9}{'on-disk':>9}{'changed':>10}")
    for c, r in out["conditions"].items():
        nom = r.get("nominal_sigma", "")
        print(f"{c:16}{str(nom):>9}{r['sigma_in_memory']:9.2f}"
              f"{r['sigma_on_disk']:9.2f}{r['changed_frac_on_disk']*100:9.2f}%")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
