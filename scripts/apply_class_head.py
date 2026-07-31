#!/usr/bin/env python3
"""Relabel detections with the trained class head, at a chosen confidence.

Boxes are NOT touched. Only the label moves, and only where the head disagrees
with the detector confidently -- exactly as the seed vote does -- so any metric
change is attributable to labels alone.

The threshold matters more than the accuracy. Strict end-to-end success is a
product over every component in an image, so one wrong relabel destroys an image
that was previously correct while one right relabel only helps if the rest of
that image is already perfect. The asymmetry means this should be run at high
precision and low recall, and the sweep below reports precision against GT at
each operating point rather than assuming one.

Nothing here is adopted on accuracy alone: scripts/audit_relabels.py prices the
changes against ground truth first, and only a clearly positive net justifies
spending a benchmark.

Usage:
    python scripts/apply_class_head.py --threshold 0.9 --out data/detections_1024_head
    python scripts/apply_class_head.py --sweep       # price every operating point
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_class_dataset import crop_component
from train_class_head import Net

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


def score_all(cfg, ckpt, device):
    """(records) one per detection: current label, head label, head confidence,
    and the GT class where one can be matched."""
    names = ckpt["names"]
    size = ckpt.get("size", 128)
    model = Net(len(names))
    model.load_state_dict(ckpt["state"])
    model.to(device).eval()

    idir = Path(cfg["preprocess"]["images_dir"])
    cdir = Path(cfg["detect"]["cache_dir"])
    gdir = Path(cfg["benchmark"]["gt_dir"])
    conf = cfg["detect"].get("confidence")
    names_can = [canonical_class(n) for n in names]

    recs = []
    for lp in sorted(Path("data/splits/test.txt").read_text().split()):
        stem = Path(lp).stem
        cp, ip, gp = cdir / f"{stem}.json", idir / lp, gdir / f"{stem}.json"
        if not (cp.exists() and ip.exists()):
            continue
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        cache = json.loads(cp.read_text())
        dets = [d for d in cache["detections"]
                if conf is None or d.get("confidence", 1.0) >= conf]
        if not dets:
            continue
        gboxes = []
        if gp.exists():
            gt = load_gt(str(gp))
            gboxes = [(canonical_class(c["class"]), tuple(c["bbox"]))
                      for c in gt["components"]]
        crops = []
        keep = []
        for j, d in enumerate(dets):
            c = crop_component(gray, d["x"], d["y"], d["width"], d["height"],
                               size, 0.25)
            if c is None:
                continue
            crops.append(c)
            keep.append(j)
        if not crops:
            continue
        xb = torch.from_numpy(np.array(crops)).float().div(255).unsqueeze(1)
        with torch.no_grad():
            p = torch.softmax(model(xb.to(device)), 1).cpu().numpy()
        for k, j in enumerate(keep):
            d = dets[j]
            gi = int(p[k].argmax())
            box = (d["x"], d["y"], d["width"], d["height"])
            best, gc = 0.0, None
            for g_cls, g_box in gboxes:
                o = iou(box, g_box)
                if o > best:
                    best, gc = o, g_cls
            recs.append({"image": lp, "det_index": j,
                         "cur": canonical_class(d["class"]),
                         "head": names_can[gi], "conf": float(p[k][gi]),
                         "gt": gc if best >= 0.3 else None})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default="experiments/class_head/best.pt")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = torch.device(args.device if (args.device != "mps"
                                       or torch.backends.mps.is_available())
                       else "cpu")
    ckpt = torch.load(ROOT / args.weights, map_location="cpu",
                      weights_only=False)
    recs = score_all(cfg, ckpt, dev)
    dis = [r for r in recs if r["head"] != r["cur"]]
    print(f"{len(recs)} detections scored, head disagrees on {len(dis)} "
          f"({len(dis)/max(len(recs),1):.1%})\n")

    if args.sweep or args.out is None:
        print(f"  {'thresh':>7s} {'changes':>8s} {'corrected':>10s} "
              f"{'broke':>7s} {'net':>6s} {'precision':>10s}")
        for th in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
            sel = [r for r in dis if r["conf"] >= th and r["gt"] is not None]
            corr = sum(1 for r in sel if r["cur"] != r["gt"] and r["head"] == r["gt"])
            broke = sum(1 for r in sel if r["cur"] == r["gt"] and r["head"] != r["gt"])
            den = corr + broke
            print(f"  {th:7.2f} {len(sel):8d} {corr:10d} {broke:7d} "
                  f"{corr-broke:+6d} {(corr/den if den else float('nan')):10.1%}")
        print(f"\n  A wrong relabel destroys an image that was already correct;")
        print(f"  a right one helps only where the rest of that image is perfect.")
        print(f"  Pick the highest NET at a precision that clears the seed vote's")
        print(f"  87.1%, then price it with scripts/audit_relabels.py.")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        by_img = {}
        for r in dis:
            if r["conf"] >= args.threshold:
                by_img.setdefault(r["image"], []).append(r)
        cdir = Path(cfg["detect"]["cache_dir"])
        n_changed = 0
        changed = Counter()
        for lp in Path("data/splits/test.txt").read_text().split():
            stem = Path(lp).stem
            cp = cdir / f"{stem}.json"
            if not cp.exists():
                continue
            cache = json.loads(cp.read_text())
            for r in by_img.get(lp, []):
                d = cache["detections"][r["det_index"]]
                changed[f"{r['cur']} -> {r['head']}"] += 1
                d["class"] = r["head"]
                n_changed += 1
            (out / f"{stem}.json").write_text(json.dumps(cache) + "\n")
        print(f"\nwrote {out} — relabelled {n_changed} at threshold "
              f"{args.threshold}")
        for k, v in changed.most_common(12):
            print(f"  {k:34s} {v:4d}")


if __name__ == "__main__":
    main()
