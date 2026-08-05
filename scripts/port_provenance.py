#!/usr/bin/env python3
"""Which published port crops came from TRAIN images? Recover it by matching.

The Digitize-HCD archive ships a "Component Port Location Data" tree: 320x320
crops of multi-terminal components, each with a NAMED port coordinate file
(Base/Collector/Emitter, In+/In-/Out, Drain/Gate/Source). That is exactly the
supervision a port-identity model needs, and this project needs one, because
terminal ORDER is currently decided from wire geometry and cannot read an
arrowhead or a +/- glyph.

The problem: those crops were cut from the archive's 1,277 source photographs,
and 192 of those are this project's TEST split. There is no filename provenance
and no manifest, so training on the tree as shipped is training on test data.

Provenance is therefore recovered by image matching, under POSITIVE SELECTION:
keep only crops that can be PROVEN to come from a train-split image -- never
"everything except what matched test". If matching is imperfect, positive
selection loses training data (recoverable); exclusion leaks contamination
(not recoverable).

Method: a contrast-normalised low-resolution descriptor for every published
crop and every COCO component region in all 1,277 photographs, compared with
one matrix product per dihedral symmetry (the crop set is ~6x the instance
count, i.e. augmented). Best match per crop. The accept threshold is read off
the bimodality of the score distribution, and a negative control -- crops
matched against regions of a DIFFERENT class -- establishes what "no match"
looks like.

Usage:
    python scripts/port_provenance.py --calibrate    # recover the crop convention
    python scripts/port_provenance.py --pad 0.1      # full match
    python scripts/port_provenance.py --pad 0.1 --accept 0.9   # + selection
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

ARCHIVE = ROOT / "data/digitize_hcd/extracted/Digitize-HCD Dataset"
PORT_DIR = ARCHIVE / "Component Port Location Data"
COCO = ARCHIVE / "Component Symbol and Text Label Data/component_annotations.json"
RAW = ROOT / "data/raw"
OUT = ROOT / "results/port_provenance"

CLASSES = ["BJT-NPN", "BJT-PNP", "MOSFET-N", "MOSFET-P", "Op-Amp"]
D = 32          # 32x32 = 1024 dims, ample to identify a crop


def descriptor(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (D, D), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    return (g / n).ravel() if n > 1e-6 else np.zeros(D * D, np.float32)


def dihedral_stack(regs: np.ndarray, k: int) -> np.ndarray:
    imgs = regs.reshape(-1, D, D)
    imgs = np.rot90(imgs, k % 4, axes=(1, 2))
    if k >= 4:
        imgs = imgs[:, :, ::-1]
    return np.ascontiguousarray(imgs).reshape(len(regs), -1)


def load_coco():
    d = json.loads(COCO.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i["file_name"] for i in d["images"]}
    by: dict[str, list] = {}
    for a in d["annotations"]:
        by.setdefault(cats[a["category_id"]], []).append((imgs[a["image_id"]], a["bbox"]))
    return by


def split_of() -> dict[str, str]:
    m = {}
    for s in ("train", "val", "test"):
        for n in (ROOT / f"data/splits/{s}.txt").read_text().split():
            m[Path(n).stem] = s
    return m


def region_descriptors(instances, pad, cache):
    vecs, meta = [], []
    for fn, (x, y, w, h) in instances:
        stem = Path(fn).stem
        if stem not in cache:
            p = RAW / fn
            cache[stem] = cv2.imread(str(p)) if p.exists() else None
        img = cache[stem]
        if img is None:
            continue
        H, W = img.shape[:2]
        px, py = w * pad, h * pad
        x0, y0 = max(0, int(x - px)), max(0, int(y - py))
        x1, y1 = min(W, int(x + w + px)), min(H, int(y + h + py))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        vecs.append(descriptor(img[y0:y1, x0:x1]))
        meta.append(stem)
    return (np.stack(vecs) if vecs else np.zeros((0, D * D), np.float32)), meta


def crop_descriptors(cls, limit=None):
    files = sorted((PORT_DIR / cls / "Input Images").glob("*.jpg"))
    if limit:
        files = files[:limit]
    vecs, names = [], []
    for f in files:
        im = cv2.imread(str(f))
        if im is not None:
            vecs.append(descriptor(im))
            names.append(f.name)
    return (np.stack(vecs) if vecs else np.zeros((0, D * D), np.float32)), names


def best_match(crops, regions):
    """Best (score, region idx, transform) per crop over the 8 symmetries.

    Eight matrix products. Looping this per crop is what turns a seconds-long
    job into an apparently intractable 300M-comparison one.
    """
    n = len(crops)
    if n == 0 or len(regions) == 0:
        return np.zeros(n, np.float32), np.full(n, -1), np.full(n, -1)
    best = np.full(n, -2.0, np.float32)
    bidx = np.full(n, -1)
    bk = np.full(n, -1)
    for k in range(8):
        sim = crops @ dihedral_stack(regions, k).T
        idx = sim.argmax(1)
        val = sim[np.arange(n), idx]
        upd = val > best
        best[upd], bidx[upd], bk[upd] = val[upd], idx[upd], k
    return best, bidx, bk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pad", type=float, default=None)
    ap.add_argument("--accept", type=float, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    by_class = load_coco()
    spl = split_of()
    cache: dict = {}

    if args.calibrate or args.pad is None:
        cls = "Op-Amp"
        crops, _ = crop_descriptors(cls, limit=250)
        print(f"calibrating crop convention on {cls}: {len(crops)} crops vs "
              f"{len(by_class[cls])} instances", flush=True)
        best_pad, best_med = None, -2
        for pad in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            regs, _ = region_descriptors(by_class[cls], pad, cache)
            b, _, _ = best_match(crops, regs)
            med = float(np.median(b))
            print(f"  pad={pad:.2f}  median best-cos={med:.4f}", flush=True)
            if med > best_med:
                best_pad, best_med = pad, med
        print(f"  -> chosen pad={best_pad} (median {best_med:.4f})", flush=True)
        pad = best_pad
        if args.calibrate:
            (OUT / "calibration.json").write_text(
                json.dumps({"pad": pad, "median_cos": best_med}, indent=1))
            return
    else:
        pad = args.pad

    rows, hist = [], {}
    for cls in CLASSES:
        crops, names = crop_descriptors(cls)
        regs, rmeta = region_descriptors(by_class[cls], pad, cache)
        b, bi, bk = best_match(crops, regs)
        hist[cls] = [round(float(x), 4) for x in b]
        for nm, s, i, k in zip(names, b, bi, bk):
            stem = rmeta[i] if i >= 0 else None
            rows.append({"cls": cls, "crop": nm, "score": round(float(s), 4),
                         "source": stem, "split": spl.get(stem, "unknown"),
                         "transform": int(k)})
        # negative control: these crops against a DIFFERENT class's regions
        neg, _ = region_descriptors(by_class["Resistor"][:800], pad, cache)
        nb, _, _ = best_match(crops[:400], neg)
        hist[cls + "__neg"] = [round(float(x), 4) for x in nb]
        print(f"{cls:10s} crops={len(crops):5d} inst={len(regs):5d} "
              f"median={np.median(b):.4f} p10={np.percentile(b,10):.4f} "
              f"| neg median={np.median(nb):.4f}", flush=True)

    (OUT / "match_scores.json").write_text(json.dumps(hist))
    (OUT / "matches_raw.json").write_text(json.dumps(rows))
    print(f"\nwrote {len(rows)} crop matches -> {OUT}/matches_raw.json")

    if args.accept is not None:
        keep = [r for r in rows if r["score"] >= args.accept and r["split"] == "train"]
        by = {}
        for r in keep:
            by[r["cls"]] = by.get(r["cls"], 0) + 1
        dist = {}
        for r in rows:
            if r["score"] >= args.accept:
                dist[r["split"]] = dist.get(r["split"], 0) + 1
        (OUT / "train_only_crops.json").write_text(json.dumps(
            {"pad": pad, "accept": args.accept, "per_class": by,
             "accepted_split_distribution": dist, "crops": keep}, indent=1))
        print(f"ACCEPTED (train-only, score>={args.accept}): {len(keep)}  {by}")
        print(f"  split distribution of all accepted matches: {dist}")


if __name__ == "__main__":
    main()
