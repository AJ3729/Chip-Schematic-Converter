#!/usr/bin/env python3
"""Move the published port labels onto the frames the pipeline actually sees.

The port head trains to 0.87-0.90 order accuracy on the published crops and
collapses to 0.30 inside the pipeline. The crops are cut from raw photographs;
the pipeline reads `data/cleaned_1024`, which is deskewed, shadow-normalised
and downscaled, so a component that was ~250 px across is now ~50. That is the
whole gap, and no amount of patching the inference side closed it: drawing the
annotation rectangle made it worse (0.571 -> 0.392) and no confidence gate beat
the templates.

So build training data in the target domain instead. For every published crop
that provenance matched to a TRAIN photograph, carry its named port
coordinates through: crop -> (undo the dihedral that made them match) -> the
COCO box in the source photograph -> the 1024 frame. Then cut the sample from
`cleaned_1024` exactly as `port_head._crop` will at inference.

Nothing here touches an eval-split image: the source list is filtered to the
train split before any label is transferred.

Usage:
    python scripts/build_indomain_ports.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.preprocess import project_point   # noqa: E402

AR = ROOT / "data/digitize_hcd/extracted/Digitize-HCD Dataset"
PORT = AR / "Component Port Location Data"
COCO = AR / "Component Symbol and Text Label Data/component_annotations.json"
RAW = ROOT / "data/raw"
CLEAN = ROOT / "data/cleaned_1024"
TRANSFORMS = ROOT / "data/transforms_1024.json"
OUT = ROOT / "data/port_indomain"

CLASSES = ["BJT-NPN", "BJT-PNP", "MOSFET-N", "MOSFET-P", "Op-Amp"]
PORTS = {
    "BJT-NPN": ["Base", "Collector", "Emitter"],
    "BJT-PNP": ["Base", "Collector", "Emitter"],
    "MOSFET-N": ["Drain", "Gate", "Source"],
    "MOSFET-P": ["Drain", "Gate", "Source"],
    "Op-Amp": ["In+", "In-", "Out"],
}
D = 32
MARGIN = 0.12
ACCEPT = 0.80


def descriptor(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (D, D), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    return (g / n).ravel() if n > 1e-6 else np.zeros(D * D, np.float32)


def green_box(im):
    b, g, r = im[:, :, 0].astype(int), im[:, :, 1].astype(int), im[:, :, 2].astype(int)
    ys, xs = np.nonzero((g - np.maximum(b, r)) > 40)
    return None if len(xs) < 20 else (int(xs.min()), int(ys.min()),
                                      int(xs.max()), int(ys.max()))


def fwd_uv(u, v, k):
    """Forward dihedral on NORMALISED coords, matching dihedral_stack().

    dihedral_stack does np.rot90(img, k%4) then optionally img[:, :, ::-1].
    Deriving this by reasoning is how sign errors get in, so main() checks it
    numerically against the actual array op before any label is transferred.
    """
    for _ in range(k % 4):
        u, v = v, 1.0 - u          # one np.rot90
    if k >= 4:
        u = 1.0 - u                # fliplr
    return u, v


def inv_uv(u, v, k):
    if k >= 4:
        u = 1.0 - u
    for _ in range((4 - (k % 4)) % 4):
        u, v = v, 1.0 - u
    return u, v


def _check_transform_math() -> None:
    """Verify fwd_uv against the real array operation, and inv against fwd."""
    rng = np.random.default_rng(0)
    img = rng.random((D, D)).astype(np.float32)
    for k in range(8):
        t = np.rot90(img, k % 4)
        if k >= 4:
            t = t[:, ::-1]
        iy, ix = np.unravel_index(img.argmax(), img.shape)
        ty, tx = np.unravel_index(t.argmax(), t.shape)
        u, v = fwd_uv(ix / (D - 1), iy / (D - 1), k)
        assert abs(u - tx / (D - 1)) < 1e-6 and abs(v - ty / (D - 1)) < 1e-6, \
            f"fwd_uv disagrees with np.rot90/fliplr at k={k}"
        bu, bv = inv_uv(u, v, k)
        assert abs(bu - ix / (D - 1)) < 1e-6 and abs(bv - iy / (D - 1)) < 1e-6, \
            f"inv_uv is not the inverse at k={k}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept", type=float, default=ACCEPT)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    _check_transform_math()
    print("dihedral coordinate math verified against the array ops", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tf = json.loads(TRANSFORMS.read_text())
    train = {Path(n).stem for n in (ROOT / "data/splits/train.txt").read_text().split()}

    d = json.loads(COCO.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i["file_name"] for i in d["images"]}
    inst = collections.defaultdict(list)
    for a in d["annotations"]:
        nm = cats[a["category_id"]]
        if nm in CLASSES:
            stem = Path(imgs[a["image_id"]]).stem
            if stem in train:                     # never look at an eval image
                inst[nm].append((imgs[a["image_id"]], a["bbox"]))

    raw_cache: dict = {}
    report = {}
    for cls in CLASSES:
        # region descriptors for this class, TRAIN images only
        R, meta = [], []
        for fn, bb in inst[cls]:
            if fn not in raw_cache:
                p = RAW / fn
                raw_cache[fn] = cv2.imread(str(p)) if p.exists() else None
            im = raw_cache[fn]
            if im is None:
                continue
            x, y, w, h = [int(v) for v in bb]
            sub = im[y:y + h, x:x + w]
            if sub.size == 0:
                continue
            R.append(descriptor(sub))
            meta.append((fn, (x, y, w, h)))
        if not R:
            continue
        R = np.stack(R)

        C, cname, cbox = [], [], []
        for f in sorted((PORT / cls / "Input Images").glob("*.jpg")):
            im = cv2.imread(str(f))
            if im is None:
                continue
            bb = green_box(im)
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            s = im[y0 + 5:y1 - 5, x0 + 5:x1 - 5]
            if s.size == 0:
                continue
            C.append(descriptor(s))
            cname.append(f.name)
            cbox.append(bb)
        C = np.stack(C)

        best = np.full(len(C), -2.0)
        bidx = np.full(len(C), -1)
        bk = np.full(len(C), -1)
        for k in range(8):
            I = R.reshape(-1, D, D)
            I = np.rot90(I, k % 4, axes=(1, 2))
            if k >= 4:
                I = I[:, :, ::-1]
            sim = C @ np.ascontiguousarray(I).reshape(len(R), -1).T
            j = sim.argmax(1)
            v = sim[np.arange(len(C)), j]
            u = v > best
            best[u], bidx[u], bk[u] = v[u], j[u], k

        Xs, Ys, kept = [], [], 0
        for ci in range(len(C)):
            if best[ci] < args.accept:
                continue
            fn, (bx, by, bw, bh) = meta[bidx[ci]]
            stem = Path(fn).stem
            if stem not in tf or not (CLEAN / f"{stem}.jpg").exists():
                continue
            xy = PORT / cls / "XY Coordinates" / f"{Path(cname[ci]).stem}.txt"
            if not xy.exists():
                continue
            gx0, gy0, gx1, gy1 = cbox[ci]
            pts = {}
            for line in xy.read_text().strip().splitlines():
                parts = line.rsplit(maxsplit=2)
                if len(parts) != 3:
                    continue
                pts[parts[0].strip()] = (float(parts[1]), float(parts[2]))
            if not all(n in pts for n in PORTS[cls]):
                continue

            frame = cv2.imread(str(CLEAN / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            # component box in the 1024 frame
            c0 = project_point(tf[stem], bx, by)
            c1 = project_point(tf[stem], bx + bw, by + bh)
            X0f, Y0f = min(c0[0], c1[0]), min(c0[1], c1[1])
            X1f, Y1f = max(c0[0], c1[0]), max(c0[1], c1[1])
            W, H = X1f - X0f, Y1f - Y0f
            if W < 8 or H < 8:
                continue
            mx, my = W * MARGIN, H * MARGIN
            A0 = max(0, int(X0f - mx)); B0 = max(0, int(Y0f - my))
            A1 = min(frame.shape[1], int(X1f + mx))
            B1 = min(frame.shape[0], int(Y1f + my))
            if A1 - A0 < 8 or B1 - B0 < 8:
                continue

            coords = []
            ok = True
            for nm in PORTS[cls]:
                px, py = pts[nm]
                # fraction inside the annotated box, in CROP orientation
                u = (px - gx0) / max(1, gx1 - gx0)
                v = (py - gy0) / max(1, gy1 - gy0)
                u, v = inv_uv(u, v, int(bk[ci]))     # back to source orientation
                sx, sy = bx + u * bw, by + v * bh
                fx, fy = project_point(tf[stem], sx, sy)
                cu = (fx - A0) / max(1, A1 - A0)
                cv_ = (fy - B0) / max(1, B1 - B0)
                if not (-0.25 <= cu <= 1.25 and -0.25 <= cv_ <= 1.25):
                    ok = False
                    break
                coords.append((float(np.clip(cu, 0, 1)), float(np.clip(cv_, 0, 1))))
            if not ok:
                continue

            g = cv2.resize(frame[B0:B1, A0:A1], (args.size, args.size),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
            g = (g - g.mean()) / (g.std() + 1e-6)
            Xs.append(g)
            Ys.append(np.array(coords, np.float32))
            kept += 1

        if Xs:
            np.savez_compressed(OUT / f"{cls}.npz",
                                X=np.stack(Xs), Y=np.stack(Ys))
        report[cls] = {"crops": len(C), "instances_train": len(R),
                       "matched_at_accept": int((best >= args.accept).sum()),
                       "samples_built": kept}
        print(f"  {cls:10s} crops={len(C):5d} train_inst={len(R):4d} "
              f"matched={int((best>=args.accept).sum()):4d} -> samples={kept}", flush=True)

    (OUT / "report.json").write_text(json.dumps(
        {"accept": args.accept, "size": args.size, "margin": MARGIN,
         "per_class": report}, indent=1) + "\n")
    tot = sum(v["samples_built"] for v in report.values())
    print(f"\nTOTAL in-domain samples: {tot}  -> {OUT}")


if __name__ == "__main__":
    main()
