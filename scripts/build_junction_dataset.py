#!/usr/bin/env python3
"""Build a junction-vs-crossover patch dataset from CGHD (M2 / C2).

The oracle attributes the bulk of remaining end-to-end error to wire
connectivity, and the deterministic parameter space is empirically
exhausted (results/sweeps/stitch_guards.csv: sweeping every stitching
guard buys ~+0.02 terminal-pair F1 against +0.47 of headroom). What is
left is the decision that thresholds cannot make: at a place where two
strokes meet, do they CONNECT (junction) or merely CROSS (crossover)?

CGHD annotates exactly that, as two explicit object classes. This
script turns those annotations into training patches.

Two design points matter more than the code:

**Domain match.** Our pipeline consumes binarized ink at a 512-px
canvas, not raw photos. A classifier trained on CGHD photographs would
face a domain gap at inference, so patches are binarized the same way
the pipeline binarizes (Otsu on a locally-normalized grayscale) unless
--raw is given.

**Scale normalization.** A junction annotated at 21 px in a 1600-px
photo is a different object from a 21-px junction in a 512-px frame.
Patches are cropped at a multiple of the annotated box size — so the
crop always contains the intersection plus a comparable amount of
surrounding stroke — and then resized to a fixed square. This is what
lets a model trained on CGHD photos read our preprocessed frames.

Class balance is reported, not silently fixed: junctions outnumber
crossovers by roughly 13:1, which the training script must handle with
weighting or resampling rather than pretending it does not exist.

``--zip`` reads the published CGHD archive directly, decoding each image
in memory and keeping only the cropped patches. The full archive is
3.2 GB and this machine has under 6 GB free, so extracting it is not an
option; streaming turns a 3.5 GB disk cost into roughly 200 MB of
patches.

Usage:
    python scripts/build_junction_dataset.py --root data/cghd/subset
    python scripts/build_junction_dataset.py --zip data/cghd/cghd-zenodo-12.zip \
        --out data/junctions_full
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

CLASSES = ("junction", "crossover")


def find_image(xml_path: Path) -> Path | None:
    """CGHD's XML <filename> extension does not always match what is on
    disk (.jpeg recorded, .jpg stored), so match by stem."""
    img_dir = xml_path.parent.parent / "images"
    for cand in sorted(img_dir.glob(xml_path.stem + ".*")):
        if cand.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            return cand
    return None


def objects_of_interest(xml_bytes: bytes) -> list[tuple[str, tuple[int, int, int, int]]]:
    out = []
    for obj in ET.fromstring(xml_bytes).findall("object"):
        name = (obj.find("name").text or "").strip().lower()
        if name not in CLASSES:
            continue
        bb = obj.find("bndbox")
        box = tuple(int(float(bb.find(k).text))
                    for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((name, box))
    return out


def binarize(gray: np.ndarray) -> np.ndarray:
    """Mirror the pipeline's ink extraction closely enough to transfer:
    flatten illumination, then Otsu, ink = white."""
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)
    _t, binary = cv2.threshold(norm, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def crop_patch(img: np.ndarray, box, context: float, size: int) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * context / 2.0
    if half < 2:
        return None
    X1, Y1 = int(round(cx - half)), int(round(cy - half))
    X2, Y2 = int(round(cx + half)), int(round(cy + half))
    H, W = img.shape[:2]
    # pad rather than clip, so an intersection near the page edge keeps
    # its geometry instead of being silently re-centred
    pad_l, pad_t = max(0, -X1), max(0, -Y1)
    pad_r, pad_b = max(0, X2 - W), max(0, Y2 - H)
    X1, Y1 = max(0, X1), max(0, Y1)
    X2, Y2 = min(W, X2), min(H, Y2)
    if X2 <= X1 or Y2 <= Y1:
        return None
    patch = img[Y1:Y2, X1:X2]
    if any((pad_l, pad_t, pad_r, pad_b)):
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def iter_from_dir(root: Path):
    """Yield (drafter, stem, xml_bytes, image_bytes) from an extracted tree."""
    for xml_path in sorted(root.rglob("*.xml")):
        img_path = find_image(xml_path)
        if img_path is None:
            yield xml_path.parent.parent.name, xml_path.stem, None, None
            continue
        yield (xml_path.parent.parent.name, xml_path.stem,
               xml_path.read_bytes(), img_path.read_bytes())


def zip_inventory(zip_path: Path) -> tuple[list[str], int]:
    """Drafters and annotation count, read from the archive's index only."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        xmls = [n for n in zf.namelist()
                if n.lower().endswith(".xml")
                and PurePosixPath(n).parent.name == "annotations"]
    return sorted({PurePosixPath(n).parent.parent.name for n in xmls}), len(xmls)


def iter_from_zip(zip_path: Path):
    """Yield the same tuples straight out of the published archive.

    Only one image is held in memory at a time, so the 3.2 GB archive
    never lands on disk — which matters on a nearly-full volume.
    """
    import zipfile

    IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        images: dict[tuple[str, str], str] = {}
        for n in names:
            p = PurePosixPath(n)
            if p.suffix.lower() in IMG_EXT and p.parent.name == "images":
                images[(p.parent.parent.name, p.stem)] = n
        xmls = sorted(n for n in names
                      if n.lower().endswith(".xml")
                      and PurePosixPath(n).parent.name == "annotations")
        for n in xmls:
            p = PurePosixPath(n)
            drafter, stem = p.parent.parent.name, p.stem
            img_name = images.get((drafter, stem))
            if img_name is None:
                yield drafter, stem, None, None
                continue
            yield drafter, stem, zf.read(n), zf.read(img_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/cghd/subset")
    ap.add_argument("--zip", default=None,
                    help="read the CGHD archive directly instead of an "
                         "extracted tree (no disk cost for source images)")
    ap.add_argument("--out", default="data/junctions")
    ap.add_argument("--size", type=int, default=64, help="output patch px")
    ap.add_argument("--context", type=float, default=3.0,
                    help="crop size as a multiple of the annotated box")
    ap.add_argument("--raw", action="store_true",
                    help="keep grayscale instead of binarizing (domain-mismatched "
                         "with our pipeline; for comparison only)")
    ap.add_argument("--val-drafters", type=int, default=5,
                    help="hold out this many drafters entirely, so validation "
                         "measures transfer to unseen handwriting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Enumerate drafters WITHOUT reading image bytes, then iterate lazily.
    # Materialising the iterator would hold the whole 3.2 GB archive in
    # memory, which is exactly what streaming is meant to avoid.
    if args.zip:
        source = Path(args.zip)
        drafters, n_records = zip_inventory(source)
        records = iter_from_zip(source)
    else:
        source = Path(args.root)
        xmls = sorted(source.rglob("*.xml"))
        if not xmls:
            raise SystemExit(f"no annotations under {source}")
        drafters = sorted({p.parent.parent.name for p in xmls})
        n_records = len(xmls)
        records = iter_from_dir(source)
    if not n_records:
        raise SystemExit(f"no annotations found in {source}")
    rng = random.Random(args.seed)
    rng.shuffle(drafters)
    val_drafters = set(drafters[: max(1, min(args.val_drafters, len(drafters) - 1))])
    print(f"{len(drafters)} drafters; holding out {sorted(val_drafters)} for val")

    out = Path(args.out)
    for split in ("train", "val"):
        for cls in CLASSES:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    index: list[dict] = []
    skipped = Counter()

    for n_done, (drafter, stem, xml_bytes, img_bytes) in enumerate(records, 1):
        if xml_bytes is None or img_bytes is None:
            skipped["no_image"] += 1
            continue
        objs = objects_of_interest(xml_bytes)
        if not objs:
            continue
        raw = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        if raw is None:
            skipped["unreadable"] += 1
            continue
        img = raw if args.raw else binarize(raw)
        split = "val" if drafter in val_drafters else "train"

        for i, (cls, box) in enumerate(objs):
            patch = crop_patch(img, box, args.context, args.size)
            if patch is None:
                skipped["degenerate_box"] += 1
                continue
            name = f"{drafter}__{stem}__{i}.png"
            cv2.imwrite(str(out / split / cls / name), patch)
            counts[(split, cls)] += 1
            index.append({"file": f"{split}/{cls}/{name}", "split": split,
                          "class": cls, "drafter": drafter,
                          "source": f"{drafter}/{stem}", "box": list(box)})
        if n_done % 250 == 0:
            print(f"  [{n_done}/{n_records}] {sum(counts.values())} patches",
                  flush=True)

    with (out / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    meta = {
        "source": str(source), "patch_size": args.size, "context": args.context,
        "binarized": not args.raw, "seed": args.seed,
        "val_drafters": sorted(val_drafters),
        "counts": {f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
        "skipped": dict(skipped),
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nwrote {sum(counts.values())} patches to {out}")
    for (split, cls), n in sorted(counts.items()):
        print(f"  {split:5s} {cls:10s} {n:6d}")
    tr = {c: counts[("train", c)] for c in CLASSES}
    if all(tr.values()):
        print(f"\ntrain imbalance junction:crossover = "
              f"{tr['junction'] / tr['crossover']:.1f}:1 — the training script "
              f"must weight or resample, not ignore this")
    if skipped:
        print(f"skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
