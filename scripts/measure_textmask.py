#!/usr/bin/env python3
"""Measure text-mask accuracy against the published text annotations.

Unmasked text is not a cosmetic problem: any text stroke that survives
masking enters the wire mask and becomes a phony wire, welding nets and
spawning fake intersections. This script quantifies exactly how much the
shipped heuristic misses, using Digitize-HCD's own text_annotations.json
(1,277 images, polygon + bbox + string per instance) as ground truth.

Metrics, per image and aggregated:

- ink recall      — fraction of INK pixels inside GT text regions that
                    the mask covers. The number that matters: uncovered
                    text ink is what becomes wire.
- box recall      — fraction of GT text boxes >=50% ink-covered.
- boxes missed    — fraction of GT text boxes <10% covered (fully
                    escaped the mask; each is a phony-wire candidate).
- wire damage     — fraction of ink OUTSIDE all GT text regions that the
                    mask wrongly covers (real wire/symbol ink deleted by
                    overzealous masking).

``--write-masks DIR`` additionally rasterizes the projected GT text
regions (slightly dilated) as per-image PNG masks — the input for the
GT-text oracle benchmark, which bounds how much strict success improves
if text masking were perfect.

GT boxes are annotated on the RAW photos; they are projected into the
preprocessed frame via the recorded transforms, exactly as the YOLO
labels and GT netlist boxes are.

Usage:
    python scripts/measure_textmask.py --split test
    python scripts/measure_textmask.py --split test \
        --write-masks data/gt_text_masks_1024
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from schematic2netlist.config import load_config
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.preprocess import project_bbox
from schematic2netlist.textmask import detect_text_mask

TEXT_JSON = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/text_annotations.json")


def load_text_gt(path: str) -> dict[str, list[dict]]:
    data = json.load(open(path))
    return {Path(e["file_name"]).stem: e["instances"] for e in data["data_list"]}


def gt_text_region(shape, instances, meta, pad: int) -> np.ndarray:
    """Rasterize projected GT text bboxes (pad px dilated) into a mask."""
    m = np.zeros(shape, np.uint8)
    H, W = shape
    for inst in instances:
        x1, y1, x2, y2 = inst["bbox"]
        cx, cy, bw, bh = project_bbox(meta, x1, y1, x2 - x1, y2 - y1)
        X1 = max(0, int(round(cx - bw / 2)) - pad)
        Y1 = max(0, int(round(cy - bh / 2)) - pad)
        X2 = min(W, int(round(cx + bw / 2)) + pad)
        Y2 = min(H, int(round(cy + bh / 2)) + pad)
        if X2 > X1 and Y2 > Y1:
            m[Y1:Y2, X1:X2] = 255
    return m


def ink_of(gray: np.ndarray) -> np.ndarray:
    """Binarized ink (255 = ink), Otsu — measurement-only binarization."""
    _t, ink = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--text-json", default=TEXT_JSON)
    ap.add_argument("--pad", type=int, default=3,
                    help="dilation (px) around projected GT boxes")
    ap.add_argument("--out-dir", default="results/textmask_eval")
    ap.add_argument("--write-masks", default=None, metavar="DIR",
                    help="also write per-image GT text masks (oracle input)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = [l.strip() for l in
             open(f"data/splits/{args.split}.txt") if l.strip()]
    images_dir = resolve_and_check(args.images_dir, names, cfg)
    transforms = json.load(open(
        cfg["preprocess"].get("transforms_json",
                              "data/transforms_1024.json")))
    text_gt = load_text_gt(args.text_json)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.write_masks) if args.write_masks else None
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gray = cv2.imread(str(images_dir / nm), cv2.IMREAD_GRAYSCALE)
        if gray is None or stem not in transforms:
            continue
        instances = text_gt.get(stem, [])
        meta = transforms[stem]
        gt_region = gt_text_region(gray.shape, instances, meta, args.pad)
        if mask_dir is not None:
            cv2.imwrite(str(mask_dir / f"{stem}.png"), gt_region)

        pred = detect_text_mask(gray, cfg)
        ink = ink_of(gray)

        text_ink = (ink > 0) & (gt_region > 0)
        other_ink = (ink > 0) & (gt_region == 0)
        covered = text_ink & (pred > 0)
        damaged = other_ink & (pred > 0)

        # per-box coverage
        H, W = gray.shape
        box_cov = []
        for inst in instances:
            x1, y1, x2, y2 = inst["bbox"]
            cx, cy, bw, bh = project_bbox(meta, x1, y1, x2 - x1, y2 - y1)
            X1, Y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
            X2, Y2 = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
            if X2 <= X1 or Y2 <= Y1:
                continue
            bi = (ink[Y1:Y2, X1:X2] > 0)
            if bi.sum() == 0:
                continue
            box_cov.append((pred[Y1:Y2, X1:X2] > 0)[bi].mean())

        rows.append({
            "image": nm,
            "n_text_boxes": len(box_cov),
            "ink_recall": round(covered.sum() / max(text_ink.sum(), 1), 4),
            "box_recall_50": round(
                np.mean([c >= 0.5 for c in box_cov]), 4) if box_cov else "",
            "boxes_missed_10": int(sum(c < 0.1 for c in box_cov)),
            "wire_damage": round(damaged.sum() / max(other_ink.sum(), 1), 4),
            "text_ink_px": int(text_ink.sum()),
            "uncovered_text_px": int(text_ink.sum() - covered.sum()),
        })
        if i % 40 == 0:
            print(f"[{i}/{len(names)}]", flush=True)

    with (out_dir / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_boxes = sum(r["n_text_boxes"] for r in rows)
    missed = sum(r["boxes_missed_10"] for r in rows)
    ink_rec = [r["ink_recall"] for r in rows]
    dmg = [r["wire_damage"] for r in rows]
    imgs_with_miss = sum(1 for r in rows if r["boxes_missed_10"] > 0)
    summary = {
        "split": args.split,
        "n_images": len(rows),
        "n_text_boxes": n_boxes,
        "mean_ink_recall": round(float(np.mean(ink_rec)), 4),
        "median_ink_recall": round(float(np.median(ink_rec)), 4),
        "boxes_fully_missed": missed,
        "boxes_fully_missed_frac": round(missed / max(n_boxes, 1), 4),
        "images_with_any_fully_missed_box": imgs_with_miss,
        "mean_wire_damage": round(float(np.mean(dmg)), 4),
        "mean_uncovered_text_px_per_image": round(
            float(np.mean([r["uncovered_text_px"] for r in rows])), 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/per_image.csv + summary.json"
          + (f"; GT masks -> {mask_dir}" if mask_dir else ""))


if __name__ == "__main__":
    main()
