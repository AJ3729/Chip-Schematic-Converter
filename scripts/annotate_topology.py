#!/usr/bin/env python3
"""Ground-truth topology annotation workflow (Phase B3).

The human-in-the-loop workflow is bootstrap -> correct -> render ->
verify:

1. --bootstrap : run the pipeline on each image and write a pre-filled,
   UNVERIFIED GT JSON to data/gt_netlists/<stem>.json (skips existing
   files). Much faster to correct than annotating from zero.
2. The annotator edits the JSON (fix classes, nets, add missed
   components; ground net must be "0"), optionally marking deliberately
   dangling parts with "unconnected": true.
3. --render : draw an overlay per image (wires colored by inferred
   node, bboxes, terminal labels showing the GT net names) so the
   annotator can visually check their JSON against the drawing.
4. --check : run the schema validator over all GT files; verified files
   are validated strictly.
5. The annotator sets "verified": true and "annotator": "<name>" —
   this sign-off is a human action by design.

Usage:
    python scripts/annotate_topology.py --bootstrap --images data/splits/test.txt
    python scripts/annotate_topology.py --render
    python scripts/annotate_topology.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import (
    bootstrap_from_pipeline,
    gt_to_components,
    load_gt,
    save_gt,
    validate_gt,
)
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline

PALETTE = [
    (0, 200, 0), (0, 128, 255), (255, 0, 128), (200, 200, 0),
    (255, 128, 0), (128, 0, 255), (0, 255, 255), (128, 128, 255),
]


def iter_images(args, cfg) -> list[Path]:
    images_dir = Path(args.images_dir)
    if args.images and Path(args.images).suffix == ".txt":
        names = Path(args.images).read_text().split()
        return [images_dir / n for n in names]
    if args.images:
        return [Path(p) for p in args.images.split(",")]
    return sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )


def cmd_bootstrap(args, cfg, gt_dir: Path) -> int:
    det_dir = Path(cfg["detect"]["cache_dir"])
    made, skipped_existing, missing = 0, 0, []
    for img_path in iter_images(args, cfg):
        out = gt_dir / (img_path.stem + ".json")
        if out.exists() and not args.force:
            skipped_existing += 1
            continue
        det_path = det_dir / (img_path.stem + ".json")
        if not det_path.exists():
            missing.append(img_path.name)
            continue
        detections = load_cached_detections(det_path)
        result = run_pipeline(img_path, cfg, detections=detections)
        save_gt(bootstrap_from_pipeline(img_path.name, result), out)
        made += 1
    print(f"[OK] bootstrapped {made} GT file(s) into {gt_dir} "
          f"({skipped_existing} existing kept)")
    if missing:
        print(f"[WARN] {len(missing)} image(s) skipped — no cached detections: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    return 0


def cmd_render(args, cfg, gt_dir: Path) -> int:
    render_dir = gt_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    n = 0
    for gt_path in sorted(gt_dir.glob("*.json")):
        gt = load_gt(gt_path)
        img_path = images_dir / gt["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot load {img_path}, skipping render")
            continue
        # color terminals by GT net so mislabeled nets stand out
        nets = sorted(
            {t["net"] for c in gt["components"] for t in c["terminals"] if t["net"]}
        )
        color_of = {net: PALETTE[i % len(PALETTE)] for i, net in enumerate(nets)}
        for c in gt["components"]:
            det = {"x": c["bbox"][0], "y": c["bbox"][1],
                   "width": c["bbox"][2], "height": c["bbox"][3]}
            x1, y1, x2, y2 = bbox_xyxy(det)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(img, f"{c['id']}:{c['class'][:10]}", (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            for t in c["terminals"]:
                tx = x1 + 6 + 14 * t["index"]
                ty = (y1 + y2) // 2
                net = t["net"]
                col = color_of.get(net, (0, 0, 255))
                cv2.circle(img, (tx, ty), 5, col, -1)
                cv2.putText(img, str(net), (tx - 4, ty + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        status = "VERIFIED" if gt.get("verified") else "unverified"
        cv2.putText(img, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 128, 0) if gt.get("verified") else (0, 0, 255), 1)
        cv2.imwrite(str(render_dir / (gt_path.stem + ".png")), img)
        n += 1
    print(f"[OK] rendered {n} overlay(s) into {render_dir}")
    return 0


def cmd_check(args, cfg, gt_dir: Path) -> int:
    whitelist = set(cfg["wires"]["non_wire_classes"]) if args.whitelist else None
    total, verified, bad = 0, 0, 0
    for gt_path in sorted(gt_dir.glob("*.json")):
        gt = load_gt(gt_path)
        total += 1
        verified += bool(gt.get("verified"))
        issues = validate_gt(gt, class_whitelist=whitelist)
        if issues:
            bad += 1
            print(f"[FAIL] {gt_path.name}:")
            for issue in issues:
                print(f"       - {issue}")
        # loader round-trip must always work
        gt_to_components(gt)
    print(f"\n[SUMMARY] {total} GT file(s): {verified} verified, "
          f"{total - verified} unverified, {bad} with validation issues")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--images", default=None,
                    help="split .txt file or comma-separated image paths "
                    "(default: every image in --images-dir)")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--gt-dir", default="data/gt_netlists")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true",
                    help="bootstrap: overwrite existing GT files")
    ap.add_argument("--whitelist", action="store_true",
                    help="check: enforce the configured class whitelist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_dir = Path(args.gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    if args.bootstrap:
        sys.exit(cmd_bootstrap(args, cfg, gt_dir))
    if args.render:
        sys.exit(cmd_render(args, cfg, gt_dir))
    sys.exit(cmd_check(args, cfg, gt_dir))


if __name__ == "__main__":
    main()
