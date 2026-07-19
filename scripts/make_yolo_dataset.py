#!/usr/bin/env python3
"""Build the Ultralytics YOLO dataset from the published COCO
annotations and the frozen splits (Phase C1).

Labels are generated on data/raw images (the coordinate frame of the
published annotations — data/cleaned went through an aggressive
binarizing transform, whose effect is studied separately in ablation
E2). Images are symlinked, not copied. All 17 published classes are
kept, including Wire Crossover (it is part of the detection benchmark
even though it is excluded from topology GT).

Output:
    data/yolo/dataset.yaml
    data/yolo/images/{train,val,test}/<name>.jpg   (symlinks)
    data/yolo/labels/{train,val,test}/<name>.txt   (class cx cy w h, normalized)

Usage:
    python scripts/make_yolo_dataset.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

COCO_PATH = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/component_annotations.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", default=COCO_PATH)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", default="data/yolo")
    args = ap.parse_args()

    coco = json.load(open(args.coco))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_index = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    images = {i["id"]: i for i in coco["images"]}
    by_name = {i["file_name"]: i["id"] for i in coco["images"]}
    anns: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    out = Path(args.out_dir)
    raw_dir = Path(args.raw_dir).resolve()
    total_boxes = 0
    for split in ("train", "val", "test"):
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        split_names = (Path(args.splits_dir) / f"{split}.txt").read_text().split()
        for name in split_names:
            iid = by_name[name]
            info = images[iid]
            W, H = info["width"], info["height"]
            link = img_out / name
            if not link.exists():
                link.symlink_to(raw_dir / name)
            lines = []
            for a in anns[iid]:
                x, y, w, h = a["bbox"]
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(
                    f"{class_index[a['category_id']]} "
                    f"{cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}"
                )
                total_boxes += 1
            (lbl_out / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
        print(f"[OK] {split}: {len(split_names)} images")

    yaml_text = (
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (out / "dataset.yaml").write_text(yaml_text)
    print(f"[OK] {total_boxes} boxes; dataset.yaml written to {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
