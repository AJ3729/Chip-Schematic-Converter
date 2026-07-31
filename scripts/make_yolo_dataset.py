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
    ap.add_argument("--frame", choices=["raw", "cleaned"], default="cleaned",
                    help="coordinate frame for labels/images (Day-1 decision: "
                    "the benchmark standardizes on 'cleaned'; boxes are "
                    "projected via data/transforms.json)")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cleaned-dir", default="data/cleaned")
    ap.add_argument("--transforms", default="data/transforms.json")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", default=None,
                    help="default: data/yolo_<frame>")
    ap.add_argument("--include-text", action="store_true",
                    help="add the published text annotations as an 18th "
                         "class 'Text'. Measured motivation: the heuristic "
                         "text mask fully misses 10.5%% of text boxes "
                         "(48%% of test images affected), and every "
                         "unmasked label enters the wire mask as a phony "
                         "wire. Detection-based masking is the fix.")
    ap.add_argument("--text-json",
                    default=("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
                             "Component Symbol and Text Label Data/"
                             "text_annotations.json"))
    args = ap.parse_args()

    coco = json.load(open(args.coco))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_index = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    text_by_stem: dict[str, list] = {}
    if args.include_text:
        tdata = json.load(open(args.text_json))
        text_by_stem = {Path(e["file_name"]).stem: e["instances"]
                        for e in tdata["data_list"]}
        TEXT_CLASS = len(names)
        names.append("Text")

    images = {i["id"]: i for i in coco["images"]}
    by_name = {i["file_name"]: i["id"] for i in coco["images"]}
    anns: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    cleaned = args.frame == "cleaned"
    if cleaned:
        from schematic2netlist.preprocess import project_bbox
        transforms = json.load(open(args.transforms))

    out = Path(args.out_dir) if args.out_dir else Path(f"data/yolo_{args.frame}")
    img_src = Path(args.cleaned_dir if cleaned else args.raw_dir).resolve()
    total_boxes, dropped = 0, 0
    for split in ("train", "val", "test"):
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        split_names = (Path(args.splits_dir) / f"{split}.txt").read_text().split()
        for name in split_names:
            iid = by_name[name]
            info = images[iid]
            stem = Path(name).stem
            if cleaned:
                meta = transforms.get(stem)
                if meta is None:
                    continue
                W = H = meta["target_size"]        # 512 canvas
            else:
                W, H = info["width"], info["height"]
            link = img_out / name
            if not link.exists():
                link.symlink_to(img_src / name)
            lines = []
            for a in anns[iid]:
                x, y, w, h = a["bbox"]
                if cleaned:
                    cx, cy, bw, bh = project_bbox(meta, x, y, w, h)
                else:
                    cx, cy, bw, bh = x + w / 2, y + h / 2, w, h
                # clip to frame; drop boxes projected fully out of view
                if cx < 0 or cy < 0 or cx > W or cy > H or bw <= 0 or bh <= 0:
                    dropped += 1
                    continue
                lines.append(
                    f"{class_index[a['category_id']]} "
                    f"{cx / W:.6f} {cy / H:.6f} {bw / W:.6f} {bh / H:.6f}"
                )
                total_boxes += 1
            for inst in text_by_stem.get(stem, []) if args.include_text else []:
                x1, y1, x2, y2 = inst["bbox"]
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                if cleaned:
                    cx, cy, bw, bh = project_bbox(meta, x, y, w, h)
                else:
                    cx, cy, bw, bh = x + w / 2, y + h / 2, w, h
                if cx < 0 or cy < 0 or cx > W or cy > H or bw <= 0 or bh <= 0:
                    dropped += 1
                    continue
                lines.append(
                    f"{TEXT_CLASS} "
                    f"{cx / W:.6f} {cy / H:.6f} {bw / W:.6f} {bh / H:.6f}"
                )
                total_boxes += 1
            (lbl_out / (stem + ".txt")).write_text(
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
    print(f"[OK] {total_boxes} boxes ({dropped} dropped out-of-frame); "
          f"frame={args.frame}; dataset.yaml written to {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
