#!/usr/bin/env python3
"""Crossing patches from PHOTO-LIKE renders pushed through the real pipeline.

Why this exists. The previous generator drew clean synthetic ink and used
it directly as a wire mask, so it could never contain the artifacts that
actually characterise a real mask, because real masks are the OUTPUT of

    photo -> shadow removal -> Otsu -> morphology -> speck removal
          -> component erase -> stitching

Tuning the synthetic ink's density and blob radius to imitate that output
was fitting summary statistics of a process that can simply be RUN. It
narrowed the gap (ink Cohen's d 1.79 -> 0.82) but a classifier trained on
it still scored chance on real masks (AUC 0.509 against 0.909 in-domain).

Here nothing is faked that can be taken from the data:

- **Component symbols are real crops** lifted from train/val photographs
  at their annotated boxes, pasted at their true positions, so symbol
  appearance, stroke texture and paper tone are genuine.
- **Paper is real background** sampled from the same photographs away
  from any annotation, carrying real texture, shading and sensor noise.
- **Wires are synthetic** — they must be, since connectivity ground truth
  exists only for the 190 test images and training on those would
  invalidate the benchmark. They are drawn in pen-like grey with pressure
  variation onto the photo, not onto a binary canvas.
- **The mask is produced by the real code.** The composite goes through
  `preprocess_image_meta` and `extract_wires`/`stitch_wire_islands` from
  the shipped pipeline, at the configured `target_size`, so every artifact
  the classifier will meet at inference is present by construction.

Labels come from the known synthetic net structure, read at the sites the
pipeline's own detector reports on the resulting mask — so the training
population equals the inference population.

Usage:
    python scripts/build_crossing_dataset_real.py --limit 60 --rounds 2 \
        --out data/crossings_real
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import scripts.build_crossing_dataset as B
from schematic2netlist.config import load_config
from schematic2netlist.preprocess import preprocess_image_meta
from schematic2netlist.skeleton import crop_site, intersection_sites_with_degree
from schematic2netlist.wires import (
    build_non_wire_mask, extract_wires, stitch_wire_islands, stitchable_mask)

COCO = B.COCO


def load_raw_layouts(coco_path, split_file):
    """(file_name, [(class, [x,y,w,h]) ...]) in ORIGINAL photo coordinates."""
    coco = json.load(open(coco_path))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_name = {i["file_name"]: i["id"] for i in coco["images"]}
    info = {i["id"]: i for i in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    out = []
    for name in (l.strip() for l in open(split_file) if l.strip()):
        iid = by_name.get(name)
        if iid is None:
            continue
        boxes = [(cats[a["category_id"]], list(a["bbox"]))
                 for a in anns[iid]
                 if cats[a["category_id"]] != "Wire Crossover"]
        if len(boxes) >= 3:
            out.append((name, boxes, info[iid]["width"], info[iid]["height"]))
    return out


def paper_and_symbols(raw_dir, name, boxes, rng):
    """Real paper background with real symbol crops pasted back on it.

    The photo already contains both; the wires are what gets replaced.
    Wire ink is removed by inpainting-by-median so the paper keeps its
    real texture and shading rather than becoming flat grey.
    """
    img = cv2.imread(str(Path(raw_dir) / name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    H, W = img.shape
    # background estimate: heavy median blur removes strokes, keeps shading
    paper = cv2.medianBlur(img, 31)
    canvas = paper.copy()
    # Ink darkness must come from the ORIGINAL photo, not the blurred paper.
    # Taking it from the blur gave strokes at 193-243 against paper at 253,
    # which Otsu discarded as background — the first smoke test produced
    # composites with real symbols and NO wires (mask ink fraction 0.000).
    ink_level = float(np.percentile(img, 2))
    kept = []
    for cls, (x, y, w, h) in boxes:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        canvas[y1:y2, x1:x2] = img[y1:y2, x1:x2]      # real symbol pixels
        kept.append((cls, [x1, y1, x2 - x1, y2 - y1]))
    return canvas, kept, ink_level


def draw_pen(canvas, pts, thickness, rng, ink_level):
    """Pen-like stroke at the photo's own ink darkness, with pressure
    variation. ``ink_level`` is measured from the original photograph, so a
    faint pencil drawing gets faint wires and a bold pen gets bold ones."""
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        ink = max(0, min(255, int(ink_level) + rng.randint(-25, 25)))
        t = max(1, thickness + rng.choice((-1, 0, 0, 1)))
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), ink, t,
                 lineType=cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--context", type=float, default=3.0)
    ap.add_argument("--touch-rate", type=float, default=0.6)
    ap.add_argument("--merge-q", type=float, default=0.4)
    ap.add_argument("--p-dot", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--debug-dir", default=None,
                    help="write a few composites + masks for inspection")
    ap.add_argument("--out", default="data/crossings_real")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frame = cfg["preprocess"]["target_size"]
    half = max(4, int(round(args.size * args.context / 8)))
    out = Path(args.out)
    for split in ("train", "val"):
        for cls in ("junction", "crossover"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)
    dbg = Path(args.debug_dir) if args.debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)

    counts: dict = defaultdict(int)
    index: list[dict] = []
    for split in ("train", "val"):
        layouts = load_raw_layouts(COCO, f"data/splits/{split}.txt")
        layouts = layouts[args.offset:][: args.limit]
        print(f"{split}: {len(layouts)} photos x {args.rounds} rounds",
              flush=True)
        for li, (name, boxes, W0, H0) in enumerate(layouts):
            stem = Path(name).stem
            for rd in range(args.rounds):
                rng = random.Random((args.seed, split, stem, rd).__hash__())
                canvas, kept, ink_level = paper_and_symbols(
                    args.raw_dir, name, boxes, rng)
                if canvas is None or len(kept) < 3:
                    continue

                # Route on a FRAME-sized grid, not the full photo. Lee maze
                # search is O(cells) and a 2261x1416 photo at stride 3 is
                # ~9x the cells of the 1024 frame, which made a 6-photo smoke
                # test exceed two minutes. Coordinates scale back up when
                # drawing.
                scale = max(H0, W0) / float(frame)
                comps = [{"class": c,
                          "bbox": [(x + w / 2) / scale, (y + h / 2) / scale,
                                   w / scale, h / scale]}
                         for c, (x, y, w, h) in kept]
                nets = B.synth_topology(comps, rng)
                routed = B.route_nets(
                    comps, nets, (int(H0 / scale) + 1, int(W0 / scale) + 1), rng)
                if len(routed) < 2:
                    continue
                B.add_touch_contacts(routed, rng, args.touch_rate)
                # merge a fraction of crossing pairs (their overlaps become
                # junctions) exactly as the binary generator does
                pos_nets = defaultdict(set)
                for net, cells in routed.items():
                    for (y, x, s) in cells:
                        pos_nets[(y // 6, x // 6)].add(net)
                group = {n: n for n in routed}

                def find(n):
                    while group[n] != n:
                        group[n] = group[group[n]]
                        n = group[n]
                    return n

                for ns in pos_nets.values():
                    if len(ns) >= 2:
                        a, b = sorted(ns)[:2]
                        if rng.random() < args.merge_q and find(a) != find(b):
                            group[find(a)] = find(b)
                merged = defaultdict(set)
                for net, cells in routed.items():
                    merged[find(net)] |= cells
                merged = dict(merged)

                # draw wires ON THE PHOTO in pen grey (route coords -> photo)
                t_pen = max(2, int(round(rng.uniform(2.0, 4.0) * scale)))
                for net, cells in merged.items():
                    grid = {(y // s, x // s): s for (y, x, s) in cells}
                    for (r, c), s in grid.items():
                        for dr, dc in ((1, 0), (0, 1)):
                            if (r + dr, c + dc) not in grid:
                                continue
                            if rng.random() < 0.004:        # pen lift
                                continue
                            draw_pen(canvas,
                                     [((c * s + s // 2) * scale,
                                       (r * s + s // 2) * scale),
                                      (((c + dc) * s + s // 2) * scale,
                                       ((r + dr) * s + s // 2) * scale)],
                                     t_pen, rng, ink_level)

                # acquisition effects, then THE REAL PIPELINE
                if rng.random() < 0.5:
                    k = rng.choice((3, 5))
                    canvas = cv2.GaussianBlur(canvas, (k, k), 0)
                tmp = out / f"_tmp_{split}_{stem}_{rd}.jpg"
                cv2.imwrite(str(tmp), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
                res = preprocess_image_meta(str(tmp), cfg,
                                            ann_boxes=[b for _c, b in kept])
                if res is None:
                    tmp.unlink(missing_ok=True)
                    continue
                gray, meta = res
                # project boxes into the frame and mask them as the pipeline does
                from schematic2netlist.preprocess import project_bbox
                dets = []
                for cls, (x, y, w, h) in kept:
                    cx, cy, bw, bh = project_bbox(meta, x, y, w, h)
                    dets.append({"class": cls, "confidence": 1.0, "x": cx,
                                 "y": cy, "width": bw, "height": bh})
                nwm = build_non_wire_mask(gray, dets, cfg, None)
                _cand, mask = extract_wires(gray, nwm, cfg)
                if cfg["wires"].get("stitch_masked_gaps"):
                    mask = stitch_wire_islands(
                        mask, stitchable_mask(gray.shape, dets, cfg, None), cfg)
                tmp.unlink(missing_ok=True)

                # net-id map in FRAME coordinates for labelling
                nmap = np.full(gray.shape, -1, np.int32)
                for net, cells in merged.items():
                    for (y, x, s) in cells:
                        px, py, pw, ph = project_bbox(
                            meta, x * scale, y * scale, s * scale, s * scale)
                        cv2.rectangle(nmap,
                                      (int(px - pw / 2), int(py - ph / 2)),
                                      (int(px + pw / 2), int(py + ph / 2)),
                                      int(net), -1)
                nmap[nwm > 0] = -1

                if dbg and li < 3 and rd == 0:
                    cv2.imwrite(str(dbg / f"{stem}_composite.png"), gray)
                    cv2.imwrite(str(dbg / f"{stem}_mask.png"), mask)

                for si, (x, y, deg, cls) in enumerate(
                        B.label_pipeline_sites(mask, nmap)):
                    patch = crop_site(mask, x, y, half, args.size)
                    if patch is None or (patch > 0).mean() < 0.02:
                        continue
                    fn = f"{stem}__r{rd}__{si}.png"
                    cv2.imwrite(str(out / split / cls / fn), patch)
                    counts[(split, cls)] += 1
                    index.append({"file": f"{split}/{cls}/{fn}", "split": split,
                                  "class": cls, "drafter": stem,
                                  "source": stem, "box": [x, y],
                                  "degree": deg})
            if (li + 1) % 20 == 0:
                print(f"  [{li+1}/{len(layouts)}] {sum(counts.values())} "
                      f"patches", flush=True)

    if args.no_index:
        print(f"shard done: {sum(counts.values())} patches")
        return
    with (out / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    (out / "dataset_meta.json").write_text(json.dumps({
        "source": "real paper + real symbol crops + synthetic wires, "
                  "pushed through the shipped preprocess/extract_wires",
        "frame": frame, "patch_size": args.size, "context": args.context,
        "rounds": args.rounds, "touch_rate": args.touch_rate,
        "p_dot": args.p_dot, "seed": args.seed,
        "counts": {f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
        "test_split_touched": False,
    }, indent=2) + "\n")
    print(json.dumps({f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
                     indent=2))


if __name__ == "__main__":
    main()
