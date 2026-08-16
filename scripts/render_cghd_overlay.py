#!/usr/bin/env python3
"""Render CGHD frames with detection and extraction overlays.

Built to make the two negative cross-corpus results inspectable rather than
just tabulated: detection transfer is mAP@0.5 0.3445 and only 2 of 165
drawings reconstruct identically across their four photographs. Those numbers
say a failure happened; these renders say what it looked like.

Panels, left to right:
  1. frame        the rectified 1024 frame the pipeline actually reads
  2. detection    CGHD ground-truth boxes (green) vs predictions (red),
                  so a miss and a false positive are told apart on sight
  3. extraction   the conductor skeleton, with snapped terminals marked

`--group` renders all four captures of one physical drawing stacked, which is
the view that shows capture variance directly.

Usage:
    python scripts/render_cghd_overlay.py --limit 12
    python scripts/render_cghd_overlay.py --group drafter_1__C1_D1
    python scripts/render_cghd_overlay.py --worst-groups 6
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

from schematic2netlist.classes import canonical_class  # noqa: E402
from schematic2netlist.config import load_config  # noqa: E402
from schematic2netlist.nodes import bbox_xyxy  # noqa: E402
from schematic2netlist.pipeline import run_pipeline  # noqa: E402
from schematic2netlist import snapping as _snap  # noqa: E402

IMG = ROOT / "data/cghd_1024/images"
ANN = ROOT / "data/cghd_1024/annotations"
CACHE = ROOT / "data/cghd_1024/detections"
INV = ROOT / "results/cghd_capture_invariance.json"
OUT = ROOT / "results/cghd_overlays"

GT_COLOR = (60, 170, 60)      # BGR green  -- CGHD ground truth
PR_COLOR = (60, 60, 220)      # BGR red    -- our prediction
WIRE_COLOR = (200, 120, 40)   # BGR blue   -- conductor skeleton
TERM_COLOR = (30, 200, 240)   # BGR yellow -- snapped terminal


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((26, img.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def draw_boxes(canvas, boxes, color, tag_prefix="", xyxy=True):
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = ([int(v) for v in b["xyxy"]] if xyxy else
                          [int(v) for v in bbox_xyxy(b)])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        t = f"{tag_prefix}{b.get('cls','')}"
        if t.strip():
            cv2.putText(canvas, t, (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def panels_for(stem: str, cfg: dict) -> np.ndarray | None:
    frame = cv2.imread(str(IMG / f"{stem}.jpg"))
    if frame is None:
        return None
    ann = json.loads((ANN / f"{stem}.json").read_text())
    dets = json.loads((CACHE / f"{stem}.json").read_text())

    # panel 1 -- the frame as read
    p1 = label(frame.copy(), f"{stem}  |  rectified 1024 frame")

    # panel 2 -- GT (green) vs prediction (red)
    p2 = frame.copy()
    draw_boxes(p2, [{"xyxy": c["bbox_xyxy"], "cls": c["class"].replace("COARSE:", "")}
                    for c in ann["components"]], GT_COLOR)
    draw_boxes(p2, [{"xyxy": bbox_xyxy(d), "cls": canonical_class(d["class"])}
                    for d in dets], PR_COLOR)
    p2 = label(p2, f"detection  GT={len(ann['components'])} (green)  "
                   f"pred={len(dets)} (red)")

    # panel 3 -- extraction: conductor skeleton + snapped terminals
    res = run_pipeline(IMG / f"{stem}.jpg", cfg, detections=dets)
    p3 = frame.copy()
    nm = res.get("node_map")
    if nm is not None:
        mask = np.asarray(nm) > 0
        p3[mask] = WIRE_COLOR
    # The pipeline does not return terminal coordinates, so recompute the
    # boundary crossings exactly as snapping finds them: expand a window round
    # each detection until enough conductor crossings appear.
    comps = res.get("components") or []
    n_term = 0
    if nm is not None:
        sn_cfg = cfg["snapping"]
        for c in comps:
            i = c.get("id")
            if not isinstance(i, int) or not (0 <= i < len(dets)):
                continue
            x1, y1, x2, y2 = bbox_xyxy(dets[i])
            sites: list = []
            for r in range(sn_cfg["expand_step"],
                           sn_cfg["max_expand"] + 1, sn_cfg["expand_step"]):
                f = _snap._boundary_run_sites(nm, x1 - r, y1 - r, x2 + r, y2 + r)
                if len(f) > len(sites):
                    sites = f
                if len(f) >= len(c.get("node_names") or []):
                    break
            for site in sites:
                try:
                    x, y = int(site[1]), int(site[2])
                except (TypeError, IndexError, ValueError):
                    continue
                cv2.circle(p3, (x, y), 5, TERM_COLOR, -1)
                cv2.circle(p3, (x, y), 5, (20, 20, 20), 1)
                n_term += 1
    nets = {n for c in comps for n in (c.get("node_names") or []) if n is not None}
    p3 = label(p3, f"extraction  components={len(comps)}  nets={len(nets)}  "
                   f"terminals={n_term}")
    return np.hstack([p1, p2, p3])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--group", default=None,
                    help="render all captures of one drawing group")
    ap.add_argument("--worst-groups", type=int, default=None,
                    help="render the N groups with the most distinct topologies")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["preprocess"]["images_dir"] = "data/cghd_1024/images"
    cfg["detect"]["cache_dir"] = str(CACHE.relative_to(ROOT))
    OUT.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for f in sorted(ANN.glob("*.json")):
        groups[json.loads(f.read_text())["drawing_group"]].append(f.stem)

    targets: list[tuple[str, list[str]]] = []
    if a.group:
        if a.group not in groups:
            sys.exit(f"unknown group {a.group}; e.g. {sorted(groups)[0]}")
        targets = [(a.group, sorted(groups[a.group]))]
    elif a.worst_groups:
        inv = json.loads(INV.read_text())["per_group"]
        ranked = sorted(inv.items(),
                        key=lambda kv: (-kv[1]["distinct_topologies"],
                                        -(max(kv[1]["component_counts"])
                                          - min(kv[1]["component_counts"]))))
        for g, _ in ranked[: a.worst_groups]:
            if g in groups:
                targets.append((g, sorted(groups[g])))
    else:
        stems = sorted(s for v in groups.values() for s in v)
        stems = stems[: a.limit] if a.limit else stems
        targets = [(s, [s]) for s in stems]

    n = 0
    for name, stems in targets:
        rows = []
        for s in stems:
            row = panels_for(s, cfg)
            if row is not None:
                rows.append(row)
        if not rows:
            continue
        sheet = np.vstack(rows)
        if a.scale != 1.0:
            sheet = cv2.resize(sheet, None, fx=a.scale, fy=a.scale,
                               interpolation=cv2.INTER_AREA)
        dst = OUT / f"{name}.jpg"
        cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        n += 1
        print(f"  wrote {dst.name}  ({len(rows)} capture(s))", flush=True)

    print(f"\n{n} sheet(s) -> {OUT}")


if __name__ == "__main__":
    main()
