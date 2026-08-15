#!/usr/bin/env python3
"""Convert CGHD images and Pascal VOC annotations into the pipeline's inputs.

Task B3. Produces rectified 1024x1024 frames with the transform recorded, and
annotations projected into frame coordinates under the B2 class map.

CONSTRAINT, from the plan and enforced here: this adapter normalises FORMAT
ONLY. It does not alter a threshold, apply corpus-specific preprocessing, or
select any parameter using CGHD content. It calls the same
`preprocess_image_meta` the Digitize-HCD path calls, with the same frozen
config. If CGHD frames come out worse, that is the finding.

Losslessness is asserted, not assumed: every projected box is unprojected and
compared against the original, and the adapter refuses to write a frame whose
round trip exceeds one pixel.

Usage:
    python adapters/cghd_to_pipeline.py                    # detection pool
    python adapters/cghd_to_pipeline.py --pool netlist
    python adapters/cghd_to_pipeline.py --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.config import load_config          # noqa: E402
from schematic2netlist.preprocess import (                # noqa: E402
    preprocess_image_meta, project_point, unproject_point)

CGHD = ROOT / "data/cghd/extracted"
MAP = ROOT / "spec/class_map_cghd.yaml"
COV = ROOT / "results/cghd_coverage.json"
OUT_IMG = ROOT / "data/cghd_1024/images"
OUT_ANN = ROOT / "data/cghd_1024/annotations"
OUT_TF = ROOT / "data/cghd_1024/transforms.json"

ROUND_TRIP_TOL_PX = 1.0


def load_map() -> tuple[dict, dict]:
    m = yaml.safe_load(MAP.read_text())
    return m["mapping"], m["coarse_groups"]


def target_class(name: str, mapping: dict, pool: str) -> str | None:
    """Pipeline class for a CGHD label, or None if it is not a component.

    `pool` selects the stricter netlist rule or the looser detection rule; the
    difference is only for classes whose box is determinate while their
    electrical treatment is not (see spec/class_map_cghd.yaml).
    """
    e = mapping.get(name)
    if e is None:
        return None
    to = e["to"]
    if to == "NOT_A_COMPONENT":
        return None
    if to in ("OUT_OF_VOCABULARY", "AMBIGUOUS"):
        if pool == "detection" and e.get("detection_ok"):
            return e["detection_ok"]
        return None
    return to            # includes "COARSE:bjt" / "COARSE:fet"


def parse_voc(xml_path: Path) -> tuple[list[dict], tuple[int, int]]:
    r = ET.parse(xml_path).getroot()
    size = r.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height")))
    out = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        out.append({
            "name": o.findtext("name", "").strip(),
            "xyxy": [float(b.findtext(k)) for k in
                     ("xmin", "ymin", "xmax", "ymax")],
        })
    return out, wh


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", choices=["detection", "netlist"],
                    default="detection")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    mapping, _ = load_map()
    cov = json.loads(COV.read_text())
    key = ("evaluable_images_detection" if a.pool == "detection"
           else "evaluable_images_netlist")
    stems = cov[key][: a.limit] if a.limit else cov[key]
    print(f"pool={a.pool}  images={len(stems)}")

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_ANN.mkdir(parents=True, exist_ok=True)
    transforms: dict[str, dict] = {}
    n_ok = n_skip = 0
    worst_rt = 0.0
    reasons: dict[str, int] = {}

    def skip(why: str) -> None:
        nonlocal n_skip
        n_skip += 1
        reasons[why] = reasons.get(why, 0) + 1

    for i, key_ in enumerate(stems, 1):
        drafter, stem = key_.split("/")
        xml = CGHD / drafter / "annotations" / f"{stem}.xml"
        if not xml.exists():
            skip("missing annotation")
            continue
        img_dir = CGHD / drafter / "images"
        src = next((p for p in (img_dir / f"{stem}{e}" for e in
                                (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"))
                    if p.exists()), None)
        if src is None:
            skip("missing image")
            continue

        objs, _ = parse_voc(xml)
        boxes = [o for o in objs
                 if target_class(o["name"], mapping, a.pool) is not None]
        if not boxes:
            skip("no in-vocabulary component")
            continue

        # Annotation-aware preprocessing, exactly as the Digitize-HCD path
        # does it: the boxes guide cropping so a component is never cut off.
        ann = [[b["xyxy"][0], b["xyxy"][1],
                b["xyxy"][2] - b["xyxy"][0], b["xyxy"][3] - b["xyxy"][1]]
               for b in boxes]
        try:
            frame, meta = preprocess_image_meta(str(src), cfg, ann_boxes=ann)
        except Exception as e:                                # noqa: BLE001
            skip(f"preprocess failed: {type(e).__name__}")
            continue
        if frame is None:
            skip("preprocess returned None")
            continue

        # Project, then assert the round trip closes.
        out_objs, bad = [], False
        for b in boxes:
            x0, y0, x1, y1 = b["xyxy"]
            corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            proj = [project_point(meta, x, y) for x, y in corners]
            back = [unproject_point(meta, px, py) for px, py in proj]
            err = max(max(abs(bx - cx), abs(by - cy))
                      for (bx, by), (cx, cy) in zip(back, corners))
            worst_rt = max(worst_rt, err)
            if err > ROUND_TRIP_TOL_PX:
                bad = True
                break
            xs = [p[0] for p in proj]
            ys = [p[1] for p in proj]
            X0, Y0, X1, Y1 = min(xs), min(ys), max(xs), max(ys)
            out_objs.append({
                "cghd_class": b["name"],
                "class": target_class(b["name"], mapping, a.pool),
                "bbox": [(X0 + X1) / 2, (Y0 + Y1) / 2, X1 - X0, Y1 - Y0],
                "bbox_xyxy": [X0, Y0, X1, Y1],
            })
        if bad:
            skip(f"round trip > {ROUND_TRIP_TOL_PX} px")
            continue

        out_stem = f"{drafter}__{stem}"
        if not a.dry_run:
            cv2.imwrite(str(OUT_IMG / f"{out_stem}.jpg"), frame)
            (OUT_ANN / f"{out_stem}.json").write_text(json.dumps({
                "schema_version": 1,
                "image": f"{out_stem}.jpg",
                "source": "cghd_v12_voc",
                "source_path": str(src.relative_to(ROOT)),
                "drafter": int(drafter.split("_")[1]),
                "drawing_group": f"{drafter}__{stem.rsplit('_P', 1)[0]}",
                "picture": int(stem.rsplit("_P", 1)[1]),
                "pool": a.pool,
                "bbox_frame": "cghd_1024",
                "components": out_objs,
            }, indent=1) + "\n")
            transforms[out_stem] = meta
        n_ok += 1
        if i % 100 == 0:
            print(f"  ...{i}/{len(stems)}  ok={n_ok} skipped={n_skip}", flush=True)

    if not a.dry_run:
        OUT_TF.write_text(json.dumps(transforms, indent=1) + "\n")

    print(f"\nconverted {n_ok}, skipped {n_skip}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {v:5d}  {k}")
    print(f"worst box round-trip error: {worst_rt:.4f} px "
          f"(tolerance {ROUND_TRIP_TOL_PX})")
    if not a.dry_run:
        print(f"wrote {OUT_IMG}, {OUT_ANN}, {OUT_TF}")


if __name__ == "__main__":
    main()
