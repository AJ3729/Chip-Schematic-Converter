#!/usr/bin/env python3
"""Are welded nets actually TOUCHING, or did downsampling merge them?

The dominant connectivity failure is two ground-truth nets sharing one
predicted node, and max-flow on the skeleton says 39 of 56 pairwise welds have
both nets reaching the SAME arm -- one continuous conductor with no branch
point, so no cut separates them. That was read as an information limit. But
information limits have causes, and there is an obvious candidate that has
never been tested: the pipeline renders a ~1907x2453 photograph onto a 1024-px
canvas, roughly a 0.52 linear scale. Two strokes 3 px apart in the original
land 1.5 px apart after scaling, and binarization plus the anisotropic
along-stroke closing then fuse them.

If that is what happens, the fusion is self-inflicted rather than intrinsic --
and the 512-to-1024 change already moved strict success 0.305 -> 0.353, which
is the trend that hypothesis predicts.

Testing it needs no new frames. For each welded node, the region it occupies is
unprojected to original-image coordinates with ``preprocess.unproject_point``,
the ORIGINAL ink is thresholded there, and the connected-component count is
compared with the same region in the pipeline's own mask. More components in
the original means the strokes were separate before scaling and the pipeline
merged them.

Two controls, because a raw photograph is noisier than a cleaned frame and
would show more components for uninteresting reasons:

  correct nodes   the same count on nodes carrying exactly ONE gt net. If
                  those also split in the original, the extra components are
                  photograph noise rather than recovered separation.
  ink fraction    reported per region, so a wildly different threshold
                  outcome is visible instead of silently inflating the count.

Usage:
    python scripts/measure_resolution_fusion.py --limit 25
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.preprocess import unproject_point

ORIG = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
        "Component Symbol and Text Label Data/Circuit Diagram Images")


def region_components(img_gray, ink, x1, y1, x2, y2, min_area):
    """Connected components of ink inside a region, ignoring specks."""
    H, W = ink.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    sub = ink[y1:y2, x1:x2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    keep = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area)
    return {"n_components": keep, "ink_frac": float((sub > 0).mean()),
            "area_px": int((x2 - x1) * (y2 - y1))}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--config", default=None)
    ap.add_argument("--orig-dir", default=ORIG)
    ap.add_argument("--out-dir", default="results/resolution_fusion")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf = json.loads(Path("data/transforms_1024.json").read_text())

    bench = {r["image"]: r for r in csv.DictReader(
        open("results/benchmark_1024/seed0/per_image.csv"))}
    names = sorted(im for im, r in bench.items()
                   if int(r["unmatched_gt"]) == 0)[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)

    rows = []
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        meta = tf.get(stem)
        op = Path(args.orig_dir) / nm
        if meta is None or not op.exists():
            continue
        orig = cv2.imread(str(op), cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        _t, orig_ink = cv2.threshold(orig, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        gcomps = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            f"{cfg['detect']['cache_dir']}/{stem}.json",
            min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(images_dir / nm, cfg, detections=dets)
        node_map, wires = res["node_map"], res["clean_wires"]
        wink = (wires > 0).astype(np.uint8) * 255

        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [res["detections"][c["id"]]["x"],
                          res["detections"][c["id"]]["y"],
                          res["detections"][c["id"]]["width"],
                          res["detections"][c["id"]]["height"]]}
                for c in res["components"]]
        p, g, _ = align_components(pred, gcomps)
        pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)
        name_to_id = {}
        for c in res["components"]:
            for n, nn in zip(c.get("nodes", []), c.get("node_names", [])):
                if n is not None and nn is not None:
                    name_to_id[nn] = int(n)
        pred_of, gt_of = {}, {}
        for c in pc:
            for k, n in enumerate(c["nets"]):
                pred_of[(c["id"], k)] = n
        for c in gc:
            for k, net in enumerate(c["nets"]):
                gt_of[(c["id"], k)] = net
        load = defaultdict(set)
        for t, pn in pred_of.items():
            gn = gt_of.get(t)
            if pn is not None and gn is not None:
                load[pn].add(gn)

        inv = 1.0 / max(meta["scale"], 1e-9)
        for pn, nets in load.items():
            nid = name_to_id.get(pn)
            if nid is None:
                continue
            ys, xs = np.nonzero(node_map == nid)
            if ys.size < 40:
                continue
            # a compact window around the node's densest area, so the region
            # is comparable between scales instead of spanning the page
            cy, cx = int(np.median(ys)), int(np.median(xs))
            half = 60
            c1024 = region_components(None, wink, cx-half, cy-half,
                                      cx+half, cy+half, min_area=12)
            if c1024 is None:
                continue
            ox, oy = unproject_point(meta, cx, cy)
            oh = half * inv
            # scale the speck filter by area so the same physical speck is
            # filtered at both resolutions
            corig = region_components(orig, orig_ink, ox-oh, oy-oh,
                                      ox+oh, oy+oh,
                                      min_area=max(12, int(12 * inv * inv)))
            if corig is None:
                continue
            rows.append({"image": nm, "pred_node": pn,
                         "n_gt_nets": len(nets),
                         "cc_1024": c1024["n_components"],
                         "cc_orig": corig["n_components"],
                         "ink_1024": round(c1024["ink_frac"], 4),
                         "ink_orig": round(corig["ink_frac"], 4)})
        if i % 5 == 0:
            print(f"  [{i}/{len(names)}] regions={len(rows)}", flush=True)

    welded = [r for r in rows if r["n_gt_nets"] >= 2]
    clean = [r for r in rows if r["n_gt_nets"] == 1]
    print(f"\n=== DOES THE ORIGINAL RESOLUTION SEPARATE WHAT 1024 FUSED? ===")
    print(f"{len(rows)} node regions: {len(welded)} welded, "
          f"{len(clean)} correct (control)\n")
    print(f"  {'group':22s} {'n':>5s} {'cc@1024':>8s} {'cc@orig':>8s} "
          f"{'delta':>7s} {'more in orig':>13s}")
    for lbl, grp in (("WELDED (>=2 gt nets)", welded),
                     ("correct (1 gt net)", clean)):
        if not grp:
            continue
        a = np.mean([r["cc_1024"] for r in grp])
        b = np.mean([r["cc_orig"] for r in grp])
        more = sum(1 for r in grp if r["cc_orig"] > r["cc_1024"])
        print(f"  {lbl:22s} {len(grp):5d} {a:8.2f} {b:8.2f} {b-a:+7.2f} "
              f"{more:6d} ({more/len(grp):5.1%})")
    print(f"\n  ink fraction (a sanity check on thresholding):")
    for lbl, grp in (("welded", welded), ("correct", clean)):
        if grp:
            print(f"    {lbl:8s} 1024 {np.mean([r['ink_1024'] for r in grp]):.4f}"
                  f"   orig {np.mean([r['ink_orig'] for r in grp]):.4f}")
    print(f"\n  The CONTROL is what makes this readable. If correct nodes also")
    print(f"  gain components in the original, the extra ones are photograph")
    print(f"  noise, not recovered separation. Only a gap between the two")
    print(f"  groups supports downsampling as the cause of the fusion.")

    if rows:
        with (out / "regions.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_regions": len(rows), "n_welded": len(welded), "n_clean": len(clean),
        "welded_cc_1024": round(float(np.mean([r["cc_1024"] for r in welded])), 3)
        if welded else None,
        "welded_cc_orig": round(float(np.mean([r["cc_orig"] for r in welded])), 3)
        if welded else None,
        "clean_cc_1024": round(float(np.mean([r["cc_1024"] for r in clean])), 3)
        if clean else None,
        "clean_cc_orig": round(float(np.mean([r["cc_orig"] for r in clean])), 3)
        if clean else None,
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
