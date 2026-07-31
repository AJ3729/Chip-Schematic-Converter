#!/usr/bin/env python3
"""Does the noise-blob filter delete wire fragments that were CONNECTING things?

``_filter_blobs`` keeps a component when its area >= ``min_blob_area`` OR its
extent >= ``min_blob_extent``. At the shipped 80 and 30 that still discards, for
example, a 3x25 px stub: area 75, extent 25, both under threshold. At 1024 px a
stroke is ~2.7 px wide, so a 25-px stub is a perfectly ordinary piece of wire.

Counting deleted pixels alone cannot say whether that matters -- most deleted
blobs really are specks, and the filter exists because a pure area threshold
shattered nets. The question is specifically whether a deleted blob was
BRIDGING: sitting between two fragments that are now separated. So for every
blob the filter removes, this dilates it by the bridging distance and counts how
many DISTINCT surviving components it touches. Two or more means removing it
severed a connection, which is a split the filter manufactured.

Reported alongside are the sizes of the bridging blobs, because the fix depends
on which threshold is at fault: if they are long and thin the extent rule is too
strict, if they are small and round the area rule is.

The control is the same count for blobs the filter KEEPS. Any blob dilated far
enough touches two components, so the bridging rate is only meaningful against
the rate among survivors.

Usage:
    python scripts/measure_blob_filter_damage.py --limit 40
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.textmask import detect_text_mask
from schematic2netlist.wires import build_non_wire_mask


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--config", default=None)
    ap.add_argument("--reach", type=int, default=3,
                    help="px to dilate a blob by when asking what it touches")
    ap.add_argument("--out-dir", default="results/blob_filter")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    wcfg = cfg["wires"]
    min_area = wcfg["min_blob_area"]
    min_extent = wcfg.get("min_blob_extent", 15)

    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)

    rows: list[dict] = []
    n_img = 0
    tot = Counter()
    for nm in names:
        stem = Path(nm).stem
        ip = images_dir / nm
        dp = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
        if not (ip.exists() and dp.exists()):
            continue
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        dets = load_cached_detections(
            str(dp), min_confidence=cfg["detect"].get("confidence"))
        tm = detect_text_mask(gray, cfg) if cfg["textmask"]["enabled"] else None
        nwm = build_non_wire_mask(gray, dets, cfg, tm)

        # reproduce extract_wires_ink up to, but not including, the filter
        cand = gray.copy()
        cand[nwm > 0] = 255
        ink = cv2.threshold(
            cand, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ink[nwm > 0] = 0
        from schematic2netlist.wires import _bridge_collinear
        bridged = _bridge_collinear(ink, cfg)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            bridged, connectivity=8)
        keep = np.zeros(num, dtype=bool)
        for i in range(1, num):
            a = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            keep[i] = bool(a >= min_area or max(w, h) >= min_extent)

        # surviving mask, labelled, so we can ask what a blob touches
        surv = np.zeros_like(bridged)
        for i in range(1, num):
            if keep[i]:
                surv[labels == i] = 255
        s_num, s_lab = cv2.connectedComponents(surv, connectivity=8)
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * args.reach + 1, 2 * args.reach + 1))

        for i in range(1, num):
            a = int(stats[i, cv2.CC_STAT_AREA])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            blob = (labels == i).astype(np.uint8) * 255
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            # work in a local window; a full-frame dilate per blob is too slow
            y0, y1 = max(0, y - args.reach - 2), min(bridged.shape[0], y + h + args.reach + 2)
            x0, x1 = max(0, x - args.reach - 2), min(bridged.shape[1], x + w + args.reach + 2)
            sub = blob[y0:y1, x0:x1]
            grown = cv2.dilate(sub, k)
            touched = set(np.unique(s_lab[y0:y1, x0:x1][grown > 0])) - {0}
            if keep[i]:
                # a survivor is part of s_lab itself; discount its own label
                touched -= {int(s_lab[y + h // 2, x + w // 2])}
            rec = {"image": nm, "kept": int(keep[i]), "area": a,
                   "width": w, "height": h, "extent": max(w, h),
                   "n_touched": len(touched)}
            rows.append(rec)
            grp = "kept" if keep[i] else "deleted"
            tot[f"{grp}_n"] += 1
            tot[f"{grp}_px"] += a
            if len(touched) >= 2:
                tot[f"{grp}_bridging"] += 1
        n_img += 1
        if n_img % 10 == 0:
            print(f"  [{n_img}/{len(names)}] blobs={len(rows)}", flush=True)

    dele = [r for r in rows if not r["kept"]]
    kept = [r for r in rows if r["kept"]]
    br = [r for r in dele if r["n_touched"] >= 2]

    print(f"\n=== DOES THE BLOB FILTER SEVER CONNECTIONS? ===")
    print(f"{n_img} frames, thresholds area>={min_area} OR extent>={min_extent}, "
          f"reach {args.reach} px\n")
    print(f"  {'group':10s} {'blobs':>7s} {'ink px':>9s} {'bridging >=2':>13s} "
          f"{'rate':>7s}")
    for lbl, grp in (("deleted", dele), ("kept (control)", kept)):
        nb = sum(1 for r in grp if r["n_touched"] >= 2)
        print(f"  {lbl:10s} {len(grp):7d} {sum(r['area'] for r in grp):9d} "
              f"{nb:13d} {nb/max(len(grp),1):7.2%}")

    if br:
        print(f"\n  the {len(br)} deleted blobs that were BRIDGING:")
        print(f"    median area {int(np.median([r['area'] for r in br]))}, "
              f"median extent {int(np.median([r['extent'] for r in br]))}")
        print(f"    would survive area>=40:    "
              f"{sum(1 for r in br if r['area'] >= 40)}")
        print(f"    would survive extent>=15:  "
              f"{sum(1 for r in br if r['extent'] >= 15)}")
        print(f"    would survive extent>=20:  "
              f"{sum(1 for r in br if r['extent'] >= 20)}")
        print(f"    per frame: {len(br)/max(n_img,1):.2f}")
    print(f"\n  A bridging rate among DELETED blobs that clearly exceeds the")
    print(f"  KEPT rate is the signal; any blob dilated far enough touches two")
    print(f"  components, so the control is what makes the number readable.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "blobs.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": n_img, "min_blob_area": min_area,
        "min_blob_extent": min_extent, "reach_px": args.reach,
        "n_deleted": len(dele), "n_kept": len(kept),
        "deleted_bridging": len(br),
        "deleted_bridging_rate": round(len(br) / max(len(dele), 1), 4),
        "kept_bridging_rate": round(
            sum(1 for r in kept if r["n_touched"] >= 2) / max(len(kept), 1), 4),
        "deleted_px": sum(r["area"] for r in dele),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
