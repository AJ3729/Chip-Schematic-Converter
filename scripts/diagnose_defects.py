#!/usr/bin/env python3
"""What EXACTLY is wrong on the images that are two decisions from correct?

The gating analysis (``results/blockers/strict_blockers.json``) leaves 85
detection-clean images that are not strict, carrying a median of 2
connectivity defects, and 45 of them carry at most 2. Those 45 are the
reachable population: fixing one or two net decisions each converts them to
strict success. Aggregate rates cannot say what to fix, so this dumps every
individual defect with the geometry needed to name its mechanism.

For each GT net that is not clean:

  WELD   the net's predicted node also carries another GT net's terminals.
         The causal cut test is then run over every skeleton site on that
         node: remove a disk, ask whether the fused nets actually separate.
         Reported is how many sites are single-point cures, and where they
         sit relative to component boxes and detected crossover boxes --
         which distinguishes "two wires cross here" from "ink bridges
         through a component body" from "the stitcher joined two nets".

  SPLIT  the net's terminals landed on several predicted nodes. Reported is
         the gap distance between the nearest points of the two fragments
         and whether a component box or a masked text region sits in the
         gap, which distinguishes an unbridged pen lift from a notch that
         cut the net from text masking that erased a segment.

The output is per-defect rows meant to be counted by mechanism, because a
mechanism that explains 40 of 90 defects is a fix and one that explains 3
is not.

Usage:
    python scripts/diagnose_defects.py --max-defects 2
    python scripts/diagnose_defects.py --max-defects 99 --out-dir results/defects_all
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
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.skeleton import intersection_sites_with_degree


def box_gap(x, y, boxes):
    """Distance from (x, y) to the nearest box edge, 1e9 if no boxes."""
    best = 1e9
    for b in boxes:
        dx = max(abs(x - b["x"]) - b["width"] / 2, 0.0)
        dy = max(abs(y - b["y"]) - b["height"] / 2, 0.0)
        best = min(best, float(np.hypot(dx, dy)))
    return best


def analyse_image(nm, cfg, images_dir):
    stem = Path(nm).stem
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

    pred = [{"id": c["id"], "class": c["class"],
             "nets": list(c.get("node_names", [])),
             "bbox": [res["detections"][c["id"]]["x"],
                      res["detections"][c["id"]]["y"],
                      res["detections"][c["id"]]["width"],
                      res["detections"][c["id"]]["height"]]}
            for c in res["components"]]
    p, g, _ = align_components(pred, gcomps)
    pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)

    # terminal -> predicted net name, and the raw node id behind it
    name_to_id = {}
    for c in res["components"]:
        for n, nn in zip(c.get("nodes", []), c.get("node_names", [])):
            if n is not None and nn is not None:
                name_to_id[nn] = int(n)

    pred_of, gt_terms = {}, defaultdict(list)
    for c in pc:
        for k, n in enumerate(c["nets"]):
            pred_of[(c["id"], k)] = n
    for c in gc:
        for k, net in enumerate(c["nets"]):
            if net is not None:
                gt_terms[net].append((c["id"], k))
    pred_terms = defaultdict(list)
    for t, n in pred_of.items():
        if n is not None:
            pred_terms[n].append(t)

    gn_list, pn_list = sorted(gt_terms), sorted(pred_terms)
    corr = {}
    if gn_list and pn_list:
        cost = np.zeros((len(gn_list), len(pn_list)))
        for i, gn in enumerate(gn_list):
            gs = set(gt_terms[gn])
            for j, pn in enumerate(pn_list):
                cost[i, j] = -len(gs & set(pred_terms[pn]))
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] < 0:
                corr[gn_list[i]] = pn_list[j]

    xover = [d for d in dets if canonical_class(d["class"]) == "Wire Crossover"]
    cboxes = [d for d in dets if canonical_class(d["class"]) != "Wire Crossover"]
    sites = intersection_sites_with_degree((wires > 0).astype(np.uint8))
    out = []

    for gn in gn_list:
        pn = corr.get(gn)
        terms = gt_terms[gn]
        nodes = {pred_of.get(t) for t in terms} - {None}
        if pn is None:
            out.append({"image": nm, "gt_net": gn, "defect": "unmatched"})
            continue
        foreign = [t for t in pred_terms[pn] if t not in set(terms)]
        is_weld, is_split = bool(foreign), len(nodes) > 1

        if is_weld:
            nid = name_to_id.get(pn)
            cures = []
            if nid is not None:
                m = (node_map == nid).astype(np.uint8)
                ys, xs = np.nonzero(m)
                # anchor each involved net to a pixel of the node, then test
                # every site: does cutting HERE separate them?
                anchors = {}
                for t in terms + foreign:
                    cid = t[0]
                    gb = by[cid]["bbox"] if cid in by else None
                    if gb is None or ys.size == 0:
                        continue
                    k = int(np.argmin((xs - gb[0])**2 + (ys - gb[1])**2))
                    anchors.setdefault("own" if t in set(terms) else "foreign",
                                       []).append((xs[k], ys[k]))
                on_node = [(sx, sy, sd) for sx, sy, sd in sites
                           if 0 <= sy < m.shape[0] and 0 <= sx < m.shape[1]
                           and m[sy, sx]]
                for (sx, sy, sd) in on_node:
                    probe = m.copy()
                    cv2.circle(probe, (sx, sy), 7, 0, -1)
                    _n, lab = cv2.connectedComponents(probe, connectivity=8)
                    a = {k: {int(lab[yy, xx]) for xx, yy in v}
                         for k, v in anchors.items()}
                    if (a.get("own") and a.get("foreign")
                            and not (a["own"] & a["foreign"])):
                        cures.append((sx, sy, sd))
                out.append({
                    "image": nm, "gt_net": gn, "defect": "weld",
                    "n_foreign_terminals": len(foreign),
                    "n_sites_on_node": len(on_node),
                    "n_single_cut_cures": len(cures),
                    "cure_degrees": ",".join(str(d) for _x, _y, d in cures[:6]),
                    "cure_min_d_comp": round(min(
                        (box_gap(x, y, cboxes) for x, y, _d in cures),
                        default=-1.0), 1),
                    "cure_min_d_xover": round(min(
                        (box_gap(x, y, xover) for x, y, _d in cures),
                        default=-1.0), 1),
                    "also_split": int(is_split),
                })
        if is_split:
            frag = defaultdict(list)
            for t in terms:
                n = pred_of.get(t)
                if n is not None:
                    frag[n].append(t)
            ids = [name_to_id.get(n) for n in frag]
            ids = [i for i in ids if i is not None]
            gap, gx, gy = -1.0, None, None
            if len(ids) >= 2:
                m1 = np.column_stack(np.nonzero(node_map == ids[0])[::-1])
                m2 = np.column_stack(np.nonzero(node_map == ids[1])[::-1])
                if len(m1) and len(m2):
                    s1 = m1[::max(1, len(m1)//400)]
                    s2 = m2[::max(1, len(m2)//400)]
                    d2 = ((s1[:, None, :] - s2[None, :, :])**2).sum(-1)
                    k = int(d2.argmin())
                    i1, i2 = k // len(s2), k % len(s2)
                    gap = float(np.sqrt(d2.min()))
                    gx = int((s1[i1][0] + s2[i2][0]) / 2)
                    gy = int((s1[i1][1] + s2[i2][1]) / 2)
            out.append({
                "image": nm, "gt_net": gn, "defect": "split",
                "n_fragments": len(frag),
                "gap_px": round(gap, 1),
                "gap_d_comp": round(box_gap(gx, gy, cboxes), 1)
                if gx is not None else -1.0,
                "gap_d_xover": round(box_gap(gx, gy, xover), 1)
                if gx is not None else -1.0,
                "also_weld": int(is_weld),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-defects", type=int, default=2,
                    help="only images carrying at most this many defects")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/defects")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bench = {r["image"]: r for r in csv.DictReader(
        open("results/benchmark_1024/seed0/per_image.csv"))}
    conn = {r["image"]: r for r in csv.DictReader(
        open("results/connectivity_diag/per_image.csv"))}
    strict = lambda r: r["strict_success"] in ("True", "1", "true")
    targets = []
    for im, r in bench.items():
        if int(r["unmatched_gt"]) > 0 or strict(r) or im not in conn:
            continue
        c = conn[im]
        d = int(c["welded"]) + int(c["split"]) + int(c["welded+split"])
        if d <= args.max_defects:
            targets.append(im)
    print(f"{len(targets)} detection-clean, non-strict images with "
          f"<= {args.max_defects} defects")

    names = sorted(targets)
    images_dir = resolve_and_check(None, names, cfg)
    rows = []
    for i, nm in enumerate(names, 1):
        try:
            rows.extend(analyse_image(nm, cfg, images_dir))
        except Exception as e:                     # keep going, record it
            print(f"  [{i}] {nm} FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({"image": nm, "defect": "ERROR", "gt_net": str(e)[:80]})
        if i % 10 == 0:
            print(f"  [{i}/{len(names)}] defects={len(rows)}", flush=True)

    kinds = Counter(r["defect"] for r in rows)
    print(f"\n=== {len(rows)} defects over {len(names)} images ===")
    for k, v in kinds.most_common():
        print(f"  {k:12s} {v:4d}")

    welds = [r for r in rows if r["defect"] == "weld"]
    if welds:
        cur = Counter(r["n_single_cut_cures"] for r in welds)
        print(f"\nWELDS ({len(welds)}): how many single-site cuts cure it?")
        for k in sorted(cur):
            print(f"  {k:2d} cure sites  {cur[k]:4d}  {cur[k]/len(welds):5.1%}")
        curable = [r for r in welds if r["n_single_cut_cures"] > 0]
        print(f"  curable by ONE cut: {len(curable)}/{len(welds)} = "
              f"{len(curable)/len(welds):.1%}")
        if curable:
            print(f"\n  where the cure sites sit (stroke-relative px):")
            dc = [r["cure_min_d_comp"] for r in curable
                  if r["cure_min_d_comp"] >= 0]
            dx = [r["cure_min_d_xover"] for r in curable
                  if r["cure_min_d_xover"] >= 0]
            if dc:
                print(f"    distance to nearest COMPONENT box: "
                      f"median {np.median(dc):.1f}  "
                      f"<=5px: {sum(1 for v in dc if v <= 5)}/{len(dc)}")
            if dx:
                print(f"    distance to nearest CROSSOVER box: "
                      f"median {np.median(dx):.1f}  "
                      f"<=5px: {sum(1 for v in dx if v <= 5)}/{len(dx)}")
            degs = Counter()
            for r in curable:
                for d in str(r.get("cure_degrees", "")).split(","):
                    if d:
                        degs[int(d)] += 1
            print(f"    cure-site degrees: {dict(sorted(degs.items()))}")

    splits = [r for r in rows if r["defect"] == "split"]
    if splits:
        gaps = [r["gap_px"] for r in splits if r["gap_px"] >= 0]
        print(f"\nSPLITS ({len(splits)}): fragment gap distances")
        if gaps:
            print(f"  median {np.median(gaps):.1f}px  "
                  f"<=5px: {sum(1 for v in gaps if v <= 5)}  "
                  f"<=15px: {sum(1 for v in gaps if v <= 15)}  "
                  f">30px: {sum(1 for v in gaps if v > 30)}")
            near_c = [r for r in splits
                      if 0 <= r.get("gap_d_comp", -1) <= 5]
            near_x = [r for r in splits
                      if 0 <= r.get("gap_d_xover", -1) <= 5]
            print(f"  gap within 5px of a COMPONENT box: {len(near_c)}")
            print(f"  gap within 5px of a CROSSOVER box: {len(near_x)}"
                  f"   <- a notch that cut the net")

    keys = sorted({k for r in rows for k in r})
    with (out / "defects.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": len(names), "n_defects": len(rows),
        "kinds": dict(kinds),
        "welds_curable_by_one_cut": sum(
            1 for r in welds if r.get("n_single_cut_cures", 0) > 0),
        "n_welds": len(welds), "n_splits": len(splits),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/defects.csv + summary.json")


if __name__ == "__main__":
    main()
