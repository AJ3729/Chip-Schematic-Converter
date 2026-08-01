#!/usr/bin/env python3
"""Render each weld's CUT SITE from the ORIGINAL photograph, for adjudication.

``review_welds.py`` produces the review sheet, but it crops the *pipeline
frame* around the whole conductor path. Two things make that the wrong
evidence for a verdict:

1. The path is long (median detour 1.3, up to 5.1) while the disputed
   connection is ONE site. Looking at the path shows you the route, not the
   decision.
2. The pipeline frame has already destroyed the evidence. 93.5% of its pixels
   are exactly 255 and ink is crushed to median grey ~8, whereas the original
   photographs hold ink median 86, sd 29.3. A hop's gap, or the doubled ink
   where two strokes genuinely cross, survives only in the original.

So this script locates the cut site causally — the intersection whose removal
separates the two fused nets' anchors, the same probe ``locate_welds.py`` uses
— then unprojects it through the recorded preprocessing transform and crops
the ORIGINAL photograph there at full resolution.

Each weld yields one sheet with three panels:

  cut site (original)   the ink as drawn, magnified. This is what the verdict
                        rests on: a U/semicircle detour is a HOP, a plain
                        crossing or T with no detour is not.
  what is disputed      the same window with the merged node tinted and the
                        two nets' anchors marked, so you know which strokes
                        are the ones the pipeline fused.
  context (original)    a wide view with the site boxed, for reading the
                        circuit well enough to spot a GT error.

Welds with no single cutting site (14% of them) are still rendered, centred on
the path midpoint and labelled NO-SINGLE-CUT, because those cannot be repaired
by any one split regardless of verdict.

Usage:
    python scripts/adjudicate_welds.py --out-dir results/weld_adjudication
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from localize_welds import bfs_path

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.preprocess import unproject_point
from schematic2netlist.skeleton import intersection_sites_with_degree
from schematic2netlist.snapping import _boundary_run_sites

TIGHT = 130      # half-window at the cut site, in FRAME px
WIDE = 420       # half-window for the context panel, in FRAME px
PANEL = 660      # rendered panel side


def frame_to_orig(meta, x, y):
    return unproject_point(meta, float(x), float(y))


def orig_window(meta, fx, fy, half):
    """Corners of the original-image window covering a frame-space square."""
    pts = [frame_to_orig(meta, fx - half, fy - half),
           frame_to_orig(meta, fx + half, fy - half),
           frame_to_orig(meta, fx - half, fy + half),
           frame_to_orig(meta, fx + half, fy + half)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def crop_orig(orig, meta, fx, fy, half, side=PANEL):
    x0, y0, x1, y1 = orig_window(meta, fx, fy, half)
    H, W = orig.shape[:2]
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(W, int(x1)), min(H, int(y1))
    if xi1 - xi0 < 8 or yi1 - yi0 < 8:
        return None, None
    sub = orig[yi0:yi1, xi0:xi1]
    s = side / max(sub.shape[:2])
    out = cv2.resize(sub, None, fx=s, fy=s,
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    # a window clipped by the image edge is not square; pad so every panel
    # stacks, and keep the offset so overlays still land correctly
    canvas = np.full((side, side, 3), 255, np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = out
    return canvas, (xi0, yi0, s)


def warp_mask_to_crop(mask, meta, box, shape):
    """Sample a frame-space mask over an original-space crop, exactly."""
    xi0, yi0, s = box
    h, w = shape[:2]
    ox = xi0 + np.arange(w, dtype=np.float64) / s
    oy = yi0 + np.arange(h, dtype=np.float64) / s
    OX, OY = np.meshgrid(ox, oy)
    m = np.asarray(meta["rotation_matrix"], dtype=np.float64)
    XR = m[0, 0] * OX + m[0, 1] * OY + m[0, 2]
    YR = m[1, 0] * OX + m[1, 1] * OY + m[1, 2]
    if meta["rotated90"]:
        wb = meta["size_before_rot90"][0]
        XR, YR = YR, (wb - 1) - XR
    cx, cy = meta["crop"][0], meta["crop"][1]
    sc = meta["scale"]
    ax, ay = meta["canvas_offset"]
    FX = np.rint((XR - cx) * sc + ax).astype(np.int32)
    FY = np.rint((YR - cy) * sc + ay).astype(np.int32)
    ok = ((FX >= 0) & (FX < mask.shape[1]) & (FY >= 0) & (FY < mask.shape[0]))
    out = np.zeros((h, w), bool)
    out[ok] = mask[FY[ok], FX[ok]]
    return out


def label_strip(width, text, sub="", h=62):
    strip = np.full((h, width, 3), 255, np.uint8)
    cv2.putText(strip, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (0, 0, 0), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(strip, sub, (10, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (90, 90, 90), 1, cv2.LINE_AA)
    return strip


def cap(img, text):
    s = np.full((30, img.shape[1], 3), 245, np.uint8)
    cv2.putText(s, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([s, img])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--welds", default="results/weld_review/welds.csv")
    ap.add_argument("--out-dir", default="results/weld_adjudication")
    ap.add_argument("--cut-radius", type=int, default=7)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--transforms", default="data/transforms_1024.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    idir = Path(cfg["preprocess"]["images_dir"])
    tf = json.load(open(args.transforms))
    rows = list(csv.DictReader(open(args.welds)))
    want = defaultdict(list)
    for i, r in enumerate(rows):
        want[r["image"]].append((i, r))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for nm in sorted(want):
        stem = Path(nm).stem
        gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        gc0 = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gc0:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            f"{cfg['detect']['cache_dir']}/{stem}.json",
            min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(str(idir / nm), cfg, detections=dets)
        node_map, comps = res["node_map"], res["components"]
        meta = tf[stem]
        orig = cv2.imread(str(Path(args.raw_dir) / nm))
        if orig is None:
            print(f"  !! no original for {nm}")
            continue

        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [res["detections"][c["id"]]["x"],
                          res["detections"][c["id"]]["y"],
                          res["detections"][c["id"]]["width"],
                          res["detections"][c["id"]]["height"]]}
                for c in comps]
        p, g, _ = align_components(pred, gc0)
        pc, gcn = canonicalize_terminals(p), canonicalize_terminals(g)
        pof, gof = {}, {}
        for c in pc:
            for k, n in enumerate(c["nets"]):
                pof[(c["id"], k)] = n
        for c in gcn:
            for k, n in enumerate(c["nets"]):
                gof[(c["id"], k)] = n
        name_to_id = {}
        for c in comps:
            for n_, nn_ in zip(c.get("nodes", []), c.get("node_names", [])):
                if n_ is not None and nn_ is not None:
                    name_to_id[nn_] = int(n_)
        byp = {c["id"]: c for c in comps}
        cls_of = {c["id"]: canonical_class(res["detections"][c["id"]]["class"])
                  for c in comps}
        # Exact terminal anchors. review_welds.py spread terminal k evenly
        # ACROSS THE BOX WIDTH at the vertical midline, which is only right for
        # a horizontally-drawn component: for a vertical one both terminals
        # land on the same midline and snap to the same pixel, producing a
        # 3-4 px "path" and a meaningless detour. Terminal k's true location is
        # the boundary crossing of the node the snapper assigned to it.
        sn = cfg["snapping"]
        txy = {}
        for c in pc:
            s = byp.get(c["id"])
            if s is None:
                continue
            x1, y1, x2, y2 = bbox_xyxy(res["detections"][s["id"]])
            nodes_k = list(s.get("nodes", []))
            sites = []
            for rr in range(sn["expand_step"], sn["max_expand"] + 1,
                            sn["expand_step"]):
                sites = _boundary_run_sites(node_map, x1 - rr, y1 - rr,
                                            x2 + rr, y2 + rr)
                if len({n for n, _, _ in sites}) >= len(nodes_k):
                    break
            used = set()
            for k in range(len(c["nets"])):
                nid_k = nodes_k[k] if k < len(nodes_k) else None
                cands = [(i, sx, sy) for i, (n_, sx, sy) in enumerate(sites)
                         if nid_k is not None and n_ == nid_k and i not in used]
                if cands:
                    i, sx, sy = cands[0]
                    used.add(i)
                    txy[(c["id"], k)] = (int(sy), int(sx))
                else:
                    n = len(c["nets"])
                    txy[(c["id"], k)] = (int((y1 + y2) / 2),
                                         int(x1 + (k + 1) * (x2 - x1) / (n + 1)))

        onn = defaultdict(lambda: defaultdict(list))
        for t, pn in pof.items():
            gn = gof.get(t)
            if pn is not None and gn is not None and t in txy:
                onn[pn][gn].append(t)

        sites_all = intersection_sites_with_degree((node_map >= 0).astype(np.uint8))

        for idx, r in want[nm]:
            pn, na, nb = r["node"], r["net_a"], r["net_b"]
            nid = name_to_id.get(pn)
            if nid is None or pn not in onn:
                continue
            nets = onn[pn]
            if na not in nets or nb not in nets:
                continue
            m = node_map == nid
            if not m.any():
                continue
            pts = np.argwhere(m)

            def snap(q):
                return tuple(pts[np.argmin(((pts - np.array(q)) ** 2).sum(1))])

            # A weld whose two GT nets meet on the SAME component is not two
            # distant conductors fused — it is one component's own two pins
            # joined, which is a different defect with a different fix.
            shorted = sorted({t[0] for t in nets[na]} & {t[0] for t in nets[nb]})
            kind = "SELF-SHORT" if shorted else "INTER-COMPONENT"

            SA = [snap(txy[t]) for t in nets[na] if t in txy]
            SB = [snap(txy[t]) for t in nets[nb] if t in txy]
            if not SA or not SB:
                continue
            path = bfs_path(m, SA, SB)
            if not path:
                continue
            arr = np.array(path)

            mask8 = m.astype(np.uint8)
            cand = [(x, y, d) for (x, y, d) in sites_all
                    if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]]
            # Keep EVERY cutting site, not the first one the site list happens
            # to yield. Welds routinely have several (up to 9 here), and the
            # arbitrary first is usually a legitimate T-junction while the one
            # that matters is a hop the binarisation bridged.
            cuts = []
            for (sx, sy, deg) in cand:
                probe = mask8.copy()
                cv2.circle(probe, (sx, sy), args.cut_radius, 0, -1)
                nl, lab = cv2.connectedComponents(probe, connectivity=8)
                if nl <= 2:
                    continue
                ga = {lab[a[0], a[1]] for a in SA}
                gb = {lab[b[0], b[1]] for b in SB}
                if 0 in ga or 0 in gb:
                    continue
                if ga.isdisjoint(gb):
                    cuts.append((sx, sy, deg))
            n_cuts = len(cuts)
            cut = (cuts[0][0], cuts[0][1]) if cuts else None
            cut_deg = cuts[0][2] if cuts else None
            if cut is None:
                mid = arr[len(arr) // 2]
                fx, fy = int(mid[1]), int(mid[0])
                tag = "NO-SINGLE-CUT"
            else:
                fx, fy = cut
                tag = ("cut deg-" + "/".join(str(c[2]) for c in cuts[:6]))

            # Panels 1-2 cover the WHOLE disputed conductor, because the
            # decisive feature (a hop the binarisation bridged) can sit
            # anywhere along it. Panel 3 zooms one cut site for stroke detail.
            pcy, pcx = (int((arr[:, 0].min() + arr[:, 0].max()) / 2),
                        int((arr[:, 1].min() + arr[:, 1].max()) / 2))
            phalf = max(int(max(np.ptp(arr[:, 0]), np.ptp(arr[:, 1])) / 2) + 45,
                        110)
            tight, box = crop_orig(orig, meta, pcx, pcy, phalf)
            wide, wbox = crop_orig(orig, meta, fx, fy, 95)
            if tight is None or wide is None:
                continue

            # disputed-strokes panel: node tint + anchors + path, warped exactly
            ann = tight.copy()
            nm_c = warp_mask_to_crop(m, meta, box, ann.shape)
            ann[nm_c] = (0.55 * ann[nm_c] + 0.45 * np.array([255, 205, 205])
                         ).astype(np.uint8)
            pmask = np.zeros_like(m)
            pmask[arr[:, 0], arr[:, 1]] = True
            pmask = cv2.dilate(pmask.astype(np.uint8), np.ones((3, 3), np.uint8)
                               ).astype(bool)
            pm_c = warp_mask_to_crop(pmask, meta, box, ann.shape)
            ann[pm_c] = (0.35 * ann[pm_c] + 0.65 * np.array([0, 210, 255])
                         ).astype(np.uint8)
            xi0, yi0, s = box
            for (ay, ax), col in ([(a, (0, 0, 225)) for a in SA]
                                  + [(b, (0, 160, 0)) for b in SB]):
                ox, oy = frame_to_orig(meta, ax, ay)
                cx_, cy_ = int((ox - xi0) * s), int((oy - yi0) * s)
                if 0 <= cx_ < ann.shape[1] and 0 <= cy_ < ann.shape[0]:
                    cv2.circle(ann, (cx_, cy_), 11, col, -1)
                    cv2.circle(ann, (cx_, cy_), 11, (0, 0, 0), 2)
            for j_, (sx, sy, deg) in enumerate(cuts):
                ox, oy = frame_to_orig(meta, sx, sy)
                cx_, cy_ = int((ox - xi0) * s), int((oy - yi0) * s)
                if 0 <= cx_ < ann.shape[1] and 0 <= cy_ < ann.shape[0]:
                    cv2.drawMarker(ann, (cx_, cy_), (255, 0, 255),
                                   cv2.MARKER_CROSS, 34, 3)
                    cv2.putText(ann, f"{deg}", (cx_ + 15, cy_ - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255),
                                2, cv2.LINE_AA)

            wi0, wj0, ws = wbox
            ctx = wide.copy()
            ox, oy = frame_to_orig(meta, fx, fy)
            cxw, cyw = int((ox - wi0) * ws), int((oy - wj0) * ws)
            cv2.drawMarker(ctx, (cxw, cyw), (255, 0, 255), cv2.MARKER_CROSS,
                           40, 2)

            comps_on = sorted({cls_of[c["id"]] for c in pc
                               for k, n in enumerate(c["nets"])
                               if n == pn and c["id"] in cls_of})
            head = label_strip(
                PANEL * 3 + 12,
                f"#{idx+1}  {nm}   nets {na} vs {nb}   {kind}   [{tag}]"
                f"   cuts {n_cuts}",
                f"detected x-over {r['near_detected_xover'] or 'none'} px | "
                f"GT x-over {r['near_gt_xover'] or 'none'} px | "
                + (f"shorted comp ids {shorted} | " if shorted else "")
                + f"node touches: {', '.join(comps_on)[:90]}")
            panels = np.hstack([
                cap(tight, "1  the whole disputed conductor - original photo"),
                np.full((tight.shape[0] + 30, 6, 3), 200, np.uint8),
                cap(ann, "2  same, annotated (magenta X = every cutting site)"),
                np.full((tight.shape[0] + 30, 6, 3), 200, np.uint8),
                cap(ctx, "3  zoom on first cutting site"),
            ])
            sheet = np.vstack([head, panels])
            fn = f"weld_{idx+1:02d}_{stem}_{na}_{nb}.png".replace("/", "_")
            cv2.imwrite(str(out / fn), sheet)
            manifest.append({"idx": idx + 1, "file": fn, "image": nm,
                             "node": pn, "net_a": na, "net_b": nb,
                             "kind": kind,
                             "shorted_components": ";".join(map(str, shorted)),
                             "cut_degree": cut_deg if cut_deg else "",
                             "n_cut_sites": n_cuts,
                             "single_cut": int(cut is not None),
                             "path_len_exact": len(arr),
                             "detour": r["detour"],
                             "path_len": r["path_len"],
                             "near_detected_xover": r["near_detected_xover"],
                             "near_gt_xover": r["near_gt_xover"]})
        print(f"  {nm}: {len([x for x in manifest if x['image']==nm])} rendered",
              flush=True)

    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]) + ["verdict", "note"])
        w.writeheader()
        for r in manifest:
            w.writerow({**r, "verdict": "", "note": ""})
    n_sc = sum(r["single_cut"] for r in manifest)
    print(f"\n{len(manifest)} sheets -> {out}")
    print(f"  single cutting site exists: {n_sc} "
          f"({n_sc/max(1,len(manifest)):.1%})")


if __name__ == "__main__":
    main()
