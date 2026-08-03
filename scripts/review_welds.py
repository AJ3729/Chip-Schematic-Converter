#!/usr/bin/env python3
"""A review sheet of welds, so a human can judge WHY ground truth disagrees.

Every weld here is a place where the pipeline found a continuous drawn conductor
between two terminals that the hand-verified netlist says belong to DIFFERENT
nets. Measurement has already established that these are not artifacts: 98.8% run
at full stroke width with no bottleneck, and 100% are >=80% originally-drawn ink
with none depending on morphological fill. So the ink really does connect them
and the question is what the DRAWING means, which only a person can settle.

Three verdicts are possible for each one, and they lead to completely different
work:

  HOP          the drawing has a wire hop (a U or semicircle detour) that the
               dataset's Wire Crossover class does not annotate. Fixable, and
               self-labelable from exactly these locations.
  JUDGEMENT    no hop is drawn; the annotator separated the nets using circuit
               knowledge. Not recoverable from pixels -- this bounds the ceiling.
  GT ERROR     the drawing plainly connects them and the annotation is wrong.
               Changes the denominator every number is measured against.

Each row shows the RAW crop beside the annotated one, because the verdict rests
on the shape of the ink and an overlay hides it. Rows are ordered by detour ratio
so the hop-shaped candidates come first, and each carries whether a Wire
Crossover box -- detected or ground-truth -- lies anywhere near, since a hop with
no box nearby is precisely the under-annotation case.

Usage:
    python scripts/review_welds.py --limit 60 --max-welds 60
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
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
from schematic2netlist.splits import add_split_arg, load_split


def png_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


def crop_around(img, pts, pad=70, min_side=240):
    ys, xs = pts[:, 0], pts[:, 1]
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + pad
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + pad
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    side = max(y1 - y0, x1 - x0, min_side)
    y0, y1 = cy - side // 2, cy + side // 2
    x0, x1 = cx - side // 2, cx + side // 2
    H, W = img.shape[:2]
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(H, y1), min(W, x1)
    return img[y0:y1, x0:x1], (y0, x0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_split_arg(ap, "val")
    ap.add_argument("--limit", type=int, default=60, help="images to scan")
    ap.add_argument("--max-welds", type=int, default=60)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/weld_review")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    idir = Path(cfg["preprocess"]["images_dir"])
    names = load_split(args.split, args.splits_dir)
    names = names[: args.limit]

    welds = []
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        gp = Path(cfg["benchmark"]["gt_dir"]) / f"{stem}.json"
        dp = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
        ip = idir / nm
        if not (gp.exists() and dp.exists() and ip.exists()):
            continue
        gt = load_gt(str(gp))
        gc0 = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gc0:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            str(dp), min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(str(ip), cfg, detections=dets)
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        node_map, comps = res["node_map"], res["components"]

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
        txy = {}
        for c in pc:
            s = byp.get(c["id"])
            if s is None:
                continue
            x1, y1, x2, y2 = bbox_xyxy(res["detections"][s["id"]])
            n = len(c["nets"])
            for k in range(n):
                txy[(c["id"], k)] = (int((y1 + y2) / 2),
                                     int(x1 + (k + 1) * (x2 - x1) / (n + 1)))

        # crossover boxes present in this frame, detected and ground-truth
        det_x = [bbox_xyxy(d) for d in res["detections"]
                 if canonical_class(d["class"]) == "Wire Crossover"]
        gt_x = [c["bbox"] for c in gt["components"]
                if canonical_class(c["class"]) == "Wire Crossover"]

        onn = defaultdict(lambda: defaultdict(list))
        for t, pn in pof.items():
            gn = gof.get(t)
            if pn is not None and gn is not None and t in txy:
                onn[pn][gn].append(t)

        for pn, nets in onn.items():
            if len(nets) < 2:
                continue
            nid = name_to_id.get(pn)
            if nid is None:
                continue
            m = node_map == nid
            if not m.any():
                continue
            pts = np.argwhere(m)
            snap = lambda q: tuple(pts[np.argmin(((pts - np.array(q)) ** 2).sum(1))])
            keys = sorted(nets)
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    SA = [snap(txy[t]) for t in nets[keys[a]] if t in txy]
                    SB = [snap(txy[t]) for t in nets[keys[b]] if t in txy]
                    if not SA or not SB:
                        continue
                    path = bfs_path(m, SA, SB)
                    if not path or len(path) < 3:
                        continue
                    arr = np.array(path)
                    straight = float(np.hypot(*(arr[0] - arr[-1]))) or 1.0
                    # nearest crossover box (detected / GT) to the path
                    def near(boxes, is_xyxy):
                        best = 1e9
                        for bx in boxes:
                            if is_xyxy:
                                bcx, bcy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                            else:
                                bcx, bcy = bx[0], bx[1]
                            d = np.min(np.hypot(arr[:, 1] - bcx, arr[:, 0] - bcy))
                            best = min(best, float(d))
                        return None if best > 1e8 else round(best, 1)
                    welds.append({
                        "image": nm, "node": pn,
                        "net_a": keys[a], "net_b": keys[b],
                        "path_len": len(path),
                        "straight": round(straight, 1),
                        "detour": round(len(path) / straight, 2),
                        "near_detected_xover": near(det_x, True),
                        "near_gt_xover": near(gt_x, False),
                        "_arr": arr, "_base": base, "_mask": m,
                        "_sa": SA, "_sb": SB,
                    })
        if i % 10 == 0:
            print(f"  [{i}/{len(names)}] welds found {len(welds)}", flush=True)

    welds.sort(key=lambda w: -w["detour"])
    welds = welds[: args.max_welds]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for j, w in enumerate(welds):
        arr, base, m = w["_arr"], w["_base"], w["_mask"]
        raw_crop, _ = crop_around(base, arr)
        ann = base.copy()
        ann[m] = (0.55 * ann[m] + 0.45 * np.array([255, 210, 210])).astype(np.uint8)
        for y, x in arr:
            cv2.circle(ann, (x, y), 1, (0, 200, 255), -1)
        for y, x in w["_sa"]:
            cv2.circle(ann, (x, y), 8, (0, 0, 225), -1)
            cv2.circle(ann, (x, y), 8, (0, 0, 0), 2)
        for y, x in w["_sb"]:
            cv2.circle(ann, (x, y), 8, (0, 160, 0), -1)
            cv2.circle(ann, (x, y), 8, (0, 0, 0), 2)
        ann_crop, _ = crop_around(ann, arr)
        ctx = ann.copy()
        y0, x0 = arr[:, 0].min(), arr[:, 1].min()
        y1, x1 = arr[:, 0].max(), arr[:, 1].max()
        cv2.rectangle(ctx, (int(x0) - 60, int(y0) - 60),
                      (int(x1) + 60, int(y1) + 60), (255, 0, 255), 3)
        up = lambda im: cv2.resize(im, None, fx=2.0, fy=2.0,
                                   interpolation=cv2.INTER_NEAREST) \
            if min(im.shape[:2]) < 260 else im
        rows.append({**{k: v for k, v in w.items() if not k.startswith("_")},
                     "raw": png_b64(up(raw_crop)), "ann": png_b64(up(ann_crop)),
                     "ctx": png_b64(cv2.resize(ctx, (430, 430)))})

    css = """
:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--card:#161b22;--bd:#30363d;--acc:#58a6ff}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#1f2328;--mut:#59636e;
--card:#f6f8fa;--bd:#d1d9e0;--acc:#0969da}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:26px 18px 90px}
h1{font-size:25px;margin:0 0 6px}.sub{color:var(--mut);margin-bottom:22px;max-width:900px}
.legend{background:var(--card);border:1px solid var(--bd);border-radius:10px;
padding:13px 16px;margin-bottom:26px;font-size:13.5px}
.legend b{color:var(--acc)}
.w{background:var(--card);border:1px solid var(--bd);border-radius:12px;
margin-bottom:22px;overflow:hidden}
.hd{padding:11px 16px;border-bottom:1px solid var(--bd);display:flex;
flex-wrap:wrap;gap:16px;align-items:baseline;font-size:13.5px}
.hd .n{background:var(--acc);color:#fff;border-radius:6px;padding:1px 9px;font-weight:600}
.hd code{font-size:12.5px}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0}
@media(max-width:1000px){.grid{grid-template-columns:1fr}}
.cell{padding:12px;border-right:1px solid var(--bd)}
.cell:last-child{border-right:0}
.cell h5{margin:0 0 7px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
color:var(--mut)}
.cell img{width:100%;height:auto;border-radius:7px;background:#fff;display:block}
.tag{display:inline-block;border-radius:5px;padding:1px 7px;font-size:12px;
margin-left:4px}
.no{background:rgba(220,60,60,.16);color:#e5534b}
.yes{background:rgba(60,180,90,.16);color:#3fb950}
"""
    parts = [f"<title>Weld review — {len(rows)} cases</title>",
             f"<style>{css}</style>", "<div class=wrap>",
             f"<h1>Weld review — {len(rows)} cases</h1>",
             "<div class=sub>Each case is a place where the pipeline found a "
             "continuous <b>drawn</b> conductor between two terminals that the "
             "hand-verified netlist assigns to different nets. These are not "
             "artifacts: 98.8% run at full stroke width with no bottleneck and "
             "100% are &ge;80% originally-drawn ink. The question is what the "
             "drawing means.</div>",
             "<div class=legend>"
             "<b>Look at the RAW crop first</b> — the overlay hides the ink.<br>"
             "&bull; <b>HOP</b>: a U or semicircle detour is drawn &rarr; the "
             "dataset's <code>Wire Crossover</code> class is missing it, and this "
             "location can be self-labelled to train a detector.<br>"
             "&bull; <b>JUDGEMENT</b>: no hop drawn; the nets were separated using "
             "circuit knowledge &rarr; not recoverable from pixels, and this "
             "bounds the ceiling.<br>"
             "&bull; <b>GT ERROR</b>: the drawing plainly connects them &rarr; the "
             "annotation is wrong and the denominator changes.<br><br>"
             "Red and green dots are the two nets' terminals; yellow is the "
             "conductor joining them; pink is the merged node. Ordered by detour "
             "ratio, so hop-shaped candidates come first "
             "(&pi;/2&nbsp;&asymp;&nbsp;1.57 for a semicircle, 2.0 for a square U)."
             "</div>"]
    for j, r in enumerate(rows):
        dx = r["near_detected_xover"]
        gx = r["near_gt_xover"]
        dtag = (f"<span class='tag yes'>detected x-over {dx} px</span>"
                if dx is not None and dx < 60 else
                "<span class='tag no'>no detected x-over within 60 px</span>")
        gtag = (f"<span class='tag yes'>GT x-over {gx} px</span>"
                if gx is not None and gx < 60 else
                "<span class='tag no'>no GT x-over within 60 px</span>")
        parts.append(
            f"<div class=w><div class=hd><span class=n>{j+1}</span>"
            f"<code>{html.escape(r['image'])}</code>"
            f"<span>nets <b>{html.escape(str(r['net_a']))}</b> vs "
            f"<b>{html.escape(str(r['net_b']))}</b></span>"
            f"<span>detour <b>{r['detour']}</b></span>"
            f"<span>path {r['path_len']} px, straight {r['straight']} px</span>"
            f"{dtag}{gtag}</div><div class=grid>"
            f"<div class=cell><h5>raw drawing</h5>"
            f"<img src='data:image/png;base64,{r['raw']}'></div>"
            f"<div class=cell><h5>same crop, annotated</h5>"
            f"<img src='data:image/png;base64,{r['ann']}'></div>"
            f"<div class=cell><h5>where it is in the frame</h5>"
            f"<img src='data:image/png;base64,{r['ctx']}'></div>"
            f"</div></div>")
    parts.append("</div>")
    (out / "review.html").write_text("\n".join(parts))
    with (out / "welds.csv").open("w", newline="") as fh:
        keys = [k for k in rows[0] if k not in ("raw", "ann", "ctx")]
        wr = csv.DictWriter(fh, fieldnames=keys + ["verdict"])
        wr.writeheader()
        for r in rows:
            wr.writerow({**{k: r[k] for k in keys}, "verdict": ""})
    n_no_box = sum(1 for r in rows
                   if (r["near_detected_xover"] is None
                       or r["near_detected_xover"] >= 60)
                   and (r["near_gt_xover"] is None or r["near_gt_xover"] >= 60))
    print(f"\n{len(rows)} welds written")
    print(f"  with NO crossover box (detected or GT) within 60 px: "
          f"{n_no_box} ({n_no_box/len(rows):.1%})")
    print(f"\nwrote {out}/review.html")
    print(f"      {out}/welds.csv  (a blank 'verdict' column to fill in)")


if __name__ == "__main__":
    main()
