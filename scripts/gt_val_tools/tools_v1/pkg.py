"""Build the per-image review package: overlay, zoomed site montages,
component montages, and a machine-readable report."""
from __future__ import annotations
import json, os, sys, math
sys.path.insert(0, "/home/claude/tools")
import cv2
import numpy as np
from netbuild import analyse

PAL = [(0,150,0),(220,60,0),(0,80,230),(190,0,190),(0,150,190),(130,90,0),
       (230,0,120),(90,0,220),(0,180,110),(150,110,240),(190,130,0),(0,110,110),
       (240,110,170),(60,60,240),(110,180,0),(190,60,60),(0,60,160),(140,0,80),
       (0,200,200),(170,170,0),(240,140,80),(80,220,140),(140,80,190),(80,140,80)]


def colour(i):
    return PAL[i % len(PAL)]


def overlay(img_path, gt, res, out, scale=1.6):
    g = cv2.imread(str(img_path), 0)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    vis[:] = 255 - (255 - vis) // 3
    tr = res["tr"]
    roots = sorted({res["euf"].find(e["id"]) for e in tr["graph"].edges})
    ridx = {r: i for i, r in enumerate(roots)}
    for e in tr["graph"].edges:
        c = colour(ridx[res["euf"].find(e["id"])])
        for (y, x) in e["pix"]:
            vis[y, x] = c
    for s_ in tr["site_pos"]:
        for (y, x) in s_["pix"]:
            vis[y, x] = (40, 40, 40)
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = scale
    for ci, c in enumerate(gt["components"]):
        x1, y1, x2, y2 = [int(v * S) for v in tr["boxes"][ci]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.putText(vis, f"#{c['id']} {c['class']}", (x1, max(11, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    for (cid, ti), p in res["detail"].items():
        x, y = int(p["x"] * S), int(p["y"] * S)
        net = res["nets"].get((cid, ti))
        cv2.circle(vis, (x, y), 5, (0, 0, 220), 2)
        cv2.putText(vis, f"{cid}.{ti}={net}", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
    for s in res["sites"]:
        x, y = int(s["x"] * S), int(s["y"] * S)
        crit = s["critical"]
        col = (0, 0, 255) if crit else (110, 110, 110)
        cv2.drawMarker(vis, (x, y), col, cv2.MARKER_SQUARE, 16 if crit else 10, 2 if crit else 1)
        tag = f"S{s['site']}{'X' if s['default']=='crossing' else 'J'}" \
            if isinstance(s["default"], str) else f"S{s['site']}G"
        cv2.putText(vis, tag, (x + 9, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 if crit else 0.34, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out), vis)


def montage(img_path, spots, out, win=70, zoom=5, cols=3, rows=2, boxes=None,
            marks=None):
    """spots: list of (label, x, y[, win]). Writes ceil(n/(cols*rows)) files."""
    g = cv2.imread(str(img_path), 0)
    H, W = g.shape
    cell = win * 2 * zoom
    per = cols * rows
    files = []
    for page in range(max(1, math.ceil(len(spots) / per))):
        canvas = np.full((rows * (cell + 26), cols * (cell + 6), 3), 245, np.uint8)
        for k in range(per):
            i = page * per + k
            if i >= len(spots):
                break
            label, cx, cy = spots[i][0], int(spots[i][1]), int(spots[i][2])
            wn = int(spots[i][3]) if len(spots[i]) > 3 else win
            zm = cell / float(2 * wn)
            x0, y0 = max(0, cx - wn), max(0, cy - wn)
            x1, y1 = min(W, cx + wn), min(H, cy + wn)
            crop = cv2.cvtColor(g[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
            crop = cv2.resize(crop, None, fx=zm, fy=zm, interpolation=cv2.INTER_NEAREST)
            zoom_l = zm
            # draw overlays in crop coords
            if boxes:
                for (bx1, by1, bx2, by2) in boxes:
                    cv2.rectangle(crop, (int((bx1 - x0) * zoom_l), int((by1 - y0) * zoom_l)),
                                  (int((bx2 - x0) * zoom_l), int((by2 - y0) * zoom_l)),
                                  (255, 120, 0), 1)
            for m in (marks or []):
                mx, my, txt, col, ddy = m
                if x0 <= mx < x1 and y0 <= my < y1:
                    px, py = int((mx - x0) * zoom_l), int((my - y0) * zoom_l)
                    cv2.circle(crop, (px, py), 7, col, 2)
                    cv2.putText(crop, txt, (px + 9, py + ddy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
            cv2.drawMarker(crop, (int((cx - x0) * zoom_l), int((cy - y0) * zoom_l)),
                           (0, 0, 255), cv2.MARKER_CROSS, 26, 1)
            r, c = divmod(k, cols)
            oy = r * (cell + 26) + 24
            ox = c * (cell + 6)
            hh, ww = crop.shape[:2]
            canvas[oy:oy + hh, ox:ox + ww] = crop[:min(hh, canvas.shape[0] - oy),
                                                  :min(ww, canvas.shape[1] - ox)]
            cv2.putText(canvas, label, (ox + 4, oy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (ox, oy), (ox + cell, oy + cell), (180, 180, 180), 1)
        p = f"{out}_{page}.png"
        cv2.imwrite(p, canvas)
        files.append(p)
    return files


def build(stem, root, outdir, decisions=None):
    os.makedirs(outdir, exist_ok=True)
    gtp = f"{root}/gt/{stem}.json"
    gt = json.load(open(gtp))
    img = f"{root}/img1024/{stem}.jpg"
    res = analyse(img, gt, decisions)
    overlay(img, gt, res, f"{outdir}/{stem}_overview.png")

    for s in res["sites"]:
        rs = res["tr"]["sites"][s["site"]]
        s["hop_score"] = rs.get("hop_score")
        s["turns"] = rs.get("turns")
        s["kind"] = rs.get("kind")
    crit = [s for s in res["sites"] if s["critical"]]
    other = [s for s in res["sites"] if not s["critical"]]
    spots = [(f"S{s['site']} deg{s['degree']} def={str(s['default'])[:5]} dot={s['dot_score']}"
              + ("  CRITICAL" if s["critical"] else ""),
              s["x"], s["y"]) for s in crit + other]
    site_files = montage(img, spots, f"{outdir}/{stem}_sites",
                         win=60, zoom=5, cols=3, rows=2, boxes=res["tr"]["boxes"]) if spots else []

    cspots = []
    marks = []
    for ci, c in enumerate(gt["components"]):
        cx, cy, w, h = c["bbox"]
        pl = res["tr"]["ports"].get(ci, [])
        lbl = f"#{c['id']} {c['class']}"
        cspots.append((lbl, cx, cy, max(52, int(max(w, h) / 2) + 34)))
        for pi, p in enumerate(pl):
            marks.append((p["x"], p["y"], f"p{pi}", (0, 0, 220), -10))
    for (cid, ti), p in res["detail"].items():
        marks.append((p["x"], p["y"], f"=t{ti}", (0, 140, 0), 20))
    comp_files = montage(img, cspots, f"{outdir}/{stem}_comps", win=58, zoom=4,
                         cols=4, rows=3, boxes=res["tr"]["boxes"], marks=marks)

    report = {
        "image": gt["image"], "stem": stem,
        "components": [{
            "id": c["id"], "class": c["class"], "bbox": c["bbox"],
            "n_terminals": len(c["terminals"]),
            "ports": [{"i": i, "x": p["x"], "y": p["y"], "side": p["side"],
                       "wire_len": p["len"]} for i, p in enumerate(res["tr"]["ports"].get(ci, []))],
            "assigned": {str(ti): {"x": p["x"], "y": p["y"]}
                         for ti, p in [(ti, p) for (cid, ti), p in res["detail"].items() if cid == c["id"]]},
            "nets": {str(t["index"]): res["nets"].get((c["id"], t["index"]))
                     for t in c["terminals"]},
        } for ci, c in enumerate(gt["components"])],
        "sites": res["sites"],
        "bridges": res["bridges"],
        "warnings": res["warnings"],
        "net_summary": _net_summary(gt, res),
        "files": {"overview": f"{outdir}/{stem}_overview.png",
                  "sites": site_files, "comps": comp_files},
    }
    json.dump(report, open(f"{outdir}/{stem}_report.json", "w"), indent=1)
    _summary(report, res, f"{outdir}/{stem}_summary.txt")
    return report, res


def _summary(rep, res, path):
    L = []
    L.append(f"IMAGE {rep['image']}   (1024x1024 frame: data/cleaned_1024/{rep['image']})")
    L.append("")
    L.append("COMPONENTS  (ports = wire ends the tracer found touching the box;")
    L.append("             '->tN' marks which terminal index currently uses that port)")
    for c in rep["components"]:
        asg = {(v["x"], v["y"]): k for k, v in c["assigned"].items()}
        ports = ", ".join(
            f"p{p['i']}@({p['x']},{p['y']}) {p['side']} wire={p['wire_len']}"
            + (f" ->t{asg[(p['x'], p['y'])]}" if (p["x"], p["y"]) in asg else "")
            for p in c["ports"]) or "NONE FOUND"
        L.append(f"  #{c['id']:<3} {c['class']:<16} centre=({c['bbox'][0]:.0f},{c['bbox'][1]:.0f}) "
                 f"size={c['bbox'][2]:.0f}x{c['bbox'][3]:.0f} terminals={c['n_terminals']}")
        L.append(f"       ports: {ports}")
        L.append(f"       nets : " + " ".join(f"t{k}={v}" for k, v in sorted(c["nets"].items())))
    L.append("")
    L.append("INTERSECTION SITES  (junction = all branches merge; crossing = opposite")
    L.append("                     branches pass through each other)")
    for s in rep["sites"]:
        star = "  <-- CRITICAL" if s["critical"] else ""
        L.append(f"  S{s['site']:<4} ({s['x']},{s['y']}) deg={s['degree']} "
                 f"default={str(s['default']):<9} dot={s['dot_score']:<5} hop={s.get('hop_score')}{star}")
    cb = [b for b in rep["bridges"] if b["critical"]]
    L.append("")
    L.append(f"GAP BRIDGES: {len(rep['bridges'])} total, {len(cb)} change the netlist")
    for b in cb:
        L.append(f"  B{b['bridge']:<3} ({b['x']},{b['y']}) -> ({b['to_x']},{b['to_y']}) "
                 f"dist={b['dist']}  <-- CRITICAL: verify these really touch")
    L.append("")
    L.append("CURRENT NETS")
    for n, ts in sorted(rep["net_summary"].items()):
        L.append(f"  {n:<6} {len(ts):>2} terminals: " + ", ".join(ts))
    if rep["warnings"]:
        L.append("")
        L.append("WARNINGS")
        for w in rep["warnings"]:
            L.append("  - " + w)
    open(path, "w").write("\n".join(L) + "\n")


def _net_summary(gt, res):
    out = {}
    for c in gt["components"]:
        for t in c["terminals"]:
            n = res["nets"].get((c["id"], t["index"]))
            out.setdefault(str(n), []).append(f"{c['id']}.{t['index']}({c['class']})")
    return out


if __name__ == "__main__":
    stem = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/val"
    outdir = sys.argv[3] if len(sys.argv) > 3 else f"/home/claude/pkg/{stem}"
    dec = None
    if len(sys.argv) > 4 and os.path.exists(sys.argv[4]):
        dec = json.load(open(sys.argv[4]))
    rep, _ = build(stem, root, outdir, dec)
    print(json.dumps({k: rep[k] for k in ("warnings", "files")}, indent=1))
    print("critical sites:", [s["site"] for s in rep["sites"] if s["critical"]])
