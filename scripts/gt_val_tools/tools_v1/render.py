"""Render an inspection overlay: wire pixels coloured by traced net, GT/auto
labels at each terminal, and every intersection site marked."""
import sys, json
import cv2, numpy as np
sys.path.insert(0, "/home/claude/tools")
from trace import trace, assign_terminals, nets_from

PAL = [(0,160,0),(255,60,0),(0,90,255),(200,0,200),(0,170,200),(120,90,0),
       (255,0,120),(90,0,220),(0,200,120),(160,120,255),(200,140,0),(0,110,110),
       (255,120,180),(60,60,255),(120,200,0),(200,60,60),(0,60,160),(140,0,80),
       (0,220,220),(180,180,0),(255,150,80),(100,255,150),(150,80,200),(80,150,80)]

def colour(i):
    return PAL[i % len(PAL)]

def render(img_path, gt, out_path, gt_nets=None, scale=1.0, show_sites=True):
    tr = trace(img_path, gt)
    asg = assign_terminals(gt, tr)
    auto, _ = nets_from(gt, tr, asg)
    g = cv2.imread(str(img_path), 0)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    vis[:] = 255 - (255 - vis) // 4          # fade the drawing
    # colour wire segments by net
    roots = sorted(set(tr["net_of_edge"].values()))
    ridx = {r: i for i, r in enumerate(roots)}
    for e in tr["graph"].edges:
        c = colour(ridx[tr["net_of_edge"][e["id"]]])
        for (y, x) in e["pix"]:
            cv2.circle(vis, (x, y), 1, c, -1)
    for s_ in tr["site_pos"]:
        for (y, x) in s_["pix"]:
            cv2.circle(vis, (x, y), 1, (0, 0, 0), -1)
    # boxes + labels
    for ci, c in enumerate(gt["components"]):
        x1, y1, x2, y2 = tr["boxes"][ci]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.putText(vis, f"{c['id']}:{c['class'][:9]}", (x1, max(9, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1)
        rec = [r for r in asg if r["id"] == c["id"]][0]
        for ti, p in rec["assign"].items():
            net = auto.get((c["id"], ti))
            cv2.circle(vis, (p["x"], p["y"]), 4, (0, 0, 255), 1)
            lbl = str(net)
            if gt_nets:
                gtv = gt_nets.get((c["id"], ti))
                lbl = f"{ti}:{net}"
            else:
                lbl = f"{ti}:{net}"
            cv2.putText(vis, lbl, (p["x"] - 6, p["y"] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1)
    if show_sites:
        for s in tr["sites"]:
            if s["degree"] >= 3:
                col = (0, 0, 255) if s["kind"] == "cross" else (0, 150, 0)
                cv2.drawMarker(vis, (s["x"], s["y"]), col, cv2.MARKER_SQUARE, 13, 1)
                cv2.putText(vis, f"{s['degree']}{s['kind'][0]}{s['dot_score']}",
                            (s["x"] + 8, s["y"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out_path), vis)
    return tr, asg, auto

if __name__ == "__main__":
    stem = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/cal"
    gt = json.load(open(f"{root}/gt/{stem}.json"))
    gtn = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"]}
    tr, asg, auto = render(f"{root}/img1024/{stem}.jpg", gt, f"/tmp/{stem}_dbg.png", gtn)
    print("GT vs AUTO per terminal:")
    for c in gt["components"]:
        row = []
        for t in c["terminals"]:
            row.append(f"t{t['index']} gt={t['net']} auto={auto.get((c['id'], t['index']))}")
        print(f"  {c['id']:>3} {c['class']:<16} " + " | ".join(row))
