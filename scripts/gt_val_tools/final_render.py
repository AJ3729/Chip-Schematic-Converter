"""Regenerate the review overlay for every finished GT file, straight from the
final GT (not from the decisions), so the render always matches the shipped
JSON: wire ink coloured by net, component boxes, and every terminal labelled
<component>.<terminal>=<net>."""
import json, os, sys, glob
sys.path.insert(0, "/home/claude/tools")
import cv2
import numpy as np
from trace import trace
from netbuild import analyse

PAL = [(0,150,0),(220,60,0),(0,80,230),(190,0,190),(0,150,190),(130,90,0),
       (230,0,120),(90,0,220),(0,180,110),(150,110,240),(190,130,0),(0,110,110),
       (240,110,170),(60,60,240),(110,180,0),(190,60,60),(0,60,160),(140,0,80),
       (0,200,200),(170,170,0),(240,140,80),(80,220,140),(140,80,190),(80,140,80)]


def render(stem, gt_dir, dec_dir, img_dir, out_dir, scale=1.5):
    gt = json.load(open(f"{gt_dir}/{stem}.json"))
    decp = f"{dec_dir}/{stem}.json"
    dec = json.load(open(decp)) if os.path.exists(decp) else {}
    src = json.load(open(f"/home/claude/val/gt/{stem}.json"))
    # rebuild the geometry so terminal dots land on the real leads
    for c in src["components"]:
        m = [x for x in gt["components"] if x["id"] == c["id"]]
        if m:
            c["class"] = m[0]["class"]
            if len(c["terminals"]) != len(m[0]["terminals"]):
                c["terminals"] = [{"index": i, "net": None}
                                  for i in range(len(m[0]["terminals"]))]
    res = analyse(f"{img_dir}/{stem}.jpg", src, dec)
    nets = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"]}

    g = cv2.imread(f"{img_dir}/{stem}.jpg", 0)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    vis[:] = 255 - (255 - vis) // 3
    tr = res["tr"]
    names = sorted({v for v in nets.values() if v})
    cidx = {n: i for i, n in enumerate(names)}
    # colour each wire edge by the GT net of any terminal sitting on it
    edge_net = {}
    for (cid, ti), p in res["detail"].items():
        r = res["euf"].find(p["edge"])
        if nets.get((cid, ti)):
            edge_net[r] = nets[(cid, ti)]
    for e in tr["graph"].edges:
        n = edge_net.get(res["euf"].find(e["id"]))
        col = PAL[cidx[n] % len(PAL)] if n else (170, 170, 170)
        for (y, x) in e["pix"]:
            vis[y, x] = col
    for s_ in tr["site_pos"]:
        for (y, x) in s_["pix"]:
            vis[y, x] = (40, 40, 40)
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = scale
    for ci, c in enumerate(gt["components"]):
        cx, cy, w, h = c["bbox"]
        x1, y1 = int((cx - w / 2) * S), int((cy - h / 2) * S)
        x2, y2 = int((cx + w / 2) * S), int((cy + h / 2) * S)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.putText(vis, f"#{c['id']} {c['class']}", (x1, max(11, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    for (cid, ti), p in res["detail"].items():
        n = nets.get((cid, ti))
        col = PAL[cidx[n] % len(PAL)] if n else (0, 0, 255)
        x, y = int(p["x"] * S), int(p["y"] * S)
        cv2.circle(vis, (x, y), 5, col, 2)
        cv2.putText(vis, f"{cid}.{ti}={n}", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
    unresolved = [k for k in nets if k not in res["detail"]]
    cv2.putText(vis, f"{stem}  {len(gt['components'])} components  {len(names)} nets"
                     f"   verified={gt.get('verified')}  annotator={gt.get('annotator')}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1, cv2.LINE_AA)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/{stem}.png", vis)
    return len(unresolved)


if __name__ == "__main__":
    gt_dir = "/home/claude/out/gt"
    n = 0
    miss = 0
    for f in sorted(glob.glob(gt_dir + "/*.json")):
        stem = os.path.basename(f)[:-5]
        try:
            miss += render(stem, gt_dir, "/home/claude/dec", "/home/claude/val/img1024",
                           "/home/claude/out/renders")
            n += 1
        except Exception as e:
            print("ERR", stem, repr(e)[:120])
        if n % 40 == 0:
            print(n, flush=True)
    print(f"rendered {n}; {miss} terminal dots could not be placed geometrically")
