"""Render every shipped GT file, using whichever tracer version reproduces that
file's netlist so the terminal dots land on the leads the annotation actually
used. Writes /home/claude/out/renders/<stem>.png and a provenance table."""
import json, os, sys, glob, subprocess

STEMS = sorted(os.path.splitext(f)[0] for f in os.listdir("/home/claude/out/gt"))

WORKER = r'''
import json, os, sys
sys.path.insert(0, TOOLS)
import cv2, numpy as np
from netbuild import analyse

PAL = [(0,150,0),(220,60,0),(0,80,230),(190,0,190),(0,150,190),(130,90,0),
       (230,0,120),(90,0,220),(0,180,110),(150,110,240),(190,130,0),(0,110,110),
       (240,110,170),(60,60,240),(110,180,0),(190,60,60),(0,60,160),(140,0,80),
       (0,200,200),(170,170,0),(240,140,80),(80,220,140),(140,80,190),(80,140,80)]

def prep(stem):
    gt = json.load(open(f"/home/claude/out/gt/{stem}.json"))
    src = json.load(open(f"/home/claude/val/gt/{stem}.json"))
    p = f"/home/claude/dec/{stem}.json"
    dec = json.load(open(p)) if os.path.exists(p) else {}
    for c in src["components"]:
        m = [x for x in gt["components"] if x["id"] == c["id"]]
        if m:
            c["class"] = m[0]["class"]
            if len(c["terminals"]) != len(m[0]["terminals"]):
                c["terminals"] = [{"index": i, "net": None} for i in range(len(m[0]["terminals"]))]
    return gt, src, dec

def part(m):
    g = {}
    for k, v in m.items():
        g.setdefault(v, set()).add(k)
    return frozenset(frozenset(s) for s in g.values())

def run(stem, mode):
    gt, src, dec = prep(stem)
    res = analyse(f"/home/claude/val/img1024/{stem}.jpg", src, dec)
    old = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"] if t["net"]}
    new = {k: v for k, v in res["nets"].items() if v}
    match = (set(old) == set(new)) and (part(old) == part(new))
    if mode == "check":
        print(json.dumps({"stem": stem, "match": bool(match)}))
        return
    nets = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"]}
    g = cv2.imread(f"/home/claude/val/img1024/{stem}.jpg", 0)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    vis[:] = 255 - (255 - vis) // 3
    tr = res["tr"]
    names = sorted({v for v in nets.values() if v})
    cidx = {n: i for i, n in enumerate(names)}
    edge_net = {}
    for (cid, ti), p in res["detail"].items():
        if nets.get((cid, ti)):
            edge_net[res["euf"].find(p["edge"])] = nets[(cid, ti)]
    for e in tr["graph"].edges:
        n = edge_net.get(res["euf"].find(e["id"]))
        col = PAL[cidx[n] % len(PAL)] if n else (175, 175, 175)
        for (y, x) in e["pix"]:
            vis[y, x] = col
    for s_ in tr["site_pos"]:
        for (y, x) in s_["pix"]:
            vis[y, x] = (40, 40, 40)
    S = 1.5
    vis = cv2.resize(vis, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    for c in gt["components"]:
        cx, cy, w, h = c["bbox"]
        x1, y1 = int((cx - w/2) * S), int((cy - h/2) * S)
        x2, y2 = int((cx + w/2) * S), int((cy + h/2) * S)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.putText(vis, f"#{c['id']} {c['class']}", (x1, max(11, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    placed = set()
    for (cid, ti), p in res["detail"].items():
        n = nets.get((cid, ti))
        col = PAL[cidx[n] % len(PAL)] if n else (0, 0, 255)
        x, y = int(p["x"] * S), int(p["y"] * S)
        cv2.circle(vis, (x, y), 5, col, 2)
        cv2.putText(vis, f"{cid}.{ti}={n}", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        placed.add((cid, ti))
    missing = [k for k in nets if k not in placed]
    hdr = (f"{stem}   {len(gt['components'])} components   {len(names)} nets   "
           f"verified={gt.get('verified')}  annotator={gt.get('annotator')}")
    cv2.putText(vis, hdr, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1, cv2.LINE_AA)
    if missing:
        cv2.putText(vis, f"(dots for {len(missing)} terminal(s) not drawn: net asserted in the annotation)",
                    (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 160), 1, cv2.LINE_AA)
    os.makedirs("/home/claude/out/renders", exist_ok=True)
    cv2.imwrite(f"/home/claude/out/renders/{stem}.png", vis)
    print(json.dumps({"stem": stem, "match": bool(match), "missing": len(missing)}))

run(sys.argv[1], sys.argv[2])
'''


def call(stem, tools, mode):
    src = f'TOOLS = {tools!r}\n' + WORKER
    r = subprocess.run([sys.executable, "-c", src, stem, mode],
                       capture_output=True, text=True, timeout=300)
    for line in r.stdout.strip().splitlines()[::-1]:
        try:
            return json.loads(line)
        except Exception:
            continue
    return {"stem": stem, "match": False, "error": r.stderr.strip()[-200:]}


if __name__ == "__main__":
    prov = []
    for i, s in enumerate(STEMS):
        chosen = None
        for tools in ("/home/claude/tools", "/home/claude/tools_v1"):
            res = call(s, tools, "check")
            if res.get("match"):
                chosen = tools
                break
        chosen = chosen or "/home/claude/tools"
        out = call(s, chosen, "render")
        prov.append({"stem": s, "tracer": os.path.basename(chosen),
                     "reproduced": bool(out.get("match")),
                     "dots_missing": out.get("missing", 0)})
        if (i + 1) % 25 == 0:
            print(i + 1, flush=True)
    json.dump(prov, open("/home/claude/out/render_provenance.json", "w"), indent=1)
    ok = sum(1 for p in prov if p["reproduced"])
    from collections import Counter
    print(f"rendered {len(prov)}; netlist reproduced from the decisions file for {ok}")
    print(Counter(p["tracer"] for p in prov))
    print("not reproduced:", [p["stem"] for p in prov if not p["reproduced"]])
