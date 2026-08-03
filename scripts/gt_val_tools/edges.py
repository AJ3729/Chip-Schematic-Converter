"""Dump the traced wire graph: edge ids with endpoints, and the branches of
each intersection site. Needed for the explicit-group site override form.

usage: python3 edges.py <stem> [x y radius] [root]
"""
import sys, json
sys.path.insert(0, "/home/claude/tools")
from trace import trace
stem = sys.argv[1]
root = sys.argv[-1] if sys.argv[-1].startswith("/") else "/home/claude/val"
gt = json.load(open(f"{root}/gt/{stem}.json"))
tr = trace(f"{root}/img1024/{stem}.jpg", gt)
sel = None
if len(sys.argv) >= 5 and sys.argv[2].isdigit():
    sel = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
print("EDGES  (id: length, endpoints as (x,y), sites it touches)")
for e in tr["graph"].edges:
    p = e["path"]
    a, b = (p[0][1], p[0][0]), (p[-1][1], p[-1][0])
    if sel and not any(abs(q[0] - sel[0]) <= sel[2] and abs(q[1] - sel[1]) <= sel[2]
                       for q in (a, b)):
        continue
    print(f"  e{e['id']:<4} len={len(p):<4} {a} -> {b}   sites={e['sites']}")
print()
print("SITES  (id, position, degree, branch edge ids, default unions)")
for i, s in enumerate(tr["sites"]):
    if s["degree"] < 3:
        continue
    if sel and not (abs(s["x"] - sel[0]) <= sel[2] and abs(s["y"] - sel[1]) <= sel[2]):
        continue
    print(f"  S{i:<4} ({s['x']},{s['y']}) deg={s['degree']} kind={s['kind']:<16} "
          f"branches={s['branches']} unions={s['unions']}")
