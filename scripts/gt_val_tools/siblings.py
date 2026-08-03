"""Find human-verified test-split sheets with the same component inventory."""
import json, glob, os, sys
from collections import Counter
sys.path.insert(0, "/home/claude/tools")

def inv(d): return tuple(sorted(Counter(c["class"] for c in d["components"]).items()))
def nets(d): return len({t["net"] for c in d["components"] for t in c["terminals"] if t["net"]})

stem = sys.argv[1]
mine = json.load(open(f"/home/claude/out/gt/{stem}.json")) if os.path.exists(
    f"/home/claude/out/gt/{stem}.json") else json.load(open(f"/home/claude/val/gt/{stem}.json"))
target = inv(mine)
print(f"{stem}: {len(mine['components'])} components, {nets(mine)} nets")
print("inventory:", ", ".join(f"{n}x{c}" for c, n in target))
for c in mine["components"]:
    print(f"   #{c['id']:<3} {c['class']:<16} " + " ".join(f"t{t['index']}={t['net']}" for t in c["terminals"]))
found = 0
for f in sorted(glob.glob("/home/claude/cal/gt/*.json")):
    d = json.load(open(f))
    if inv(d) != target:
        continue
    found += 1
    s = os.path.basename(f)[:-5]
    print(f"\n--- verified sibling {s}  ({nets(d)} nets)   image: /home/claude/cal/img1024/{s}.jpg")
    for c in d["components"]:
        print(f"   #{c['id']:<3} {c['class']:<16} " + " ".join(f"t{t['index']}={t['net']}" for t in c["terminals"]))
print(f"\n{found} verified sibling(s)")
