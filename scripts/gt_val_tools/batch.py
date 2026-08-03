import json, os, sys, time, traceback
sys.path.insert(0, "/home/claude/tools")
from pkg import build

root = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/val"
outroot = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/pkg"
stems = [l.strip()[:-4] for l in open(f"{root}/val.txt") if l.strip()] \
    if os.path.exists(f"{root}/val.txt") else \
    [l.strip()[:-4] for l in open(f"{root}/test.txt") if l.strip()]
if len(sys.argv) > 3:
    a, b = sys.argv[3].split(":")
    stems = stems[int(a):int(b)]
stats = []
t0 = time.time()
for i, s in enumerate(stems):
    try:
        t = time.time()
        rep, res = build(s, root, f"{outroot}/{s}")
        crit = [x for x in rep["sites"] if x["critical"]]
        stats.append({"stem": s, "ok": True, "ncomp": len(rep["components"]),
                      "nsites": len(rep["sites"]), "ncrit": len(crit),
                      "nwarn": len(rep["warnings"]),
                      "unresolved": sum(1 for c in rep["components"]
                                        for v in c["nets"].values() if v is None),
                      "nets": len({v for c in rep["components"] for v in c["nets"].values() if v}),
                      "secs": round(time.time() - t, 1)})
    except Exception as e:
        stats.append({"stem": s, "ok": False, "err": repr(e)[:200]})
        traceback.print_exc()
    if (i + 1) % 10 == 0:
        print(f"{i+1}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)
json.dump(stats, open(f"{outroot}/_batch_stats.json", "w"), indent=1)
ok = [s for s in stats if s["ok"]]
print(f"done {len(ok)}/{len(stats)} in {time.time()-t0:.0f}s")
print("mean critical sites", sum(s["ncrit"] for s in ok) / max(1, len(ok)))
print("total unresolved terminals", sum(s["unresolved"] for s in ok))
print("failures:", [s for s in stats if not s["ok"]][:5])
