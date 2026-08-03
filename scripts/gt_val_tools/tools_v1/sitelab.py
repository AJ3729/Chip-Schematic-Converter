"""Label every intersection site with the truth implied by verified GT, then
dump features so the junction-vs-crossing rule can be calibrated instead of
guessed."""
import json, glob, os, sys, math
sys.path.insert(0, "/home/claude/tools")
import numpy as np
import cv2
from collections import defaultdict
from trace import trace, assign_terminals, UF


def site_features(root, stem, gt, **kw):
    tr = trace(f"{root}/img1024/{stem}.jpg", gt, **kw)
    asg = assign_terminals(gt, tr)
    G = tr["graph"]
    sites = tr["site_pos"]
    reports = tr["sites"]
    # terminal -> edge
    term_edge = {}
    for rec in asg:
        for ti, p in rec["assign"].items():
            term_edge[(rec["id"], ti)] = p["edge"]
    gtn = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"]}

    out = []
    for si, rep in enumerate(reports):
        if rep["degree"] < 3:
            continue
        uf = UF()
        for e in G.edges:
            uf.find(e["id"])
        for b in tr["bridges"]:
            if b["target"][0] == "e":
                uf.union(b["edge"], b["target"][1])
        for sj, r2 in enumerate(reports):
            if sj == si:
                continue
            for a, b in r2["unions"]:
                uf.union(a, b)
        branches = [e["id"] for e in G.edges if si in e["sites"]]
        # which GT nets does each branch reach?
        reach = []
        for b in branches:
            nets = set()
            for k, eid in term_edge.items():
                if uf.find(eid) == uf.find(b) and gtn.get(k):
                    nets.add(gtn[k])
            reach.append(nets)
        # truth: junction if any two branches reach DIFFERENT gt nets -> crossing
        allnets = set()
        for r in reach:
            allnets |= r
        informative = sum(1 for r in reach if r)
        truth = None
        if informative >= 2:
            truth = "junction" if len(allnets) == 1 else "crossing"
        out.append({"stem": stem, "site": si, "x": rep["x"], "y": rep["y"],
                    "degree": rep["degree"], "dot_score": rep["dot_score"],
                    "kind": rep["kind"], "truth": truth,
                    "nets_reached": len(allnets), "informative": informative})
    return out, tr


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/cal"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rows = []
    for f in sorted(glob.glob(f"{root}/gt/*.json"))[:limit]:
        gt = json.load(open(f)); stem = os.path.basename(f)[:-5]
        try:
            r, _ = site_features(root, stem, gt)
            rows += r
        except Exception as e:
            print("ERR", stem, e)
    json.dump(rows, open("/tmp/sites.json", "w"))
    lab = [r for r in rows if r["truth"]]
    print(f"{len(rows)} sites, {len(lab)} labelled")
    for deg in sorted({r['degree'] for r in lab}):
        sub = [r for r in lab if r["degree"] == deg]
        j = [r["dot_score"] for r in sub if r["truth"] == "junction"]
        c = [r["dot_score"] for r in sub if r["truth"] == "crossing"]
        print(f"deg {deg}: n={len(sub)} junction={len(j)} crossing={len(c)}")
        if j: print(f"   junction dot_score  p10={np.percentile(j,10):.2f} med={np.median(j):.2f} p90={np.percentile(j,90):.2f}")
        if c: print(f"   crossing dot_score  p10={np.percentile(c,10):.2f} med={np.median(c):.2f} p90={np.percentile(c,90):.2f}")
