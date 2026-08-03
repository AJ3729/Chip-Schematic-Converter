"""Structural audit of the finished GT.

The Digitize-HCD sheets are hand drawings of *generated* circuits, and the same
generated circuit is often drawn several times. So a val sheet almost always has
siblings — sheets with an identical component inventory — among the 190
human-verified test files. Comparing a new annotation against its siblings is a
strong, independent error signal: same inventory should mean same net count and,
usually, the same netlist up to renaming.
"""
from __future__ import annotations
import json, glob, os, sys, itertools
from collections import Counter, defaultdict

sys.path.insert(0, "/home/claude/tools")
from erc import check


def inventory(d):
    return tuple(sorted(Counter(c["class"] for c in d["components"]).items()))


def netcount(d):
    return len({t["net"] for c in d["components"] for t in c["terminals"] if t["net"]})


def degree_signature(d):
    """Multiset of (class, sorted tuple of net sizes touched) — a rename-invariant
    fingerprint of the netlist that is cheap and quite discriminative."""
    size = Counter()
    for c in d["components"]:
        for t in c["terminals"]:
            if t["net"]:
                size[t["net"]] += 1
    sig = []
    for c in d["components"]:
        sig.append((c["class"], tuple(sorted(size.get(t["net"], 0) for t in c["terminals"]))))
    return tuple(sorted(Counter(sig).items()))


def load(dirpath):
    out = {}
    for f in sorted(glob.glob(dirpath + "/*.json")):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def main(out_dir="/home/claude/out/gt", ref_dir="/home/claude/cal/gt"):
    ours = load(out_dir)
    ref = load(ref_dir)
    by_inv = defaultdict(list)
    for k, d in ref.items():
        by_inv[inventory(d)].append(k)
    rows = []
    for k, d in sorted(ours.items()):
        errs, warns = check(d)
        inv = inventory(d)
        sibs = by_inv.get(inv, [])
        sib_nets = Counter(netcount(ref[s]) for s in sibs)
        sib_sigs = Counter(degree_signature(ref[s]) for s in sibs)
        n = netcount(d)
        sig = degree_signature(d)
        flags = []
        if errs:
            flags.append(f"ERC-ERROR({len(errs)})")
        if sibs:
            if n not in sib_nets:
                flags.append(f"netcount {n} vs siblings {dict(sib_nets)}")
            if sig not in sib_sigs and len(sibs) >= 3:
                flags.append(f"structure differs from all {len(sibs)} siblings")
        if not d.get("notes"):
            flags.append("no notes")
        if d.get("verified") or d.get("annotator"):
            flags.append("verified/annotator not cleared")
        rows.append({"stem": k, "ncomp": len(d["components"]), "nets": n,
                     "nsib": len(sibs), "flags": flags,
                     "errs": errs, "warns": warns})
    return rows


if __name__ == "__main__":
    rows = main(*(sys.argv[1:3] if len(sys.argv) > 2 else []))
    flagged = [r for r in rows if r["flags"]]
    print(f"{len(rows)} files, {len(flagged)} flagged\n")
    for r in flagged:
        print(f"{r['stem']:<16} comps={r['ncomp']:<3} nets={r['nets']:<3} sibs={r['nsib']:<3} "
              + "; ".join(r["flags"]))
        for e in r["errs"]:
            print("      [ERR ]", e)
    print()
    nosib = sum(1 for r in rows if r["nsib"] == 0)
    print(f"{len(rows)-nosib} files have >=1 verified sibling; {nosib} have none")
    json.dump(rows, open("/home/claude/out/audit.json", "w"), indent=1)
