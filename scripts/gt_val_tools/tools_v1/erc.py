"""Electrical rule checks + schema validation for a finished GT file.
Mirrors the repo validator and adds Tier-1 electrical faults."""
from __future__ import annotations
import json, sys
from collections import defaultdict

TERMS = {
    "Resistor": 2, "Capacitor": 2, "Inductor": 2, "Diode": 2, "Zener Diode": 2,
    "GND": 1, "V-DC": 2, "V-DC (one port)": 1, "V-AC": 2, "I-DC": 2, "I-AC": 2,
    "MOSFET-N": 3, "MOSFET-P": 3, "BJT-NPN": 3, "BJT-PNP": 3, "Op-Amp": 3,
}
SOURCES = {"V-DC", "V-AC"}
ISOURCES = {"I-DC", "I-AC"}
PASSIVE = {"Resistor", "Capacitor", "Inductor", "Diode", "Zener Diode"}


def check(gt, strict=True):
    errs, warns = [], []
    ids = [c["id"] for c in gt["components"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate component ids")
    if gt.get("schema_version") != 1:
        errs.append("schema_version != 1")
    net_terms = defaultdict(list)
    for c in gt["components"]:
        cid, cls = c["id"], c["class"]
        if cls not in TERMS:
            errs.append(f"{cid}: unknown class {cls!r}")
            continue
        if cls == "Wire Crossover":
            errs.append(f"{cid}: Wire Crossover must not appear in topology GT")
        t = c["terminals"]
        idxs = sorted(x["index"] for x in t)
        if idxs != list(range(len(t))):
            errs.append(f"{cid}: terminal indices {idxs} not 0..{len(t)-1}")
        if len(t) != TERMS[cls]:
            errs.append(f"{cid}: {len(t)} terminals, expected {TERMS[cls]} for {cls}")
        for x in t:
            n = x["net"]
            if n is None:
                if strict and not c.get("unconnected", False):
                    errs.append(f"{cid}.{x['index']}: no net and not marked unconnected")
                continue
            if not isinstance(n, str) or not n.strip():
                errs.append(f"{cid}.{x['index']}: invalid net {n!r}")
                continue
            net_terms[n].append((cid, x["index"], cls))
        if cls == "GND" and t[0]["net"] not in (None, "0"):
            errs.append(f"{cid}: ground symbol on net {t[0]['net']!r}, must be '0'")
        nets = [x["net"] for x in t if x["net"]]
        if len(nets) == len(t) and len(set(nets)) == 1 and len(t) > 1:
            if cls in SOURCES:
                errs.append(f"{cid} ({cls}): voltage source short-circuited "
                            f"(both terminals on {nets[0]!r})")
            elif cls in ISOURCES:
                warns.append(f"{cid} ({cls}): current source with both terminals "
                             f"on {nets[0]!r}")
            elif len(t) == 3:
                errs.append(f"{cid} ({cls}): all three terminals on {nets[0]!r}")
            else:
                warns.append(f"{cid} ({cls}): both terminals on {nets[0]!r} "
                             f"(element short-circuited)")
    for n, ts in sorted(net_terms.items()):
        if len(ts) < 2:
            errs.append(f"net {n!r} touches only {len(ts)} terminal "
                        f"({ts[0][0]}.{ts[0][1]} {ts[0][2]})" if ts else f"net {n!r} empty")
    # island check: components linked through shared nets
    adj = defaultdict(set)
    for n, ts in net_terms.items():
        for a in ts:
            for b in ts:
                if a[0] != b[0]:
                    adj[a[0]].add(b[0])
    if gt["components"]:
        seen, stack = set(), [gt["components"][0]["id"]]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            stack.extend(adj[v] - seen)
        missing = [c["id"] for c in gt["components"] if c["id"] not in seen]
        if missing:
            warns.append(f"disconnected island(s): components {missing} do not "
                         f"share a net path with component {gt['components'][0]['id']}")
    grounds = [c for c in gt["components"] if c["class"] == "GND"]
    if grounds and "0" not in net_terms:
        errs.append("GND symbol present but no net '0'")
    return errs, warns


if __name__ == "__main__":
    import glob, os
    bad = 0
    paths = sys.argv[1:]
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = sorted(glob.glob(paths[0] + "/*.json"))
    for p in paths:
        gt = json.load(open(p))
        e, w = check(gt)
        if e or w:
            print(f"--- {os.path.basename(p)}")
            for x in e:
                print("  [ERR ]", x)
            for x in w:
                print("  [warn]", x)
        bad += bool(e)
    print(f"\n{len(paths)} file(s), {bad} with errors")
