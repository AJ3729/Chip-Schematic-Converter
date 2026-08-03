"""Deterministic net construction from the traced wire graph + explicit
decisions, plus a criticality analysis that tells a reviewer which
intersection sites actually matter.

decisions = {
  "sites":    {site_id: "junction" | "crossing" | [[e,e,..],[e,..]]},
  "bridges":  {"drop": [i, ...], "add": [[x1,y1,x2,y2], ...]},
  "ports":    {comp_id: {term_index: port_index | null}},
  "classes":  {comp_id: "Resistor", ...},
  "unconnected": [comp_id, ...],
}
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from trace import trace, assign_terminals, UF


def _nets(tr, site_override, drop_bridges=(), extra_unions=()):
    euf = UF()
    for e in tr["graph"].edges:
        euf.find(e["id"])
    for i, b in enumerate(tr["bridges"]):
        if i in drop_bridges:
            continue
        if b["target"][0] == "e":
            euf.union(b["edge"], b["target"][1])
    for si, rep in enumerate(tr["sites"]):
        ov = site_override[si] if si in site_override else None
        ends = rep["branches"]
        dirs = rep["dirs"]
        if ov == "none":
            pass
        elif ov == "junction":
            for e in ends:
                euf.union(ends[0], e)
        elif ov == "crossing":
            for a, b in _pair_opposite(ends, dirs):
                euf.union(a, b)
        elif isinstance(ov, list):
            for grp in ov:
                for e in grp[1:]:
                    euf.union(grp[0], e)
        else:
            for a, b in rep["unions"]:
                euf.union(a, b)
    for a, b in extra_unions:
        euf.union(a, b)
    return euf


def _pair_opposite(ends, dirs):
    idx = list(range(len(ends)))
    out = []
    while len(idx) > 1:
        best = None
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                a = dirs[idx[ii]]; b = dirs[idx[jj]]
                c = a[0] * b[0] + a[1] * b[1]
                if best is None or c < best[0]:
                    best = (c, ii, jj)
        _, ii, jj = best
        out.append((ends[idx[ii]], ends[idx[jj]]))
        for t in sorted((ii, jj), reverse=True):
            idx.pop(t)
    return out


def terminal_map(gt, tr, decisions, euf):
    """(comp_id, term_index) -> edge root, using assignments + port overrides."""
    asg = assign_terminals(gt, tr)
    pov = {int(k): {int(kk): vv for kk, vv in v.items()}
           for k, v in (decisions.get("ports") or {}).items()}
    out = {}
    detail = {}
    for ci, rec in enumerate(asg):
        cid = rec["id"]
        chosen = dict(rec["assign"])
        if cid in pov:
            for ti, pi in pov[cid].items():
                if pi is None:
                    chosen.pop(ti, None)
                else:
                    chosen[ti] = rec["ports"][pi]
        for ti, p in chosen.items():
            out[(cid, ti)] = euf.find(p["edge"])
            detail[(cid, ti)] = p
    return out, asg, detail


def label_nets(gt, tmap):
    gnd = set()
    for c in gt["components"]:
        if c["class"] == "GND":
            for ti in range(len(c["terminals"])):
                if (c["id"], ti) in tmap:
                    gnd.add(tmap[(c["id"], ti)])
    order = []
    for c in gt["components"]:
        for t in c["terminals"]:
            k = (c["id"], t["index"])
            if k in tmap and tmap[k] not in order:
                order.append(tmap[k])
    lab, i = {}, 1
    for v in order:
        if v in gnd:
            lab[v] = "0"
        else:
            lab[v] = f"n{i}"; i += 1
    return {k: lab[v] for k, v in tmap.items()}, lab


def _apply_merges(tmap, decisions):
    """Force listed terminals onto one net: {"merge": [["5.0","7.1"], ...]}"""
    merges = decisions.get("merge") or []
    if not merges:
        return tmap
    uf = UF()
    for v in set(tmap.values()):
        uf.find(v)
    for grp in merges:
        keys = []
        for tag in grp:
            cid, ti = tag.split(".")
            k = (int(cid), int(ti))
            if k in tmap:
                keys.append(k)
        for k in keys[1:]:
            uf.union(tmap[keys[0]], tmap[k])
    return {k: uf.find(v) for k, v in tmap.items()}


def analyse(img_path, gt, decisions=None, **trace_kw):
    decisions = decisions or {}
    so = {int(k): v for k, v in (decisions.get("sites") or {}).items()}
    drop = set((decisions.get("bridges") or {}).get("drop", []))
    tr = trace(img_path, gt, **trace_kw)
    euf = _nets(tr, so, drop)
    tmap, asg, detail = terminal_map(gt, tr, decisions, euf)
    tmap = _apply_merges(tmap, decisions)
    nets, _ = label_nets(gt, tmap)
    for tag, val in (decisions.get("manual_nets") or {}).items():
        cid, ti = tag.split("."); nets[(int(cid), int(ti))] = val
    for cid in (decisions.get("drop_terminals") or []):
        pass

    def partition(so2, drop2):
        e2 = _nets(tr, so2, drop2)
        t2, _, _ = terminal_map(gt, tr, decisions, e2)
        t2 = _apply_merges(t2, decisions)
        groups = defaultdict(set)
        for k, v in t2.items():
            groups[v].add(k)
        return frozenset(frozenset(v) for v in groups.values())

    base = partition(so, drop)
    # criticality: does flipping this site change the terminal partition?
    crit = []
    for si, rep in enumerate(tr["sites"]):
        if rep["degree"] < 3:
            continue
        if si in so:
            cur = so[si]
        else:
            cur = ("junction" if rep["kind"] in ("T", "dot", "pass",
                                                 "plainX-junction") else "crossing")
        other = "crossing" if cur == "junction" else "junction"
        if not isinstance(cur, str):
            cur = "explicit-groups"
            other = "junction"
        alt = dict(so); alt[si] = other
        changed = partition(alt, drop) != base
        crit.append({"site": si, "x": rep["x"], "y": rep["y"], "degree": rep["degree"],
                     "dot_score": rep["dot_score"], "default": cur,
                     "critical": bool(changed), "branches": rep["branches"]})
    # bridges that matter
    bcrit = []
    for i, b in enumerate(tr["bridges"]):
        if i in drop:
            continue
        alt = set(drop); alt.add(i)
        bcrit.append({"bridge": i, "x": b["x"], "y": b["y"], "to_x": b["to_x"],
                      "to_y": b["to_y"], "dist": b["dist"],
                      "critical": partition(so, alt) != base})
    warn = []
    for rec in asg:
        c = [c for c in gt["components"] if c["id"] == rec["id"]][0]
        need = len(c["terminals"])
        got = sum(1 for ti in range(need) if (rec["id"], ti) in tmap)
        if got < need:
            warn.append(f"component {rec['id']} ({c['class']}): only {got}/{need} "
                        f"terminals resolved ({rec['nports']} port candidates found)")
        used = {(p["x"], p["y"]) for (cid2, ti), p in detail.items() if cid2 == rec["id"]}
        spare = [p for p in rec["ports"] if (p["x"], p["y"]) not in used and p["len"] >= 12]
        if spare:
            warn.append(f"component {rec['id']} ({c['class']}): {len(spare)} unused port "
                        f"candidate(s) with a real wire attached at "
                        + ", ".join(f"({p['x']},{p['y']})" for p in spare)
                        + " — confirm the terminals point at the right leads")
    counts = defaultdict(int)
    for v in nets.values():
        counts[v] += 1
    for k, v in sorted(counts.items()):
        if v < 2:
            warn.append(f"net {k!r} touches only {v} terminal")
    return {"tr": tr, "nets": nets, "asg": asg, "detail": detail, "sites": crit,
            "bridges": bcrit, "warnings": warn, "euf": euf}
