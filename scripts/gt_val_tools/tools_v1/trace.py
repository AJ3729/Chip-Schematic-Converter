"""Wire tracing / net extraction for hand-drawn schematic GT verification (v2).

Pipeline
  1. binarise ink (cleaned 1024 images are near-binary)
  2. erase GT component bboxes -> wire ink (+ text)
  3. drop ink blobs that touch no component box (labels, page border)
  4. skeletonise; build a proper node/edge graph:
       nodes  = clusters of branch pixels (deg>=3) + free endpoints
       edges  = 1-px runs between nodes
  5. prune short spurs, merge nodes that are close together
  6. per node decide junction (merge all) vs crossing (pair opposite branches)
  7. terminals = edge endpoints landing on a component bbox border
  8. nets = connected components of the paired-up edge graph
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import cv2
import numpy as np
from skimage.morphology import skeletonize

INK = 160
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        r = a
        while self.p[r] != r:
            r = self.p[r]
        while self.p[a] != r:
            self.p[a], a = r, self.p[a]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load_ink(img_path, strong=120, weak=205, close=1):
    """Hysteresis binarisation: faint pen strokes are kept when they are
    connected to solid ink, so a light patch does not sever a wire."""
    g = cv2.imread(str(img_path), 0)
    if g is None:
        raise FileNotFoundError(img_path)
    w = (g < weak).astype(np.uint8)
    if close:
        w = cv2.morphologyEx(w, cv2.MORPH_CLOSE,
                             np.ones((2 * close + 1, 2 * close + 1), np.uint8))
    st = (g < strong).astype(np.uint8)
    n, lab = cv2.connectedComponents(w, 8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[st > 0])] = True
    keep[0] = False
    ink = keep[lab].astype(np.uint8)
    return g, ink


def boxes_of(gt, pad=0):
    out = []
    for c in gt["components"]:
        cx, cy, w, h = c["bbox"]
        out.append((int(round(cx - w / 2)) - pad, int(round(cy - h / 2)) - pad,
                    int(round(cx + w / 2)) + pad, int(round(cy + h / 2)) + pad))
    return out


def _nbcount(skel):
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) * skel


class Graph:
    """nodes: list of dicts {y,x,pix:set}; edges: list of dicts {pix:[...], ends:[node|None,node|None]}"""

    def __init__(self, skel):
        self.skel = skel.astype(np.uint8)
        self.build()

    def build(self):
        skel = self.skel
        nb = _nbcount(skel)
        branch = ((nb >= 3) & (skel > 0)).astype(np.uint8)
        # cluster branch pixels (dilate to join diagonal neighbours)
        bd = cv2.dilate(branch, np.ones((3, 3), np.uint8), iterations=1) * skel
        nlab, lab = cv2.connectedComponents(bd.astype(np.uint8), 8)
        self.nodes = []
        for i in range(1, nlab):
            ys, xs = np.where(lab == i)
            self.nodes.append({"y": float(ys.mean()), "x": float(xs.mean()),
                               "pix": set(zip(ys.tolist(), xs.tolist())), "free": False})
        node_of_pix = {}
        for ni, n in enumerate(self.nodes):
            for p in n["pix"]:
                node_of_pix[p] = ni
        # edges: connected runs of skeleton pixels not in any node cluster
        rest = (skel > 0) & (lab == 0)
        rlab_n, rlab = cv2.connectedComponents(rest.astype(np.uint8), 8)
        self.edges = []
        for i in range(1, rlab_n):
            ys, xs = np.where(rlab == i)
            pix = list(zip(ys.tolist(), xs.tolist()))
            self.edges.append({"pix": pix, "ends": [], "id": len(self.edges)})
        # attach edges to nodes by adjacency
        for e in self.edges:
            touch = defaultdict(list)
            for (y, x) in e["pix"]:
                for dy, dx in NB8:
                    q = (y + dy, x + dx)
                    ni = node_of_pix.get(q)
                    if ni is not None:
                        touch[ni].append((y, x))
            e["touch"] = {k: v for k, v in touch.items()}
        # order edge pixels into a path so we can take directions at the ends
        for e in self.edges:
            e["path"] = _order_path(e["pix"])

    # ---- helpers
    def degree(self, ni):
        d = 0
        for e in self.edges:
            if ni in e["touch"]:
                d += 1
        return d


def _order_path(pix):
    """Order a thin 8-connected pixel run from one tip to the other."""
    S = set(pix)
    if len(S) <= 2:
        return list(pix)

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in NB8 if (y + dy, x + dx) in S]

    tips = [p for p in S if len(nbrs(p)) == 1]
    start = tips[0] if tips else next(iter(S))
    path, seen, cur, prev = [start], {start}, start, None
    while True:
        cand = [q for q in nbrs(cur) if q not in seen]
        if not cand:
            break
        if len(cand) > 1 and prev is not None:
            cand.sort(key=lambda q: -((q[0] - prev[0]) ** 2 + (q[1] - prev[1]) ** 2))
        nxt = cand[0]
        path.append(nxt); seen.add(nxt); prev, cur = cur, nxt
    return path


def prune_spurs(skel, min_len=8, rounds=4, boxes=(), keep_tol=7):
    """Drop skeletonisation whiskers, but never a stub that reaches a
    component box - that stub is a component lead."""
    sk = skel.copy()
    for _ in range(rounds):
        g = Graph(sk)
        killed = 0
        for e in g.edges:
            if len(e["touch"]) != 1 or len(e["pix"]) >= min_len:
                continue
            near_box = False
            for (y, x) in e["pix"]:
                for (x1, y1, x2, y2) in boxes:
                    if (x1 - keep_tol) <= x <= (x2 + keep_tol) and \
                       (y1 - keep_tol) <= y <= (y2 + keep_tol):
                        near_box = True
                        break
                if near_box:
                    break
            if near_box:
                continue
            for p in e["pix"]:
                sk[p] = 0
            killed += 1
        if not killed:
            break
    return sk


def stroke_half(wire):
    d = cv2.distanceTransform(wire, cv2.DIST_L2, 5)
    sk = skeletonize(wire > 0)
    v = d[sk]
    return float(np.median(v)) if v.size else 1.5


def _pair_by_direction(ends):
    idx = list(range(len(ends)))
    out = []
    while len(idx) > 1:
        best = None
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                a = ends[idx[ii]]["dir"]; b = ends[idx[jj]]["dir"]
                c = a[0] * b[0] + a[1] * b[1]
                if best is None or c < best[0]:
                    best = (c, ii, jj)
        _, ii, jj = best
        out.append((idx[ii], idx[jj]))
        for t in sorted((ii, jj), reverse=True):
            idx.pop(t)
    return out


def _pairs_from_unions(unions, ends):
    idx = {}
    for i, e in enumerate(ends):
        idx.setdefault(e["edge"], i)
    return [(idx[a], idx[b]) for a, b in unions if a in idx and b in idx and idx[a] != idx[b]]


def _near_site_pix(path, e, sites):
    """The tip of `path` that lies next to the edge's single site."""
    si = e["sites"][0]
    s = sites[si]
    d0 = (path[0][0] - s["y"]) ** 2 + (path[0][1] - s["x"]) ** 2
    d1 = (path[-1][0] - s["y"]) ** 2 + (path[-1][1] - s["x"]) ** 2
    return {path[0] if d0 <= d1 else path[-1]}


def trace(img_path, gt, pad=2, dot_ratio=2.30, min_spur=8, merge_r=None,
          drop_orphans=True, weak=230, close=1, gap=11, site_override=None, hop_thresh=6.0):
    g, ink = load_ink(img_path, weak=weak, close=close)
    H, W = ink.shape
    boxes = boxes_of(gt, pad=pad)
    wire = ink.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(wire, (x1, y1), (x2, y2), 0, -1)

    # --- drop blobs that touch no component box (text labels, page border)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wire, 8)
    keep = np.zeros(n, bool)
    tol = 6
    for (x1, y1, x2, y2) in boxes:
        yy0, yy1 = max(0, y1 - tol), min(H, y2 + tol + 1)
        xx0, xx1 = max(0, x1 - tol), min(W, x2 + tol + 1)
        ring = lab[yy0:yy1, xx0:xx1]
        for v in np.unique(ring):
            if v:
                keep[v] = True
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 12:
            keep[i] = False
        if drop_orphans and not keep[i]:
            wire[lab == i] = 0
    if not drop_orphans:
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 12:
                wire[lab == i] = 0

    half = stroke_half(wire)
    dist = cv2.distanceTransform(wire, cv2.DIST_L2, 5)
    skel = skeletonize(wire > 0).astype(np.uint8)
    skel = prune_spurs(skel, min_len=min_spur, boxes=boxes)
    G = Graph(skel)

    # --- merge nodes that sit close together (a hop touch / thick dot makes several)
    R = merge_r if merge_r is not None else max(4.0, 2.6 * half)
    uf = UF()
    for i in range(len(G.nodes)):
        uf.find(i)
    for i in range(len(G.nodes)):
        for j in range(i + 1, len(G.nodes)):
            a, b = G.nodes[i], G.nodes[j]
            if abs(a["y"] - b["y"]) <= R and abs(a["x"] - b["x"]) <= R:
                uf.union(i, j)
    grp = defaultdict(list)
    for i in range(len(G.nodes)):
        grp[uf.find(i)].append(i)
    sites = []
    site_of_node = {}
    for members in grp.values():
        pix = set()
        for m in members:
            pix |= G.nodes[m]["pix"]
        ys = [p[0] for p in pix]; xs = [p[1] for p in pix]
        si = len(sites)
        sites.append({"y": float(np.mean(ys)), "x": float(np.mean(xs)), "pix": pix})
        for m in members:
            site_of_node[m] = si

    # --- assign each edge to its sites first
    for e in G.edges:
        e["sites"] = sorted({site_of_node[n] for n in e["touch"]})

    # --- bridge small gaps: a free tip that stops just short of other ink
    tip_info = []
    for e in G.edges:
        path = e["path"]
        if len(e["sites"]) == 0:
            tips = {path[0], path[-1]}
        elif len(e["sites"]) == 1 and len(path) > 1:
            tips = {path[0], path[-1]} - _near_site_pix(path, e, sites)
        else:
            tips = set()
        for t in tips:
            tip_info.append((t, e["id"]))

    bridges = []
    if tip_info and gap:
        pix_owner = {}
        for e in G.edges:
            for p in e["pix"]:
                pix_owner[p] = ("e", e["id"])
        for si, s_ in enumerate(sites):
            for p in s_["pix"]:
                pix_owner[p] = ("s", si)
        offs = [(dy, dx) for dy in range(-gap, gap + 1) for dx in range(-gap, gap + 1)
                if 4 < dy * dy + dx * dx <= gap * gap]
        offs.sort(key=lambda d: d[0] * d[0] + d[1] * d[1])
        for (ty, tx), eid in tip_info:
            for dy, dx in offs:
                o = pix_owner.get((ty + dy, tx + dx))
                if o is None or o == ("e", eid):
                    continue
                bridges.append({"x": int(tx), "y": int(ty), "to_x": int(tx + dx),
                                "to_y": int(ty + dy),
                                "dist": round((dy * dy + dx * dx) ** 0.5, 1),
                                "edge": eid, "target": list(o)})
                break

    # --- edge ends -> site
    euf = UF()
    for e in G.edges:
        euf.find(e["id"])
    ends_by_site = defaultdict(list)
    for e in G.edges:
        for si in e["sites"]:
            cy, cx = sites[si]["y"], sites[si]["x"]
            path = e["path"]
            d0 = (path[0][0] - cy) ** 2 + (path[0][1] - cx) ** 2
            d1 = (path[-1][0] - cy) ** 2 + (path[-1][1] - cx) ** 2
            pts = path if d0 <= d1 else path[::-1]
            k = min(len(pts) - 1, 16)
            dy = pts[k][0] - pts[0][0]
            dx = pts[k][1] - pts[0][1]
            nrm = math.hypot(dx, dy) or 1.0
            # curvature: angle between the initial tangent and the chord to a
            # point ~38 px out. A drawn hop arc bends hard; a straight wire does not.
            k2 = min(len(pts) - 1, 38)
            dy2 = pts[k2][0] - pts[0][0]
            dx2 = pts[k2][1] - pts[0][1]
            nrm2 = math.hypot(dx2, dy2) or 1.0
            cosang = max(-1.0, min(1.0, (dx * dx2 + dy * dy2) / (nrm * nrm2)))
            curve = math.degrees(math.acos(cosang)) if k2 > k else 0.0
            # turn: tangent over the first 10 px vs the tangent over px 14..30.
            # A drawn hop swings ~90 deg within that distance; a straight wire
            # crossing another wire does not turn at all.
            turn = 0.0
            if len(pts) > 16:
                a1 = (pts[min(10, len(pts) - 1)][1] - pts[0][1],
                      pts[min(10, len(pts) - 1)][0] - pts[0][0])
                i0, i1 = min(14, len(pts) - 1), min(30, len(pts) - 1)
                a2 = (pts[i1][1] - pts[i0][1], pts[i1][0] - pts[i0][0])
                n1 = math.hypot(*a1) or 1.0; n2_ = math.hypot(*a2) or 1.0
                cs = max(-1.0, min(1.0, (a1[0] * a2[0] + a1[1] * a2[1]) / (n1 * n2_)))
                turn = math.degrees(math.acos(cs))
            ends_by_site[si].append({"edge": e["id"], "dir": (dx / nrm, dy / nrm),
                                     "len": len(path), "curve": round(curve, 1),
                                     "turn": round(turn, 1), "reach": k2})
    # a bridge that lands on a site is treated as an extra branch of that site
    site_extra = defaultdict(list)
    for b in bridges:
        if b["target"][0] == "e":
            euf.union(b["edge"], b["target"][1])
        else:
            site_extra[b["target"][1]].append(b["edge"])

    # --- decide each site
    reports = []
    for si, s in enumerate(sites):
        ends = list(ends_by_site.get(si, []))
        for eid in site_extra.get(si, []):
            ends.append({"edge": eid, "dir": (0.0, 0.0), "len": 1, "curve": 0.0,
                         "turn": 0.0, "reach": 0})
        deg = len(ends)
        y, x = int(round(s["y"])), int(round(s["x"]))
        y0, y1 = max(0, y - 7), min(H, y + 8)
        x0, x1 = max(0, x - 7), min(W, x + 8)
        win = dist[y0:y1, x0:x1]
        local = float(win.max()) if win.size else 0.0
        score = local / half if half else 0.0
        has_dot = score >= dot_ratio
        hop_pre = 0.0
        if deg >= 4:
            _pp = _pair_by_direction(ends)
            for a, b in _pp:
                hop_pre = max(hop_pre, min(ends[a]["turn"], ends[b]["turn"]))
        unions = []
        if deg <= 2:
            kind = "pass"
            unions = [(ends[0]["edge"], a["edge"]) for a in ends]
        elif deg == 3:
            kind = "T"
            unions = [(ends[0]["edge"], a["edge"]) for a in ends]
        elif has_dot or hop_pre < hop_thresh:
            kind = "dot" if has_dot else "plainX-junction"
            unions = [(ends[0]["edge"], a["edge"]) for a in ends]
        else:
            kind = "cross"
            unions = [(ends[a]["edge"], ends[b]["edge"])
                      for a, b in _pair_by_direction(ends)]
        override = (site_override or {}).get(si)
        if override == "junction":
            kind = "junction*"; unions = [(ends[0]["edge"], a["edge"]) for a in ends]
        elif override == "crossing":
            kind = "crossing*"
            idx = list(range(len(ends))); unions = []
            while len(idx) > 1:
                best = None
                for ii in range(len(idx)):
                    for jj in range(ii + 1, len(idx)):
                        a = ends[idx[ii]]["dir"]; b = ends[idx[jj]]["dir"]
                        c = a[0] * b[0] + a[1] * b[1]
                        if best is None or c < best[0]:
                            best = (c, ii, jj)
                _, ii, jj = best
                unions.append((ends[idx[ii]]["edge"], ends[idx[jj]]["edge"]))
                for t in sorted((ii, jj), reverse=True):
                    idx.pop(t)
        elif isinstance(override, list):
            kind = "manual*"; unions = []
            for grp in override:
                for e2 in grp[1:]:
                    unions.append((grp[0], e2))
        for a, b in unions:
            euf.union(a, b)
        hop = hop_pre
        reports.append({"x": x, "y": y, "degree": deg, "kind": kind,
                        "dot_score": round(score, 2),
                        "hop_score": round(hop, 1),
                        "turns": [e["turn"] for e in ends],
                        "branches": [e["edge"] for e in ends],
                        "dirs": [[round(e["dir"][0], 2), round(e["dir"][1], 2)] for e in ends],
                        "unions": unions})

    # --- ports: edge tips (an edge end not attached to any site) near a box
    ports = defaultdict(list)
    ptol = 5 + pad
    for e in G.edges:
        path = e["path"]
        if len(e["sites"]) == 0:
            tips = [path[0], path[-1]]
        elif len(e["sites"]) == 1 and len(path) > 1:
            tips = list({path[0], path[-1]} - _near_site_pix(path, e, sites))
        else:
            tips = []
        for (ty, tx) in tips:
            for ci, (x1, y1, x2, y2) in enumerate(boxes):
                if (x1 - ptol) <= tx <= (x2 + ptol) and (y1 - ptol) <= ty <= (y2 + ptol):
                    # which side of the box
                    d = {"L": abs(tx - x1), "R": abs(tx - x2),
                         "T": abs(ty - y1), "B": abs(ty - y2)}
                    side = min(d, key=d.get)
                    ports[ci].append({"x": int(tx), "y": int(ty), "edge": e["id"],
                                      "len": len(path), "side": side,
                                      "nsites": len(e["sites"])})
                    break

    for ci in list(ports):
        keep_p = []
        for p in sorted(ports[ci], key=lambda d: (-d["nsites"], -d["len"])):
            if all(abs(p["x"] - k["x"]) + abs(p["y"] - k["y"]) > 8 for k in keep_p):
                keep_p.append(p)
        # a very short dead-end stub that reaches nothing is symbol ink, not a lead
        real = [p for p in keep_p if not (p["nsites"] == 0 and p["len"] < 10)]
        ports[ci] = real if real else keep_p

    net_of_edge = {e["id"]: euf.find(e["id"]) for e in G.edges}
    return {"shape": (H, W), "wire": wire, "skel": skel, "graph": G, "sites": reports,
            "site_pos": sites, "net_of_edge": net_of_edge, "ports": ports,
            "boxes": boxes, "half": half, "dist": dist, "bridges": bridges,
            "site_of_edge": {e["id"]: e["sites"] for e in G.edges}}


# ------------------------------------------------------------------ terminals
def assign_terminals(gt, tr):
    """Pick which port candidate feeds which terminal index, from the drawn
    geometry. Terminal ORDER for 3-pin parts and polarised 2-pin parts still
    needs a human eye; this is the geometric default."""
    out = []
    for ci, c in enumerate(gt["components"]):
        cx, cy, w, h = c["bbox"]
        pts = list(tr["ports"].get(ci, []))
        nterm = len(c["terminals"])
        rec = {"id": c["id"], "class": c["class"], "nports": len(pts), "assign": {},
               "ports": pts}
        horiz = w >= h
        if nterm == 1:
            if pts:
                rec["assign"][0] = max(pts, key=lambda p: (p["nsites"], p["len"]))
        elif nterm == 2:
            best = None
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    a, b = pts[i], pts[j]
                    if horiz:
                        sep = abs(a["x"] - b["x"])
                        opp = (a["x"] - cx) * (b["x"] - cx) < 0
                    else:
                        sep = abs(a["y"] - b["y"])
                        opp = (a["y"] - cy) * (b["y"] - cy) < 0
                    score = sep + (1000 if opp else 0) + 0.2 * (a["len"] + b["len"])
                    if best is None or score > best[0]:
                        best = (score, a, b)
            if best:
                a, b = best[1], best[2]
                p2 = sorted([a, b], key=(lambda p: p["x"]) if horiz else (lambda p: p["y"]))
                rec["assign"][0], rec["assign"][1] = p2[0], p2[1]
            elif len(pts) == 1:
                p = pts[0]
                if horiz:
                    rec["assign"][0 if p["x"] < cx else 1] = p
                else:
                    rec["assign"][0 if p["y"] < cy else 1] = p
        else:
            if len(pts) >= 3:
                p3 = sorted(pts, key=lambda p: (-p["nsites"], -p["len"]))[:3]
                best = None
                for k in range(3):
                    a, b = [p3[i] for i in range(3) if i != k]
                    va = (a["x"] - cx, a["y"] - cy); vb = (b["x"] - cx, b["y"] - cy)
                    na = math.hypot(*va) or 1; nb_ = math.hypot(*vb) or 1
                    cv_ = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb_)
                    if best is None or cv_ < best[0]:
                        best = (cv_, k)
                odd = best[1]
                others = [p3[i] for i in range(3) if i != odd]
                others.sort(key=lambda p: (p["y"], p["x"]))
                rec["assign"][0], rec["assign"][1], rec["assign"][2] = others[0], p3[odd], others[1]
            else:
                for i, p in enumerate(sorted(pts, key=lambda p: (p["y"], p["x"]))):
                    if i < nterm:
                        rec["assign"][i] = p
        out.append(rec)
    return out


def nets_from(gt, tr, assigns):
    raw = {}
    for rec in assigns:
        for ti, p in rec["assign"].items():
            raw[(rec["id"], ti)] = tr["net_of_edge"][p["edge"]]
    gnd = set()
    for c in gt["components"]:
        if c["class"] == "GND" and (c["id"], 0) in raw:
            gnd.add(raw[(c["id"], 0)])
    order, label, i = [], {}, 1
    for v in raw.values():
        if v not in order:
            order.append(v)
    for v in order:
        if v in gnd:
            label[v] = "0"
        else:
            label[v] = f"n{i}"; i += 1
    return {k: label[v] for k, v in raw.items()}, label
