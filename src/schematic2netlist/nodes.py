"""Electrical node inference: connected components over the wire mask.

Each connected region of wire pixels is one electrical node — junctions,
branches, and crossings are handled implicitly by connectivity.

Migrated verbatim from nodes_mapping_and_netlist.py (v1).
"""

from __future__ import annotations

import cv2
import numpy as np


def build_wire_nodes(
    clean_wires: np.ndarray, connectivity: int = 8
) -> tuple[np.ndarray, int]:
    """Label wire connected-components.

    Returns (node_map, num_nodes) where node_map holds a node id per
    pixel and -1 for background.
    """
    num_labels, labels = cv2.connectedComponents(
        (clean_wires > 0).astype(np.uint8), connectivity=connectivity
    )
    node_map = labels.astype(np.int32) - 1  # background -> -1
    return node_map, num_labels - 1


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _edge_label(node_map: np.ndarray, x1, y1, x2, y2, side: str, band: int) -> int | None:
    """Majority wire-node label in a thin band just OUTSIDE one box edge."""
    h, w = node_map.shape
    if side == "top":
        ys, ye, xs, xe = y1 - band, y1, x1, x2
    elif side == "bottom":
        ys, ye, xs, xe = y2, y2 + band, x1, x2
    elif side == "left":
        ys, ye, xs, xe = y1, y2, x1 - band, x1
    else:  # right
        ys, ye, xs, xe = y1, y2, x2, x2 + band
    ys, xs = max(0, ys), max(0, xs)
    ye, xe = min(h, ye), min(w, xe)
    if ye <= ys or xe <= xs:
        return None
    region = node_map[ys:ye, xs:xe]
    ids = region[region != -1]
    if ids.size == 0:
        return None
    vals, counts = np.unique(ids, return_counts=True)
    return int(vals[counts.argmax()])


def build_wire_nodes_crossover_aware(
    clean_wires: np.ndarray,
    crossover_boxes: list[dict],
    connectivity: int = 8,
    notch_frac: float = 0.6,
    band: int = 4,
    relink: str = "band",
) -> tuple[np.ndarray, int]:
    """Node inference that respects detected wire crossovers.

    At a 4-way crossover the two wires must stay on SEPARATE nets. We
    (1) notch out the box center so the four arms separate under
    connected-components, then (2) reconnect only opposite arms
    (top<->bottom, left<->right) via union-find. This breaks the
    crossing ceiling that no threshold tuning could fix.

    Assumes axis-aligned crossover arms (the case in these drawings);
    a crossover with fewer than two opposite-arm pairs is left as-is.
    """
    base = (clean_wires > 0).astype(np.uint8)

    boxes_xyxy = [bbox_xyxy(det) for det in crossover_boxes]

    # Where the notch is CENTRED decides whether it severs the crossing at
    # all. Taking the box centre makes that a function of box placement:
    # shifting a box 2 px took terminal-pair F1 from 1.0000 to 0.5233 on
    # circuit_1166 and to 0.6582 on circuit_968, because the offset notch
    # no longer covers the intersection and the two nets stay welded.
    # (Re-pairing arms by ring direction instead of edge bands does NOT
    # help — measured byte-identical — which is what localised the cause
    # to the notch rather than the re-link.)
    #
    # The detector box asserts THAT a crossing is here; the skeleton knows
    # WHERE. Centring on the enclosed branch point makes the notch
    # invariant to box placement, since the same intersection is found
    # from anywhere inside the box.
    centres: list[tuple[int, int]] = []
    if relink == "snap" and boxes_xyxy:
        from .skeleton import intersection_sites_with_degree
        sites_all = intersection_sites_with_degree(base)
        for (x1, y1, x2, y2) in boxes_xyxy:
            cx0, cy0 = (x1 + x2) // 2, (y1 + y2) // 2
            inside = [(sx, sy) for sx, sy, _d in sites_all
                      if x1 <= sx <= x2 and y1 <= sy <= y2]
            if inside:
                # nearest enclosed branch point to the box centre
                sx, sy = min(inside, key=lambda p: (p[0] - cx0) ** 2
                             + (p[1] - cy0) ** 2)
                centres.append((sx, sy))
            else:
                centres.append((cx0, cy0))
    else:
        centres = [((x1 + x2) // 2, (y1 + y2) // 2)
                   for (x1, y1, x2, y2) in boxes_xyxy]

    def notch_of(box, i=None):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        nx, ny = int(bw * notch_frac / 2), int(bh * notch_frac / 2)
        ccx, ccy = centres[i] if i is not None else ((x1 + x2) // 2,
                                                     (y1 + y2) // 2)
        return (slice(max(0, ccy - ny), ccy + ny),
                slice(max(0, ccx - nx), ccx + nx))

    def cut(keep):
        m = base.copy()
        for i in keep:
            ys, xs = notch_of(boxes_xyxy[i], i)
            m[ys, xs] = 0
        n, lab = cv2.connectedComponents(m, connectivity=connectivity)
        return n, lab, lab.astype(np.int32) - 1

    def pairs_at_band(node_map, box):
        x1, y1, x2, y2 = box
        e = {s: _edge_label(node_map, x1, y1, x2, y2, s, band)
             for s in ("top", "bottom", "left", "right")}
        return ((e["top"], e["bottom"]) if e["top"] is not None
                and e["bottom"] is not None else None,
                (e["left"], e["right"]) if e["left"] is not None
                and e["right"] is not None else None)

    def pairs_at_angle(node_map, box, ring_frac=0.75, tol_deg=45.0):
        """Pair arms by measured DIRECTION over a ring around the box.

        The band variant votes inside four fixed strips at the box edges,
        which makes the outcome a function of where the box happens to
        sit: shifting a crossover box two pixels took terminal-pair F1
        from 1.0000 to 0.5233 on circuit_1166 and to 0.6582 on
        circuit_968. Sampling a ring instead, and pairing whichever arms
        are most nearly opposite, removes that dependence — an arm is
        found by its geometry rather than by which strip it lands in.
        """
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        R = max(4.0, ring_frac * max(x2 - x1, y2 - y1))
        H, W = node_map.shape
        # accumulate, per wire-node label, the mean offset from the centre
        acc: dict[int, list] = {}
        for k in range(720):
            th = k * np.pi / 360.0
            for rr in (R, R * 1.25):
                px, py = int(round(cx + rr * np.cos(th))), \
                    int(round(cy + rr * np.sin(th)))
                if not (0 <= px < W and 0 <= py < H):
                    continue
                lab = int(node_map[py, px])
                if lab < 0:
                    continue
                acc.setdefault(lab, []).append(
                    (px - cx, py - cy))
        dirs = {}
        for lab, pts in acc.items():
            if len(pts) < 2:
                continue
            vx = float(np.mean([p[0] for p in pts]))
            vy = float(np.mean([p[1] for p in pts]))
            n = (vx * vx + vy * vy) ** 0.5
            if n > 1e-6:
                dirs[lab] = (vx / n, vy / n)
        # NOTE a label appearing on two opposite sides of the ring (an
        # arm pair already joined elsewhere) averages toward zero and is
        # dropped by the norm test above — correct, since it needs no
        # re-link.
        labs = sorted(dirs)
        cand = []
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                d1, d2 = dirs[labs[i]], dirs[labs[j]]
                cand.append((d1[0] * d2[0] + d1[1] * d2[1], labs[i], labs[j]))
        cand.sort(key=lambda t: t[0])
        thr = np.cos(np.deg2rad(180.0 - tol_deg))
        used, out = set(), []
        for dot, a, b in cand:
            if dot > thr:
                break
            if a in used or b in used:
                continue
            out.append((a, b))
            used |= {a, b}
        return ((out[0] if len(out) >= 1 else None),
                (out[1] if len(out) >= 2 else None))

    pairs_at = pairs_at_angle if relink == "angle" else pairs_at_band

    # (0) Skip boxes whose strokes are ALREADY electrically separate.
    # A hop-style crossover is drawn with a visible gap — the ink itself
    # severs the two nets, and connected components already gets it
    # right. Notching there is surgery on a healthy patient: the relink
    # re-pairs arms across the gap and can WELD the nets the drafter
    # deliberately kept apart. Measured (Phase-0 GT-crossover oracle):
    # injecting PERFECT crossover boxes broke five images the pipeline
    # had exactly right (terminal-pair F1 1.0000 -> as low as 0.5233)
    # and fixed none — strict success −0.026, significant. Only a box
    # whose interior ink is one connected blob needs the notch.
    pre_labels = cv2.connectedComponents(base, connectivity=connectivity)[1]
    already_split = set()
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        nm0 = pre_labels.astype(np.int32) - 1
        arms = [_edge_label(nm0, x1, y1, x2, y2, s, band)
                for s in ("top", "bottom", "left", "right")]
        arms = [a for a in arms if a is not None]
        if len({a for a in arms}) >= 2:
            already_split.add(i)

    # (1) Notch, but only KEEP notches that actually re-link both opposite
    # arm pairs. The notch used to be unconditional while the re-link was
    # conditional, so a box that is not a clean 4-way crossing — a T, an
    # L, or a spurious branch point thrown up by thick strokes — had its
    # centre deleted with nothing put back, permanently severing the net.
    # Measured on 30 images: 36% of classifier-proposed boxes and 36% of
    # detector boxes re-link only ONE pair, and 3% re-link none. Reverting
    # those keeps the repair strictly non-destructive: a notch is applied
    # only when the evidence to undo it is present.
    keep = set(range(len(boxes_xyxy))) - already_split
    for _ in range(2):                     # settles immediately in practice
        if not keep:
            break
        _n, _lab, nm_try = cut(keep)
        drop = {i for i in keep if None in pairs_at(nm_try, boxes_xyxy[i])}
        if not drop:
            break
        keep -= drop

    num_labels, labels, node_map = cut(keep)

    # (2) reconnect opposite arms
    uf = _UnionFind(num_labels)
    for i in keep:
        tb, lr = pairs_at(node_map, boxes_xyxy[i])
        if tb is not None:
            uf.union(tb[0] + 1, tb[1] + 1)   # +1: labels are node_map+1
        if lr is not None:
            uf.union(lr[0] + 1, lr[1] + 1)

    # relabel by union-find root, compacted to 0..k-1 (background stays -1).
    # LUT indexed by original CC label (0..num_labels-1; 0 = background).
    remap: dict[int, int] = {}
    next_id = 0
    lut = np.full(num_labels, -1, dtype=np.int32)
    for cc_label in range(1, num_labels):
        root = uf.find(cc_label)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        lut[cc_label] = remap[root]
    out = lut[labels]   # labels holds CC ids; background 0 -> lut[0] = -1
    return out, next_id


def build_wire_nodes_learned(
    clean_wires: np.ndarray,
    crossover_boxes: list[dict],
    weights: str,
    threshold: float = 0.4,
    site_box: int = 15,
    connectivity: int = 8,
    notch_frac: float = 0.6,
    band: int = 4,
    context: float = 3.0,
    min_degree: int = 4,
    thin_input: bool = False,
    relink: str = "band",
) -> tuple[np.ndarray, int, dict]:
    """Net inference that asks a classifier at EVERY stroke intersection.

    The crossover-aware variant can only act where the detector labelled
    a ``Wire Crossover`` — measured at 11% of the intersections actually
    present. Everywhere else connected components assumes touching means
    connected, and that assumption welds nets: 72.6% of wire nodes
    carrying component terminals fuse two or more ground-truth nets.

    Here every intersection is located geometrically, classified, and
    the ones judged crossings are notched and re-linked opposite-arm to
    opposite-arm exactly as detected crossovers are. Detected crossover
    boxes are still honoured unconditionally — the detector's label is
    evidence the drawing itself provides, and is not overridden by the
    model.

    Returns (node_map, num_nodes, info) where ``info`` records how many
    sites were examined and how many were judged crossings, so a run can
    be audited without re-deriving it.
    """
    from .junction_model import crossing_probabilities
    from .skeleton import intersection_sites_with_degree

    # Only 4-way sites are crossing candidates. Notch-and-relink severs
    # the middle and rejoins OPPOSITE arms, which a 3-arm T does not
    # have — notching one orphans its stem permanently. Three wires
    # meeting in a schematic is a junction by definition, so this filter
    # is semantic, not a tuning knob. Measured: roughly half of detected
    # sites are 3-arm, and passing them through was costing more than
    # the classifier won.
    all_sites = intersection_sites_with_degree(clean_wires)
    sites = [(x, y) for x, y, deg in all_sites if deg >= min_degree]
    n_t_sites = len(all_sites) - len(sites)
    labelled = []
    for det in crossover_boxes:
        x1, y1, x2, y2 = bbox_xyxy(det)
        labelled.append((x1, y1, x2, y2))

    def is_labelled(x, y):
        return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in labelled)

    unlabelled = [(x, y) for x, y in sites if not is_labelled(x, y)]
    probs = crossing_probabilities(clean_wires, unlabelled, weights,
                                   context=context, thin_input=thin_input)
    predicted = [pt for pt, p in zip(unlabelled, probs) if p >= threshold]

    # a classified site has no box, so give it one centred on the site;
    # the notch/re-link machinery below is shared with detected boxes
    half = max(3, site_box // 2)
    synthetic = [{"x": float(x), "y": float(y),
                  "width": float(site_box), "height": float(site_box)}
                 for x, y in predicted]

    node_map, n = build_wire_nodes_crossover_aware(
        clean_wires, list(crossover_boxes) + synthetic,
        connectivity=connectivity, notch_frac=notch_frac, band=band,
        relink=relink,
    )
    info = {
        "sites_found": len(all_sites),
        "t_sites_skipped": n_t_sites,
        "crossing_candidates": len(sites),
        "detector_labelled": len(sites) - len(unlabelled),
        "classified": len(unlabelled),
        "judged_crossing": len(predicted),
        "threshold": threshold,
        "mean_crossing_prob": float(probs.mean()) if len(probs) else None,
    }
    return node_map, n, info


def bbox_xyxy(det: dict) -> tuple[int, int, int, int]:
    """Center-based detection dict -> integer (x1, y1, x2, y2)."""
    cx, cy = det["x"], det["y"]
    bw, bh = det["width"], det["height"]
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2


def collect_nodes_in_rect(
    node_map: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> dict[int, int]:
    """Count wire-node pixels per node id inside a rectangle."""
    h, w = node_map.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {}
    region = node_map[y1:y2, x1:x2]
    ids = region[region != -1]
    if ids.size == 0:
        return {}
    uniq, counts = np.unique(ids, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}
