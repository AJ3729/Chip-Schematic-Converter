"""Net assembly by wire geometry instead of pixel adjacency (C2, vector).

Connected components answers "do these strokes touch?" when net assembly
needs "do they CONNECT?". Those differ exactly at a crossing, and the
whole measured connectivity deficit lives there: 72.6% of wire nodes
carrying component terminals fuse two or more ground-truth nets.

Everything tried in raster space has failed for the same structural
reason. Notch-and-relink deletes ink and re-pairs arms afterwards, which
is destructive and irreversible: injecting PERFECT crossover boxes
*lowered* strict success by 0.026 (five images at terminal-pair F1
1.0000 dropped, one to 0.5233), because a drawn hop already has an ink
gap and notching welds the nets the drafter kept apart. A patch
classifier over raster sites failed too (-0.110 terminal-pair F1), partly
because ~40% of raster "intersections" are morphology artifacts rather
than places two wires meet.

This module decides connectivity from the *arm geometry* of the wire
skeleton, and never edits ink:

1. Thin to a skeleton; cluster branch pixels into sites.
2. Cut site neighbourhoods out of the skeleton. What remains, per
   connected component, is an **arm** — a run of wire between sites.
3. Prune spurs (short arms with a free end): these are the thick-stroke
   and stitch artifacts that inflated site degree and poisoned the
   classifier's inputs.
4. At each site, measure each incident arm's direction and decide:
   - a **junction dot** (locally thick ink) -> all arms one net;
   - a detected `Wire Crossover` box -> straight-through pairing;
   - exactly four arms forming two opposite collinear pairs -> two
     nets, paired top/bottom and left/right;
   - anything else (T, corner, ambiguous) -> junction.
5. Union arms accordingly, then propagate arm labels to every wire pixel
   by nearest-arm, producing the same ``node_map`` contract the rest of
   the pipeline consumes (int32, -1 background, labels 0..n-1).

The split policy is deliberately conservative. Over-merging is the
disease, but a wrong split is the more damaging cure: it severs a net and
corrupts every component on it. So a site is only split on *positive*
collinearity evidence, and every ambiguous case stays a junction —
exactly the direction of caution the notch experiments established.

Nothing here deletes or bridges ink, so the operation is reversible by
construction: a wrong decision mislabels a region, it does not destroy
evidence.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .skeleton import thin

_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int32)


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _sites(skel: np.ndarray, min_sep: int) -> list[tuple[int, int]]:
    """Cluster branch pixels (skeleton degree >= 3) into one site each."""
    neigh = ndimage.convolve(skel.astype(np.int32), _NEIGH, mode="constant")
    branch = ((skel > 0) & (neigh >= 3)).astype(np.uint8)
    if not branch.any():
        return []
    grown = cv2.dilate(branch, np.ones((min_sep, min_sep), np.uint8))
    n, _lab, _stats, cents = cv2.connectedComponentsWithStats(grown, 8)
    return [(int(round(x)), int(round(y))) for x, y in cents[1:n]]


def _stroke_width(wires: np.ndarray, skel: np.ndarray) -> float:
    """Median stroke half-width: distance from skeleton to background."""
    dist = cv2.distanceTransform((wires > 0).astype(np.uint8), cv2.DIST_L2, 3)
    v = dist[skel > 0]
    return float(np.median(v)) if v.size else 1.0


def build_wire_nodes_vector(
    clean_wires: np.ndarray,
    crossover_boxes: list[dict] | None = None,
    connectivity: int = 8,
    site_radius_frac: float = 2.2,
    min_sep: int = 9,
    spur_frac: float = 3.0,
    arm_probe_frac: float = 4.0,
    collinear_deg: float = 30.0,
    dot_ratio: float = 1.75,
    link_gap: int = 0,
    link_collinear_deg: float = 25.0,
    split_geometry: bool = False,
    classifier_weights: str | None = None,
    classifier_threshold: float = 0.5,
    classifier_context: float = 3.0,
    classifier_min_degree: int = 3,
) -> tuple[np.ndarray, int, dict]:
    """Label wire pixels into electrical nets from skeleton arm geometry.

    Returns ``(node_map, num_nodes, info)``. ``info`` records the site
    decisions so a run is auditable without re-deriving it.

    Lengths scale with the measured stroke width rather than being fixed
    pixel counts, so the same settings hold at 512 and 1024 px.
    """
    wires = (clean_wires > 0).astype(np.uint8)
    if not wires.any():
        return np.full(clean_wires.shape, -1, np.int32), 0, {"sites": 0}

    skel = thin(wires)
    if not skel.any():
        return np.full(clean_wires.shape, -1, np.int32), 0, {"sites": 0}

    hw = max(_stroke_width(wires, skel), 1.0)
    site_r = max(3, int(round(hw * site_radius_frac)))
    spur_len = max(4, int(round(hw * spur_frac)))
    probe_r = max(5, int(round(hw * arm_probe_frac)))

    sites = _sites(skel, min_sep)

    # Merge sites whose disks would overlap. Branch clustering separates
    # sites by ``min_sep``, but the disks cut below have radius site_r
    # independently, so two nearby sites could swallow the entire arm
    # BETWEEN them — leaving the nets on either side with no connector and
    # fragmenting what connected components had right. Measured: this was
    # the whole deficit of the first version (fragmentation 1.46 vs the
    # CC baseline's 1.25). Overlapping sites are one site.
    if sites:
        merged: list[list[tuple[int, int]]] = []
        uf_s = _UF(len(sites))
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                dx = sites[i][0] - sites[j][0]
                dy = sites[i][1] - sites[j][1]
                if dx * dx + dy * dy <= (2 * site_r + 2) ** 2:
                    uf_s.union(i, j)
        groups: dict[int, list[tuple[int, int]]] = {}
        for i, p in enumerate(sites):
            groups.setdefault(uf_s.find(i), []).append(p)
        merged = list(groups.values())
        sites = [(int(round(sum(p[0] for p in g) / len(g))),
                  int(round(sum(p[1] for p in g) / len(g)))) for g in merged]
        # a merged cluster spans further than one disk; grow to cover it
        site_span = []
        for g, c in zip(merged, sites):
            rad = site_r + max(
                (int(round(max(abs(p[0] - c[0]), abs(p[1] - c[1]))))
                 for p in g), default=0)
            site_span.append(rad)
    else:
        site_span = []

    # ---- cut sites out of the skeleton; the remainder are arms ----------
    cut = skel.copy()
    H, W = skel.shape
    for (x, y), rad in zip(sites, site_span):
        cv2.circle(cut, (x, y), rad, 0, -1)
    n_arm, arm_lab = cv2.connectedComponents(cut, connectivity=connectivity)
    # arm ids are 1..n_arm-1; 0 is background

    arm_size = np.bincount(arm_lab.ravel(), minlength=n_arm)

    # which sites does each arm touch? (dilate the arm, look for sites)
    site_of = np.full((H, W), -1, np.int32)
    for i, ((x, y), rad) in enumerate(zip(sites, site_span)):
        cv2.circle(site_of, (x, y), rad, i, -1)

    arms_at: dict[int, set[int]] = {}
    touches: dict[int, set[int]] = {}
    if sites:
        # a pixel adjacent to a site disk belongs to an arm incident on it
        # 5x5, not 3x3: a skeleton entering the disk DIAGONALLY has its
        # first surviving pixel ~1.4 px out, which a 1-px ring misses.
        # A missed incidence orphans the arm and fragments a net that
        # connected components had right — the measured deficit of the
        # first version (fragmentation 1.33 vs the CC baseline's 1.25).
        ring = cv2.dilate((site_of >= 0).astype(np.uint8),
                          np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero((ring > 0) & (arm_lab > 0))
        for y, x in zip(ys, xs):
            a = int(arm_lab[y, x])
            # nearest site among the disks overlapping this neighbourhood
            y0, y1 = max(0, y - 3), min(H, y + 4)
            x0, x1 = max(0, x - 3), min(W, x + 4)
            win = site_of[y0:y1, x0:x1]
            cand = win[win >= 0]
            if cand.size == 0:
                continue
            s = int(np.bincount(cand).argmax())
            arms_at.setdefault(s, set()).add(a)
            touches.setdefault(a, set()).add(s)

    # ---- prune spurs: short arms with a free end -------------------------
    # These are thick-stroke and stitching residue. Left in, they inflate
    # site degree (a 4-way site reads as 5-way) and defeat pairing.
    pruned = set()
    for a in range(1, n_arm):
        if arm_size[a] <= spur_len and len(touches.get(a, ())) <= 1:
            pruned.add(a)
    for s in arms_at:
        arms_at[s] -= pruned

    # ---- decide each site ----------------------------------------------
    uf = _UF(n_arm)
    dist = cv2.distanceTransform(wires, cv2.DIST_L2, 3)
    xover = []
    for det in (crossover_boxes or []):
        cx, cy = det["x"], det["y"]
        bw, bh = det["width"], det["height"]
        xover.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

    def arm_dir(a: int, sx: int, sy: int):
        """Unit vector from the site toward arm ``a``'s nearby pixels."""
        y0, y1 = max(0, sy - probe_r), min(H, sy + probe_r + 1)
        x0, x1 = max(0, sx - probe_r), min(W, sx + probe_r + 1)
        sub = arm_lab[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub == a)
        if ys.size == 0:
            return None
        vx = float(np.mean(xs + x0) - sx)
        vy = float(np.mean(ys + y0) - sy)
        n = (vx * vx + vy * vy) ** 0.5
        return None if n < 1e-6 else (vx / n, vy / n)

    cos_thr = np.cos(np.deg2rad(180.0 - collinear_deg))   # ~ -0.87 @30deg
    counts = {"junction_dot": 0, "detector_crossing": 0,
              "geometric_crossing": 0, "junction_default": 0,
              "classifier_crossing": 0, "classifier_junction": 0}
    split_sites: list[int] = []      # sites where arms were NOT all unioned

    # Learned per-site decision. One batched forward pass per image rather
    # than a call per site: inference runs on CPU at ~20-60 sites/image
    # across the whole benchmark, so per-site overhead would dominate.
    #
    # Unlike the notch, a wrong answer here mislabels rather than destroys —
    # arms are simply not unioned, no ink is cut — which is why a classifier
    # is worth attaching to THIS mechanism and was not to that one. It also
    # applies at ANY degree (default >= 3), reaching the 41% of weld cut
    # points that are degree-3 T-sites and that every previous mechanism
    # refused to consider.
    site_prob: dict[int, float] = {}
    if classifier_weights and sites:
        from .junction_model import crossing_probabilities
        cand_idx = [i for i, (sx, sy) in enumerate(sites)
                    if len(arms_at.get(i, ())) >= classifier_min_degree]
        if cand_idx:
            probs = crossing_probabilities(
                wires * 255, [sites[i] for i in cand_idx],
                classifier_weights, context=classifier_context)
            site_prob = {i: float(p) for i, p in zip(cand_idx, probs)}

    for i, (sx, sy) in enumerate(sites):
        arms = sorted(arms_at.get(i, ()))
        if len(arms) < 2:
            continue
        dirs = {a: arm_dir(a, sx, sy) for a in arms}
        arms = [a for a in arms if dirs[a] is not None]
        if len(arms) < 2:
            continue

        in_box = any(x1 <= sx <= x2 and y1 <= sy <= y2
                     for x1, y1, x2, y2 in xover)

        # (a) learned decision, where available. It outranks the dot
        # heuristic because the training renders CONTAIN dots, so the model
        # subsumes that cue while also seeing arm geometry and context the
        # thickness test cannot. The detector box still outranks both.
        p_cross = site_prob.get(i)
        if not in_box and p_cross is not None:
            if p_cross >= classifier_threshold:
                if not pairs:
                    order = sorted(cand, key=lambda t: t[0])
                    seen2: set[int] = set()
                    for _dot, a, b in order:
                        if a in seen2 or b in seen2:
                            continue
                        pairs.append((a, b))
                        seen2 |= {a, b}
                        if len(pairs) == 2:
                            break
                for a, b in pairs:
                    uf.union(a, b)
                rest = [a for a in arms if all(a not in q for q in pairs)]
                for a in rest[1:]:
                    uf.union(rest[0], a)
                counts["classifier_crossing"] += 1
                split_sites.append(i)
            else:
                for a in arms[1:]:
                    uf.union(arms[0], a)
                counts["classifier_junction"] += 1
            continue

        # (b) junction dot: ink locally much thicker than the stroke.
        # Checked AFTER the box test, and skipped inside a box: a detected
        # Wire Crossover is direct evidence about this site, while
        # thickness is a heuristic that a crossing of two thick strokes
        # trips by itself (two 4-px strokes meeting measure ~3.8 px against
        # a 3.3 px threshold). With the old ordering a false dot silently
        # overrode the detector — circuit_968 split only 1 of its 2 boxes,
        # and 30 of 49 sites on circuit_470 were called dots.
        if not in_box and float(dist[sy, sx]) >= hw * dot_ratio:
            for a in arms[1:]:
                uf.union(arms[0], a)
            counts["junction_dot"] += 1
            continue

        # (b) pair opposite arms greedily by collinearity
        cand = []
        for ii in range(len(arms)):
            for jj in range(ii + 1, len(arms)):
                d1, d2 = dirs[arms[ii]], dirs[arms[jj]]
                cand.append((d1[0] * d2[0] + d1[1] * d2[1], arms[ii], arms[jj]))
        cand.sort(key=lambda t: t[0])          # most opposite first
        used, pairs = set(), []
        for dot, a, b in cand:
            if dot > cos_thr:                  # not opposite enough
                break
            if a in used or b in used:
                continue
            pairs.append((a, b))
            used |= {a, b}

        # (c) commit. Split ONLY on positive evidence: a detector box with
        # at least one straight-through pair, or a clean 4-arm site fully
        # decomposed into two opposite pairs. Everything else is a
        # junction, because a wrong split severs a net.
        if in_box:
            # A detected Wire Crossover box is the one location evidence we
            # trust: the pixel-notch experiments showed the box centre is
            # accurate to ~2 px where skeleton branch points are ~6 px off,
            # and per-box honour decisions already have zero headroom. So a
            # box site splits unconditionally, pairing whichever arms are
            # MOST opposite even if none clears `collinear_deg` — refusing
            # to split here just discards the evidence. Unlike notching,
            # nothing is cut: the arms are simply not unioned, so a wrong
            # call mislabels rather than destroys.
            if not pairs:
                order = sorted(cand, key=lambda t: t[0])
                seen: set[int] = set()
                for _dot, a, b in order:
                    if a in seen or b in seen:
                        continue
                    pairs.append((a, b))
                    seen |= {a, b}
                    if len(pairs) == 2:
                        break
            for a, b in pairs:
                uf.union(a, b)
            leftover = [a for a in arms
                        if all(a not in p for p in pairs)]
            for a in leftover[1:]:
                uf.union(leftover[0], a)
            counts["detector_crossing"] += 1
            split_sites.append(i)
        elif split_geometry and len(arms) == 4 and len(pairs) == 2 \
                and len(used) == 4:
            for a, b in pairs:
                uf.union(a, b)
            counts["geometric_crossing"] += 1
            split_sites.append(i)
        else:
            for a in arms[1:]:
                uf.union(arms[0], a)
            counts["junction_default"] += 1

    # ---- optional endpoint linking across pen-lift gaps -----------------
    n_linked = 0
    if link_gap > 0:
        ends = []           # (arm, x, y, inward unit dir)
        neigh = ndimage.convolve(cut.astype(np.int32), _NEIGH, mode="constant")
        ys, xs = np.nonzero((cut > 0) & (neigh == 1))
        for y, x in zip(ys, xs):
            a = int(arm_lab[y, x])
            if a in pruned or a == 0:
                continue
            d = arm_dir(a, x, y)
            if d is not None:
                ends.append((a, int(x), int(y), d))
        lc = np.cos(np.deg2rad(180.0 - link_collinear_deg))
        for p in range(len(ends)):
            for q in range(p + 1, len(ends)):
                a1, x1, y1, d1 = ends[p]
                a2, x2, y2, d2 = ends[q]
                if uf.find(a1) == uf.find(a2):
                    continue
                gap = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if gap > link_gap or gap < 1:
                    continue
                # each end must point at the other, and the two must be
                # roughly anti-parallel (a continuing straight run)
                ux, uy = (x2 - x1) / gap, (y2 - y1) / gap
                if (d1[0] * ux + d1[1] * uy) > -0.7:
                    continue
                if (d2[0] * -ux + d2[1] * -uy) > -0.7:
                    continue
                if (d1[0] * d2[0] + d1[1] * d2[1]) > lc:
                    continue
                uf.union(a1, a2)
                n_linked += 1

    # ---- compact labels, propagate to every wire pixel ------------------
    root_id: dict[int, int] = {}
    arm_final = np.full(n_arm, -1, np.int32)
    for a in range(1, n_arm):
        if a in pruned:
            continue
        r = uf.find(a)
        if r not in root_id:
            root_id[r] = len(root_id)
        arm_final[a] = root_id[r]

    labelled = arm_final[arm_lab]                     # -1 off-arm
    labelled[arm_lab == 0] = -1

    node_map = np.full(skel.shape, -1, np.int32)
    src = labelled >= 0
    if src.any():
        # nearest labelled skeleton pixel wins; only wire pixels get a label
        _d, (iy, ix) = ndimage.distance_transform_edt(
            ~src, return_indices=True)
        node_map = labelled[iy, ix]
        node_map[wires == 0] = -1
        # At a SPLIT site the disk holds pixels of both nets, so filling it
        # by nearest-arm paints a wedge of each and a terminal snapping
        # nearby can pick up either. Pixel notching sidesteps this by
        # deleting those pixels outright; do the same here — leave the disk
        # unlabelled — so the two nets stay cleanly separated without any
        # of the notch's placement sensitivity. Junction sites keep their
        # fill, since there the arms are one net anyway.
        if split_sites:
            blank = np.zeros(node_map.shape, np.uint8)
            for i in split_sites:
                cv2.circle(blank, sites[i], site_span[i], 1, -1)
            node_map[blank > 0] = -1
    info = {
        "sites": len(sites),
        "arms": int(max(n_arm - 1, 0)),
        "spurs_pruned": len(pruned),
        "stroke_half_width": round(hw, 2),
        "endpoint_links": n_linked,
        **counts,
    }
    return node_map.astype(np.int32), len(root_id), info
