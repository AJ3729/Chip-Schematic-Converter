"""Repair connectivity where an ELECTRICAL constraint proves it is wrong.

Distinct from ``repair.py`` (C5), which never changes the recovered topology and
only adds logged SPICE assumptions. This stage DOES change topology, and it is
allowed to because it acts only where a circuit-level fact -- not a pixel -- says
the current answer cannot be right.

Image evidence for the crossing decision is exhausted: six approaches on 4,822
causally-labelled sites top out at 0.6589 AUC, and two oracles cap the payoff
(perfect GT crossover boxes give strict success 0.3263 against 0.3526; perfect
per-box decisions give 0.0 headroom). So the remaining signal has to come from
somewhere other than the drawing, and two priors from the verified GT are close
to absolute:

    a component with all pins on ONE net    GT rate 0.60%
    a net with only ONE terminal            GT rate 0.00%  (0 of 1509 GT nets)

No currently-strict image contains either, so both are lethal to strict success
and acting on them cannot break an image that already works.

Repairs, each the minimal operation its constraint implies:

  SELF-SHORT   the pins are bridged through the component's own body:
               build_non_wire_mask blanks the bounding box, but a box that is
               slightly too small leaves body ink outside it. Erase an inner
               band -- enough to break the body path, not the leads. The class
               distribution confirms the mechanism: Capacitor 12.4%, I-DC 17.1%,
               V-DC 15.5%, MOSFET-N 15.6%, but Resistor only 0.7%, because a
               zigzag separates its own leads while parallel plates and a source
               circle leave a straight path through.

  ONE-TERMINAL a net with one pin is a fragment of a split net. Bridge it to the
               nearest other node, but only along its own outgoing stroke
               direction and within a bounded gap -- the same collinearity
               safety the global stitcher uses, applied where a constraint
               identified a fault rather than everywhere.

Nothing here reads ground truth. Fixing one violation can expose another, so the
loop re-derives nodes and re-checks, up to ``passes`` times.

Measured on all 190 test images, on top of bridge_span 7:

    strict success   0.3632 -> 0.3842   (+0.0211, 4 gained, 0 lost)
    per-component    0.5733 -> 0.6002   (+0.0269)
    terminal-pair F1 0.7321 -> 0.7442   (+0.0121)

and the gains COMPOUND with connectivity work: the identical repair on the old
span-18 baseline gained only +0.0105 (2 images), because strict success is a
product over components and a repair converts only where the rest of the image
is already right.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from schematic2netlist.classes import class_role, is_ground


def find_violations(comps: list[dict], dets: list[dict]
                    ) -> tuple[list[int], list[str]]:
    """(self-shorted component ids, one-terminal net names)."""
    shorts = []
    for c in comps:
        det = dets[c["id"]]
        if is_ground(det["class"]) or class_role(det["class"]) == "none":
            continue
        nets = [n for n in c.get("node_names", []) if n is not None]
        if len(nets) >= 2 and len(set(nets)) < len(nets):
            shorts.append(c["id"])

    cnt: Counter = Counter()
    for c in comps:
        for n in c.get("node_names", []):
            if n is not None:
                cnt[n] += 1
    return shorts, [n for n, k in cnt.items() if k == 1]


def _erase_body(wires: np.ndarray, det: dict, frac: float) -> bool:
    """Erase an inner band of a component box to break a body bridge."""
    h, w = wires.shape
    bw, bh = det["width"] * frac, det["height"] * frac
    x1, x2 = int(max(0, det["x"] - bw / 2)), int(min(w, det["x"] + bw / 2))
    y1, y2 = int(max(0, det["y"] - bh / 2)), int(min(h, det["y"] + bh / 2))
    if x2 <= x1 or y2 <= y1 or not (wires[y1:y2, x1:x2] > 0).any():
        return False
    wires[y1:y2, x1:x2] = 0
    return True


def _bridge_fragment(wires: np.ndarray, node_map: np.ndarray, nid: int,
                     max_gap: float, dir_tol_deg: float) -> bool:
    """Connect a one-terminal node to the nearest other node, collinearly."""
    mine = node_map == nid
    if not mine.any():
        return False
    other = (node_map >= 0) & (~mine)
    if not other.any():
        return False
    pts, opts = np.argwhere(mine), np.argwhere(other)      # (y, x)
    s_mine = pts[:: max(1, len(pts) // 300)]
    s_oth = opts[:: max(1, len(opts) // 600)]
    d2 = ((s_mine[:, None, :] - s_oth[None, :, :]) ** 2).sum(-1)
    k = int(d2.argmin())
    ia, ib = k // len(s_oth), k % len(s_oth)
    gap = float(np.sqrt(d2[ia, ib]))
    if gap > max_gap or gap < 1:
        return False

    pa, pb = s_mine[ia], s_oth[ib]
    seg = (pb - pa).astype(float)
    seg /= (np.linalg.norm(seg) or 1.0)
    # the fragment's own direction, from its pixels near the endpoint: a lead
    # pointing INTO a component is perpendicular to the target and is refused
    near = pts[np.abs(pts - pa).sum(1) < 15]
    if len(near) >= 4:
        c = near - near.mean(0)
        ev = np.linalg.eigh(c.T @ c)[1][:, -1]
        ev = ev / (np.linalg.norm(ev) or 1.0)
        if abs(float(seg @ ev)) < np.cos(np.radians(dir_tol_deg)):
            return False
    cv2.line(wires, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), 255, 2)
    return True


def repair_connectivity(clean_wires, node_map, comps, detections, cfg, rebuild):
    """Apply constraint-triggered repairs, re-deriving nodes after each pass.

    ``rebuild(wires)`` must return ``(node_map, num_nodes, junction_info,
    comps)`` exactly as the pipeline builds them -- it is passed in rather than
    imported so there is ONE node-construction path. A previous diagnostic that
    re-implemented the dispatch instead silently diverged from the pipeline.

    Returns ``(wires, node_map, num_nodes, junction_info, comps, report)``,
    unchanged apart from ``report`` when nothing fires.
    """
    rcfg = cfg.get("connectivity_repair", {}) or {}
    actions: Counter = Counter()
    num_nodes = junction_info = None
    wires = clean_wires
    touched = False

    for _ in range(int(rcfg.get("passes", 2))):
        shorts, ones = find_violations(comps, detections)
        if not shorts and not ones:
            break
        if not touched:
            wires = clean_wires.copy()      # copy only once, and only if needed
        changed = False

        if rcfg.get("repair_self_shorts", True):
            for cid in shorts:
                if _erase_body(wires, detections[cid],
                               float(rcfg.get("body_frac", 0.5))):
                    actions["erased_self_short_body"] += 1
                    changed = True

        if rcfg.get("repair_one_terminal_nets", True):
            name_to_id = {}
            for c in comps:
                for n, nn in zip(c.get("nodes", []), c.get("node_names", [])):
                    if n is not None and nn is not None:
                        name_to_id[nn] = int(n)
            for nn in ones:
                nid = name_to_id.get(nn)
                if nid is None:
                    continue
                if _bridge_fragment(wires, node_map, nid,
                                    float(rcfg.get("max_gap", 60)),
                                    float(rcfg.get("dir_tol_deg", 40))):
                    actions["bridged_one_terminal_net"] += 1
                    changed = True

        if not changed:
            break
        touched = True
        node_map, num_nodes, junction_info, comps = rebuild(wires)

    return wires, node_map, num_nodes, junction_info, comps, {
        "applied": bool(touched),
        "actions": dict(actions),
        "n_actions": int(sum(actions.values())),
    }
