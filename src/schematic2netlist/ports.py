"""Port-template terminal localization (contribution C3, MSP path).

Digitize-HCD ships per-class component crops with pixel **port
coordinates** and, for directional parts, **port names** (Anode/Cathode,
Positive/Negative, Drain/Gate/Source, In+/In-/Out). No published work
uses this modality. `scripts/build_port_templates.py` distills it into
`configs/port_templates.json`: per class, per pose bin, the median
normalized position of each port in canonical identity order.

This module applies those templates at inference. The problem it solves
is not *where* the wires are — boundary snapping already finds those —
but **which terminal is which**. Previously a component's terminals were
filled in the order the boundary walk happened to encounter them, so a
diode's anode and cathode (and a MOSFET's drain, gate and source) were
assigned arbitrarily, and the emitted SPICE netlist could not be
trusted for any directional device.

Matching works as follows. Boundary-crossing runs are located as usual;
each candidate pose's predicted pin sites are then scaled onto the
detected box and matched to those runs by Hungarian assignment on
distance. The pose with the lowest total distance wins, and its port
ORDER is the terminal order — so terminal *k* really is the *k*-th
named port for that class. When no template exists, too few runs are
found, or the best pose is a poor fit, the caller falls back to plain
boundary snapping: identity is a bonus, never a regression in
connectivity.

Note the honest limit, quantified in `results/ports/template_accuracy.json`:
pose selection here uses wire evidence, not symbol appearance, so
single-port classes (GND, one-port sources) carry no orientation signal
at all, and multi-terminal poses are only as good as the boundary
evidence. Closing that gap is what the learned port-heatmap model
([IDEAL]) would buy.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PORT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "port_templates.json"
)

# A pose whose mean pin-to-run distance exceeds this fraction of the
# box diagonal is not trusted; the caller falls back.
MAX_MEAN_DIST_FRAC = 0.45

# Ranks an out-of-range candidate behind reusing an in-range node. In box
# diagonals, so it must exceed the reuse penalty (1 diagonal per extra
# use) by enough that no realistic number of pins reorders the two.
OUT_OF_RANGE_PENALTY = 10.0


@lru_cache(maxsize=1)
def load_templates(path: str | None = None) -> dict:
    p = Path(path) if path else PORT_TEMPLATE_PATH
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def port_names(cls: str, templates: dict | None = None) -> list[str] | None:
    """Canonical port names for a class, or None if it is symmetric."""
    tpl = (templates if templates is not None else load_templates()).get(cls)
    return tpl.get("port_names") if tpl else None


def predicted_sites(cls: str, det: dict, pose: str, templates: dict) -> list | None:
    """Template pin sites for one pose, scaled onto a detected box."""
    tpl = templates.get(cls)
    if not tpl:
        return None
    entry = tpl["poses"].get(pose)
    if not entry:
        return None
    x1 = det["x"] - det["width"] / 2
    y1 = det["y"] - det["height"] / 2
    return [
        (x1 + p["x"] * det["width"], y1 + p["y"] * det["height"])
        for p in entry["ports"]
    ]


def match_ports(
    cls: str, det: dict, run_sites: list[tuple[int, float, float]],
    templates: dict | None = None,
) -> tuple[list, dict] | None:
    """Assign boundary crossings to canonical ports for the best pose.

    ``run_sites`` is [(node_id, x, y), ...] — one entry per place a
    conductor crosses the component boundary. Returns
    ``(nodes_in_port_order, info)``, or None when no pose fits well
    enough and the caller should fall back to boundary order.

    **Assignment is over distinct NODES, not over runs.** One net often
    crosses a component's boundary several times — it may hug the body,
    or reach two pins from the same side — so the run list contains
    repeats. Matching ports to runs then lets several ports win several
    runs *of the same net*, and nothing in the geometry objects. Measured
    on oracle mode C, where connectivity is perfect and the ring saw all
    three distinct nodes: a MOSFET-N whose boundary read
    ``[n1, n4, n4, n3, n4]`` had every one of Drain/Gate/Source assigned
    to ``n4``, at a mean fit of 0.13 box diagonals — a confident fit, so
    the trust check below never fired and no fallback happened. Every
    such collapse was a three-terminal device, and duplicate-node
    collapse accounted for ALL of the set-level snapping error
    (``scripts/diagnose_snapping.py``).

    Two pins genuinely sharing a net is real but rare — 0.60% of GT
    components with two or more terminals, concentrated in
    diode-connected transistors and Op-Amps. So distinct nodes are
    preferred rather than required: each distinct node offers one slot,
    and extra slots appear only when the component has fewer distinct
    nodes than pins. When there are enough nodes to go around the
    assignment is strictly one-net-per-pin; when there are not, reuse is
    allowed and the ``+ k * diag`` term orders which net doubles up.

    Preferring distinctness *unconditionally* is wrong on real wires, and
    measurably so. With perfect connectivity every node at the boundary
    belongs to the component, so spreading pins across them is always
    right. With predicted connectivity a weld can leave only one genuine
    node on the ring, and an unbounded preference for distinctness then
    recruits whatever second node is nearest — including an unrelated
    wire passing by — because a full box diagonal of reuse penalty
    outweighs any distance. Measured on the oracle: unbounded
    distinctness lifted mode C per-component accuracy by +0.1296 but cost
    mode B −0.0088 and mode A −0.0040. Candidates beyond the same
    ``MAX_MEAN_DIST_FRAC`` the pose-trust check already uses are
    therefore pushed behind reuse, so a distant node is never recruited
    merely to satisfy distinctness.
    """
    templates = templates if templates is not None else load_templates()
    tpl = templates.get(cls)
    if not tpl or not run_sites:
        return None
    n_ports = tpl.get("n_ports")
    if not n_ports or len(tpl.get("poses", {})) == 0:
        return None

    diag = float(np.hypot(det["width"], det["height"])) or 1.0

    # collapse runs to distinct nodes; a node keeps ALL its crossing
    # locations so each port can be scored against the nearest one
    node_pts: dict[int, list] = {}
    for nid, rx, ry in run_sites:
        node_pts.setdefault(int(nid), []).append((rx, ry))
    nodes_uniq = sorted(node_pts)
    # one slot per node, plus duplicates only if the pins outnumber nodes
    repeats = max(1, -(-n_ports // len(nodes_uniq)))
    slots = [(nid, k) for k in range(repeats) for nid in nodes_uniq]

    best = None
    for pose, entry in tpl["poses"].items():
        sites = predicted_sites(cls, det, pose, templates)
        if not sites or len(sites) != n_ports:
            continue
        true_d = np.zeros((n_ports, len(slots)))
        cost = np.zeros((n_ports, len(slots)))
        for i, (sx, sy) in enumerate(sites):
            for j, (nid, k) in enumerate(slots):
                d = min(float(np.hypot(sx - rx, sy - ry))
                        for rx, ry in node_pts[nid])
                true_d[i, j] = d
                # a candidate too far to be this pin's conductor sits behind
                # reuse, so distinctness never drags in a passing wire
                far = OUT_OF_RANGE_PENALTY if d > MAX_MEAN_DIST_FRAC * diag \
                    else 0.0
                cost[i, j] = d + k * diag + far * diag
        rows, cols = linear_sum_assignment(cost)
        # judge the pose on real distance; the reuse penalty is a tie-break
        # for WHICH net doubles up, not evidence about pose quality
        mean_dist = float(true_d[rows, cols].sum()) / max(len(rows), 1)
        if best is None or mean_dist < best[0]:
            best = (mean_dist, pose, rows, cols)

    if best is None:
        return None
    mean_dist, pose, rows, cols = best
    if mean_dist > MAX_MEAN_DIST_FRAC * diag:
        return None

    nodes: list = [None] * n_ports
    for i, j in zip(rows, cols):
        nodes[i] = slots[j][0]
    info = {
        "pose": pose,
        "mean_dist_frac": round(mean_dist / diag, 4),
        "port_names": tpl.get("port_names"),
        "matched": int(len(rows)),
        "n_distinct_nodes": len(nodes_uniq),
        "reused_nodes": int(len(nodes) - len(set(nodes))),
    }
    return nodes, info
