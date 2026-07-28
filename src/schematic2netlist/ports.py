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
    """Assign boundary runs to canonical ports for the best-fitting pose.

    ``run_sites`` is [(node_id, x, y), ...] — one entry per place a
    conductor crosses the component boundary. Returns
    ``(nodes_in_port_order, info)``, or None when no pose fits well
    enough and the caller should fall back to boundary order.
    """
    templates = templates if templates is not None else load_templates()
    tpl = templates.get(cls)
    if not tpl or not run_sites:
        return None
    n_ports = tpl.get("n_ports")
    if not n_ports or len(tpl.get("poses", {})) == 0:
        return None

    diag = float(np.hypot(det["width"], det["height"])) or 1.0
    best = None
    for pose, entry in tpl["poses"].items():
        sites = predicted_sites(cls, det, pose, templates)
        if not sites or len(sites) != n_ports:
            continue
        cost = np.zeros((len(sites), len(run_sites)))
        for i, (sx, sy) in enumerate(sites):
            for j, (_nid, rx, ry) in enumerate(run_sites):
                cost[i, j] = np.hypot(sx - rx, sy - ry)
        rows, cols = linear_sum_assignment(cost)
        total = float(cost[rows, cols].sum())
        mean_dist = total / max(len(rows), 1)
        if best is None or mean_dist < best[0]:
            best = (mean_dist, pose, rows, cols)

    if best is None:
        return None
    mean_dist, pose, rows, cols = best
    if mean_dist > MAX_MEAN_DIST_FRAC * diag:
        return None

    nodes: list = [None] * n_ports
    for i, j in zip(rows, cols):
        nodes[i] = run_sites[j][0]
    info = {
        "pose": pose,
        "mean_dist_frac": round(mean_dist / diag, 4),
        "port_names": tpl.get("port_names"),
        "matched": int(len(rows)),
    }
    return nodes, info
