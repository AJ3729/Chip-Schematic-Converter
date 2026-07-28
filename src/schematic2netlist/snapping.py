"""Terminal snapping: associate component terminals with wire nodes.

Two already-implemented strategies from the legacy variants (ablation E4):

- "directional" (v1, nodes_mapping_and_netlist.py): orientation-aware.
  A component whose bbox is wider than tall is assumed horizontal; snap
  windows extend outward from its left/right (else top/bottom) edges.
- "uniform" (v2, nodes_mapping_and_netlist2.py): expand the whole bbox
  outward in steps and take the two nodes with the strongest pixel
  support as soon as any are found.

Both migrated verbatim; thresholds come from the `snapping` config.
"""

from __future__ import annotations

import numpy as np

from schematic2netlist import ports as ports_mod
from schematic2netlist.classes import (
    canonical_class,
    class_role,
    class_terminals,
    is_ground,
)
from schematic2netlist.nodes import bbox_xyxy, collect_nodes_in_rect


def snap_directional(
    det: dict, node_map: np.ndarray, cfg: dict
) -> list[int | None]:
    """v1: orientation-aware snap windows on two opposite bbox edges."""
    s = cfg["snapping"]
    max_expand = s["max_expand"]
    depth = s["window_depth"]

    x1, y1, x2, y2 = bbox_xyxy(det)
    horizontal = (x2 - x1) >= (y2 - y1)

    if horizontal:
        first = collect_nodes_in_rect(node_map, x1 - max_expand, y1, x1 + depth, y2)
        second = collect_nodes_in_rect(node_map, x2 - depth, y1, x2 + max_expand, y2)
    else:
        first = collect_nodes_in_rect(node_map, x1, y1 - max_expand, x2, y1 + depth)
        second = collect_nodes_in_rect(node_map, x1, y2 - depth, x2, y2 + max_expand)

    def best_node(hits: dict[int, int]) -> int | None:
        if not hits:
            return None
        return max(hits, key=hits.get)

    return [best_node(first), best_node(second)]


def snap_uniform(
    det: dict, node_map: np.ndarray, cfg: dict, max_expand: int | None = None
) -> list[int | None]:
    """v2: grow the whole bbox until nodes appear; take the top 2."""
    s = cfg["snapping"]
    if max_expand is None:
        max_expand = s["max_expand"]
    step = s["expand_step"]

    x1, y1, x2, y2 = bbox_xyxy(det)
    for expand in range(step, max_expand + 1, step):
        hits = collect_nodes_in_rect(
            node_map, x1 - expand, y1 - expand, x2 + expand, y2 + expand
        )
        if hits:
            ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
            nodes: list[int | None] = [nid for nid, _ in ranked[:2]]
            if len(nodes) == 1:
                nodes = [nodes[0], None]
            return nodes
    return [None, None]


def find_ground_node(
    det: dict, node_map: np.ndarray, cfg: dict
) -> int | None:
    """v1 ground snap: expand a square search until any node is hit."""
    s = cfg["snapping"]
    x1, y1, x2, y2 = bbox_xyxy(det)
    for expand in range(s["expand_step"], s["ground_max_expand"] + 1, s["expand_step"]):
        hits = collect_nodes_in_rect(
            node_map, x1 - expand, y1 - expand, x2 + expand, y2 + expand
        )
        if hits:
            return max(hits, key=hits.get)
    return None


def _perimeter(x1: int, y1: int, x2: int, y2: int, shape) -> list:
    """Ordered (x, y) walk around a rectangle, clipped to the image."""
    H, W = shape
    pts = []
    for x in range(x1, x2 + 1):
        pts.append((x, y1))
    for y in range(y1 + 1, y2 + 1):
        pts.append((x2, y))
    for x in range(x2 - 1, x1 - 1, -1):
        pts.append((x, y2))
    for y in range(y2 - 1, y1, -1):
        pts.append((x1, y))
    return [(x, y) for x, y in pts if 0 <= x < W and 0 <= y < H]


def _boundary_runs(node_map: np.ndarray, x1, y1, x2, y2) -> list:
    """Contiguous runs of the same wire node along the rectangle walk.

    Each run is one place a conductor crosses the component boundary —
    i.e. one terminal. Returns [(node_id, run_length), ...] sorted by
    length descending. The walk is circular, so a run spanning the
    start/end seam is merged.
    """
    pts = _perimeter(x1, y1, x2, y2, node_map.shape)
    if not pts:
        return []
    ids = [int(node_map[y, x]) for x, y in pts]

    runs = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            if ids[start] != -1:
                runs.append([ids[start], i - start, start, i - 1])
            start = i
    # merge across the seam if the walk begins and ends on the same node
    if len(runs) > 1 and runs[0][2] == 0 and runs[-1][3] == len(ids) - 1 \
            and runs[0][0] == runs[-1][0]:
        runs[0][1] += runs[-1][1]
        runs.pop()

    merged: dict[int, int] = {}
    for nid, length, _, _ in runs:
        merged[nid] = merged.get(nid, 0) + length
    ordered = sorted(runs, key=lambda r: -r[1])
    seen, out = set(), []
    for nid, length, _, _ in ordered:
        if nid not in seen:
            seen.add(nid)
            out.append((nid, merged[nid]))
    return out


def _boundary_run_sites(node_map: np.ndarray, x1, y1, x2, y2) -> list:
    """Boundary runs as [(node_id, x, y), ...] at each run's midpoint.

    Same walk as :func:`_boundary_runs`, but keeping WHERE each crossing
    happened — which is what port-template matching needs in order to
    decide which crossing is which named pin.
    """
    pts = _perimeter(x1, y1, x2, y2, node_map.shape)
    if not pts:
        return []
    ids = [int(node_map[y, x]) for x, y in pts]

    runs = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            if ids[start] != -1:
                runs.append((ids[start], start, i - 1))
            start = i
    if len(runs) > 1 and runs[0][1] == 0 and runs[-1][2] == len(ids) - 1 \
            and runs[0][0] == runs[-1][0]:
        # seam-spanning run: represent it by the later arc's midpoint
        nid, s, _e = runs[-1]
        runs = runs[1:-1] + [(nid, s, len(ids) - 1)]

    out = []
    for nid, s, e in runs:
        mx, my = pts[(s + e) // 2]
        out.append((nid, float(mx), float(my)))
    return out


def snap_boundary(
    det: dict, node_map: np.ndarray, cfg: dict, n_terminals: int
) -> list:
    """Boundary-crossing snapping (contribution C2).

    Reads terminals the way a human reads a schematic: wherever a
    conductor crosses the component's boundary, that is a pin. This
    replaces the two fixed edge probes of the directional/uniform
    strategies, which structurally could not represent a third terminal
    — so every MOSFET, BJT and Op-Amp lost a pin regardless of how good
    the wire mask was (measured: 18% of components).

    The boundary ring is grown outward until it sees at least the number
    of distinct nodes the class expects, which also bridges the small
    whitespace gaps hand-drawn symbols leave between a symbol and its
    wire.
    """
    s = cfg["snapping"]
    step = s["expand_step"]
    max_expand = s["max_expand"]

    x1, y1, x2, y2 = bbox_xyxy(det)
    best: list = []
    for r in range(step, max_expand + 1, step):
        runs = _boundary_runs(node_map, x1 - r, y1 - r, x2 + r, y2 + r)
        if len(runs) > len(best):
            best = runs
        if len(runs) >= n_terminals:
            break

    nodes = [nid for nid, _ in best[:n_terminals]]
    nodes += [None] * (n_terminals - len(nodes))
    return nodes


def snap_ports(
    det: dict, node_map: np.ndarray, cfg: dict, n_terminals: int
) -> tuple[list, dict | None]:
    """Port-template snapping (contribution C3).

    Finds the boundary crossings exactly as :func:`snap_boundary` does,
    then asks the class's port templates which crossing is which named
    pin (see :mod:`schematic2netlist.ports`). This is what makes the
    emitted terminal order MEAN something for directional devices; the
    previous strategies filled terminals in whatever order the boundary
    walk happened to encounter them.

    Returns ``(nodes, info)``. ``info`` is None when the template did
    not fit and ``nodes`` came from the boundary fallback — identity is
    a bonus, never a connectivity regression.
    """
    s = cfg["snapping"]
    step, max_expand = s["expand_step"], s["max_expand"]
    x1, y1, x2, y2 = bbox_xyxy(det)

    sites: list = []
    for r in range(step, max_expand + 1, step):
        found = _boundary_run_sites(node_map, x1 - r, y1 - r, x2 + r, y2 + r)
        if len(found) > len(sites):
            sites = found
        if len(found) >= n_terminals:
            break

    matched = ports_mod.match_ports(canonical_class(det["class"]), det, sites)
    if matched is not None:
        nodes, info = matched
        if len(nodes) == n_terminals:
            return nodes, info

    return snap_boundary(det, node_map, cfg, n_terminals), None


def build_component_pin_nets(
    detections: list[dict], node_map: np.ndarray, cfg: dict
) -> list[dict]:
    """Snap every detected component's terminals to wire nodes.

    Returns one record per component:
    {"id", "class", "nodes": [raw_id|None, raw_id|None], "kind"}.
    Ground symbols get one node stored twice for shape consistency.
    """
    strategy = cfg["snapping"]["strategy"]
    comps = []
    for i, det in enumerate(detections):
        if class_role(det["class"]) == "none":
            # drawing annotations (Wire Crossover) are not electrical
            # components — no terminals to snap
            continue
        if is_ground(det["class"]):
            if strategy in ("boundary", "ports"):
                # A GND symbol has exactly ONE terminal; emitting two
                # copies would invent a terminal-pair that GT does not
                # have and cost precision. It also has no orientation
                # signal in its port data, so "ports" behaves as
                # "boundary" here.
                comps.append({
                    "id": i, "class": det["class"],
                    "nodes": snap_boundary(det, node_map, cfg, 1),
                    "kind": "ground",
                })
                continue
            if strategy == "directional":
                g = find_ground_node(det, node_map, cfg)
            else:
                nodes = snap_uniform(
                    det, node_map, cfg,
                    max_expand=cfg["snapping"]["uniform_ground_max_expand"],
                )
                g = nodes[0] if nodes[0] is not None else nodes[1]
            comps.append(
                {"id": i, "class": det["class"], "nodes": [g, g], "kind": "ground"}
            )
            continue

        port_info = None
        if strategy == "boundary":
            # class-aware terminal count — the whole point: a MOSFET has
            # three pins and must be able to report three nets
            nodes = snap_boundary(det, node_map, cfg, class_terminals(det["class"]))
        elif strategy == "ports":
            nodes, port_info = snap_ports(
                det, node_map, cfg, class_terminals(det["class"])
            )
        elif strategy == "directional":
            nodes = snap_directional(det, node_map, cfg)
        elif strategy == "uniform":
            nodes = snap_uniform(det, node_map, cfg)
        else:
            raise ValueError(f"Unknown snapping.strategy: {strategy!r}")
        rec = {
            "id": i, "class": det["class"], "nodes": nodes, "kind": "two_terminal",
        }
        if port_info is not None:
            # terminal k is now the k-th NAMED port of this class, so the
            # netlist writer can emit correct pin order and polarity
            rec["ports"] = port_info
        comps.append(rec)
    return comps
