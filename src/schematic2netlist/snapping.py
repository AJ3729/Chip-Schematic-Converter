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
        cls = det["class"].lower()
        if "ground" in cls:
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

        if strategy == "directional":
            nodes = snap_directional(det, node_map, cfg)
        elif strategy == "uniform":
            nodes = snap_uniform(det, node_map, cfg)
        else:
            raise ValueError(f"Unknown snapping.strategy: {strategy!r}")
        comps.append(
            {"id": i, "class": det["class"], "nodes": nodes, "kind": "two_terminal"}
        )
    return comps
