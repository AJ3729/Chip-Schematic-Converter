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
