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
    mask = (clean_wires > 0).astype(np.uint8)

    # (1) notch each crossover center to sever the X
    boxes_xyxy = []
    for det in crossover_boxes:
        x1, y1, x2, y2 = bbox_xyxy(det)
        boxes_xyxy.append((x1, y1, x2, y2))
        bw, bh = x2 - x1, y2 - y1
        nx, ny = int(bw * notch_frac / 2), int(bh * notch_frac / 2)
        ccx, ccy = (x1 + x2) // 2, (y1 + y2) // 2
        mask[max(0, ccy - ny):ccy + ny, max(0, ccx - nx):ccx + nx] = 0

    num_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity)
    node_map = labels.astype(np.int32) - 1

    # (2) reconnect opposite arms
    uf = _UnionFind(num_labels)
    for (x1, y1, x2, y2) in boxes_xyxy:
        top = _edge_label(node_map, x1, y1, x2, y2, "top", band)
        bottom = _edge_label(node_map, x1, y1, x2, y2, "bottom", band)
        left = _edge_label(node_map, x1, y1, x2, y2, "left", band)
        right = _edge_label(node_map, x1, y1, x2, y2, "right", band)
        if top is not None and bottom is not None:
            uf.union(top + 1, bottom + 1)      # +1: labels are node_map+1
        if left is not None and right is not None:
            uf.union(left + 1, right + 1)

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
