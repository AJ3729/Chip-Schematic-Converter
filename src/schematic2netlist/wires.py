"""Wire extraction: mask out non-wire regions (components + text), then
edge detection + morphology + area filtering to a clean binary wire mask.

Migrated verbatim from nodes_mapping_and_netlist.py (v1). The v2 variant
differed only in min_blob_area (60 vs 20) and its smaller non-wire class
set — both are config knobs.
"""

from __future__ import annotations

import cv2
import numpy as np

from schematic2netlist.classes import canonical_class


def build_non_wire_mask(
    gray: np.ndarray,
    detections: list[dict],
    cfg: dict,
    text_mask: np.ndarray | None = None,
) -> np.ndarray:
    """255 where pixels must not be treated as wire (components, text)."""
    h, w = gray.shape[:2]
    pad = cfg["wires"]["component_mask_pad"]
    non_wire_classes = {
        canonical_class(c) for c in cfg["wires"]["non_wire_classes"]
    }

    non_wire_mask = np.zeros((h, w), dtype=np.uint8)
    if text_mask is not None:
        non_wire_mask = cv2.bitwise_or(non_wire_mask, text_mask)

    for det in detections:
        if canonical_class(det["class"]) not in non_wire_classes:
            continue
        cx, cy = det["x"], det["y"]
        bw, bh = det["width"], det["height"]
        x1 = max(0, int(cx - bw / 2) - pad)
        y1 = max(0, int(cy - bh / 2) - pad)
        x2 = min(w, int(cx + bw / 2) + pad)
        y2 = min(h, int(cy + bh / 2) + pad)
        non_wire_mask[y1:y2, x1:x2] = 255

    return non_wire_mask


def extract_wires(
    gray: np.ndarray, non_wire_mask: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wire_candidate_image, clean_wire_binary_mask)."""
    wcfg = cfg["wires"]

    wire_candidate = gray.copy()
    wire_candidate[non_wire_mask > 0] = 255

    edges = cv2.Canny(wire_candidate, wcfg["canny_low"], wcfg["canny_high"])

    mk = wcfg["morph_kernel"]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    wires_img = cv2.dilate(edges, k, iterations=1)
    wires_img = cv2.morphologyEx(wires_img, cv2.MORPH_OPEN, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        wires_img, connectivity=8
    )
    clean_wires = np.zeros_like(wires_img)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= wcfg["min_blob_area"]:
            clean_wires[labels == i] = 255

    return wire_candidate, clean_wires
