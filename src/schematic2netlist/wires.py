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


def _bridge_collinear(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Close gaps ALONG a stroke without welding perpendicular neighbours.

    A square closing kernel large enough to bridge a dash gap is also
    large enough to fuse two wires that merely pass near each other. Two
    anisotropic closings (one horizontal, one vertical) bridge axis-
    aligned gaps — the dominant case in schematics — while leaving the
    perpendicular direction untouched. Their union is the bridged mask.
    """
    span = cfg["wires"].get("bridge_span", 9)
    if span < 2:
        return mask
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, span))
    h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, h_k)
    v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, v_k)
    return cv2.bitwise_or(h, v)


def _filter_blobs(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Drop noise blobs, keeping anything with real area OR real extent.

    A hairline wire segment has small pixel area but large extent; a
    pure area threshold deletes it and shatters the net.
    """
    wcfg = cfg["wires"]
    min_area = wcfg["min_blob_area"]
    min_extent = wcfg.get("min_blob_extent", 15)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if a >= min_area or max(w, h) >= min_extent:
            out[labels == i] = 255
    return out


def extract_wires_ink(
    gray: np.ndarray, non_wire_mask: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Ink-based wire extraction (contribution C2).

    Replaces Canny edge detection with the binarized ink itself. Canny
    converts each pen stroke — which has width — into TWO parallel edge
    lines with a hollow gap between them, which morphology then has to
    glue back together; that is a primary source of net fragmentation.
    The ink is already a solid mark, so thresholding it directly yields
    connected strokes.

    Pipeline: binarize -> remove component/text regions -> bridge gaps
    along-stroke (anisotropic closing) -> extent-aware blob filter.
    """
    wire_candidate = gray.copy()
    wire_candidate[non_wire_mask > 0] = 255

    # ink = dark pixels. Otsu adapts to the (now anti-aliased) cleaned frame.
    ink = cv2.threshold(
        wire_candidate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    ink[non_wire_mask > 0] = 0

    bridged = _bridge_collinear(ink, cfg)
    clean_wires = _filter_blobs(bridged, cfg)
    return wire_candidate, clean_wires


def extract_wires(
    gray: np.ndarray, non_wire_mask: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wire_candidate_image, clean_wire_binary_mask).

    Dispatches on ``wires.method``: "ink" (C2, default) or "canny"
    (the legacy edge-based baseline, kept for the ablation).
    """
    wcfg = cfg["wires"]
    if wcfg.get("method", "canny") == "ink":
        return extract_wires_ink(gray, non_wire_mask, cfg)

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
