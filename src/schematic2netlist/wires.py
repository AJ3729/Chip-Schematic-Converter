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


def stitchable_mask(
    shape,
    detections: list[dict],
    cfg: dict,
    text_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Regions where OUR OWN masking deleted ink, and where reconnecting
    a wire across the hole is therefore safe (tier-1 fix):

    - text-mask rectangles (a wire running under a "10k" label is one
      continuous conductor), and
    - the pad RING around each component box (a rail clipping a
      component's padding), but NEVER the un-padded component body —
      a component's two leads are different nets THROUGH the component,
      and stitching across the body would weld them.
    """
    h, w = shape[:2]
    pad = cfg["wires"]["component_mask_pad"]
    non_wire_classes = {
        canonical_class(c) for c in cfg["wires"]["non_wire_classes"]
    }

    mask = np.zeros((h, w), dtype=np.uint8)
    if text_mask is not None:
        mask = cv2.bitwise_or(mask, text_mask)

    body = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        if canonical_class(det["class"]) not in non_wire_classes:
            continue
        cx, cy = det["x"], det["y"]
        bw, bh = det["width"], det["height"]
        # padded box -> stitchable candidate
        x1 = max(0, int(cx - bw / 2) - pad)
        y1 = max(0, int(cy - bh / 2) - pad)
        x2 = min(w, int(cx + bw / 2) + pad)
        y2 = min(h, int(cy + bh / 2) + pad)
        mask[y1:y2, x1:x2] = 255
        # un-padded body -> never stitchable
        bx1 = max(0, int(cx - bw / 2))
        by1 = max(0, int(cy - bh / 2))
        bx2 = min(w, int(cx + bw / 2))
        by2 = min(h, int(cy + bh / 2))
        body[by1:by2, bx1:bx2] = 255

    mask[body > 0] = 0
    return mask


def _local_direction(pts: np.ndarray) -> np.ndarray | None:
    """Principal direction (unit vector) of a small pixel cloud."""
    if len(pts) < 4:
        return None
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, -1]
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def stitch_wire_islands(
    clean_wires: np.ndarray,
    stitchable: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Reconnect wire islands separated by regions we ourselves masked.

    The measured dominant failure mode is rail nets shattered into 4-6
    islands with 20-45 px gaps located exactly where component padding
    or text rectangles deleted the ink. Those holes are self-inflicted
    and their locations are known, so bridging them is principled:

    for a pair of islands, take the closest pair of "frontier" points
    (island pixels adjacent to a stitchable region) and connect them iff
    (a) the gap is small enough, (b) the straight segment runs almost
    entirely through stitchable-or-wire pixels (the hole explains the
    gap), (c) the segment does not run through a third island (no
    accidental three-way welds), and (d) the segment direction agrees
    with the local stroke direction on BOTH sides (a pin lead pointing
    into a component is perpendicular to the ring and is refused —
    collinearity is the safety property that keeps distinct nets apart).

    Bridges are drawn as thin lines so downstream connected-components
    merge the islands naturally. Two passes handle chained holes.
    """
    wcfg = cfg["wires"]
    max_gap = wcfg.get("stitch_max_gap", 60)
    angle_tol = np.cos(np.radians(wcfg.get("stitch_angle_tol_deg", 35.0)))
    min_inside = wcfg.get("stitch_min_inside_frac", 0.75)
    dir_radius = wcfg.get("stitch_dir_radius", 7)

    out = clean_wires.copy()
    k3 = np.ones((3, 3), np.uint8)

    for _ in range(wcfg.get("stitch_passes", 2)):
        num, labels = cv2.connectedComponents((out > 0).astype(np.uint8), connectivity=8)
        if num <= 2:
            break
        near_stitch = cv2.dilate(stitchable, k3) > 0

        frontiers: dict[int, np.ndarray] = {}
        for lab in range(1, num):
            island = labels == lab
            f = np.argwhere(island & near_stitch)          # (y, x)
            if len(f):
                if len(f) > 300:
                    f = f[:: len(f) // 300]
                frontiers[lab] = f

        drew = False
        labs = sorted(frontiers)
        for ai in range(len(labs)):
            for bi in range(ai + 1, len(labs)):
                a, b = labs[ai], labs[bi]
                fa, fb = frontiers[a], frontiers[b]
                d2 = ((fa[:, None, :] - fb[None, :, :]) ** 2).sum(-1)
                ia, ib = np.unravel_index(d2.argmin(), d2.shape)
                dist = float(np.sqrt(d2[ia, ib]))
                if dist > max_gap or dist < 1:
                    continue
                pa, pb = fa[ia], fb[ib]                     # (y, x)

                # (b) segment must be explained by masked holes / wire
                n_samp = max(int(dist), 2)
                ts = np.linspace(0.0, 1.0, n_samp)
                ys = np.clip((pa[0] + ts * (pb[0] - pa[0])).round().astype(int), 0, out.shape[0] - 1)
                xs = np.clip((pa[1] + ts * (pb[1] - pa[1])).round().astype(int), 0, out.shape[1] - 1)
                inside = (stitchable[ys, xs] > 0) | (out[ys, xs] > 0)
                if inside.mean() < min_inside:
                    continue

                # (c) no third-island welds
                seg_labels = labels[ys, xs]
                if np.any((seg_labels != 0) & (seg_labels != a) & (seg_labels != b)):
                    continue

                # (d) collinearity on both sides
                seg = (pb - pa).astype(float)
                seg /= np.linalg.norm(seg)
                ok = True
                for lab, p in ((a, pa), (b, pb)):
                    island_pts = np.argwhere(labels == lab)
                    win = island_pts[
                        (np.abs(island_pts - p) <= dir_radius).all(axis=1)
                    ]
                    v = _local_direction(win.astype(float))
                    if v is not None and abs(float(v @ seg)) < angle_tol:
                        ok = False
                        break
                if not ok:
                    continue

                cv2.line(out, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), 255, 2)
                drew = True
        if not drew:
            break
    return out


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
