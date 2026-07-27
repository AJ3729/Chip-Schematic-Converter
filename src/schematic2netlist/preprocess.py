"""Image preprocessing: deskew, shadow normalization, binarization, crop,
aspect-preserving resize onto a square canvas.

``preprocess_image_meta`` also returns the full geometric transform
(rotation matrix, optional 90° rotation, crop, scale, canvas offset) so
points/boxes annotated on the ORIGINAL image can be projected into
cleaned-image coordinates (``project_point`` / ``project_bbox``) —
required because the published Digitize-HCD annotations are in
original-image coordinates.

Design notes (v2 — fixes the "annotations cropped out / ink destroyed"
class of bugs):

1. **The crop rectangle is computed BEFORE speck removal.** Previously
   faint pencil components were deleted as specks and then fell outside
   the content bounding box, so their annotations projected off-canvas.
   Speck removal now only affects the emitted pixels, never the crop.
2. **The crop is annotation-aware.** Pass ``ann_boxes`` (original
   coordinates) and every annotated component is unioned into the crop
   rectangle, making it structurally impossible for a labeled component
   to land off-canvas.
3. **The pad scales with image size** (``crop_pad_frac``), instead of a
   fixed 12 px that is only 0.6% of a ~2000 px photo.
4. **Speck removal is length-aware and gentler** — a thin wire segment
   has small area but large extent, so blobs are kept when either their
   area or their extent is meaningful.
5. **Skew estimation is robust.** A length-weighted median over
   ``HoughLinesP`` segments (folded mod 90°) replaces "take the single
   largest interior blob's minAreaRect", which a page ruling or a big
   text label could capture.
6. **Downscaling uses INTER_AREA** (area-averaging) rather than
   INTER_NEAREST, which produced jagged, aliased strokes.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(path: str, cfg: dict, ann_boxes=None) -> np.ndarray | None:
    """Preprocess one image file; returns the canvas or None."""
    result = preprocess_image_meta(path, cfg, ann_boxes=ann_boxes)
    return None if result is None else result[0]


def _rotate_point(meta: dict, x: float, y: float) -> tuple[float, float]:
    """Apply the rotation half of the transform only (rotation + rot90).

    Factored out so the crop step can project annotation boxes into
    post-rotation coordinates before the crop rectangle exists.
    """
    m = meta["rotation_matrix"]
    xr = m[0][0] * x + m[0][1] * y + m[0][2]
    yr = m[1][0] * x + m[1][1] * y + m[1][2]
    if meta["rotated90"]:
        w_before = meta["size_before_rot90"][0]
        xr, yr = yr, (w_before - 1) - xr
    return xr, yr


def project_point(meta: dict, x: float, y: float) -> tuple[float, float]:
    """Project a point from original-image to cleaned-image coordinates."""
    xr, yr = _rotate_point(meta, x, y)
    cx, cy = meta["crop"][0], meta["crop"][1]
    s = meta["scale"]
    ox, oy = meta["canvas_offset"]
    return (xr - cx) * s + ox, (yr - cy) * s + oy


def unproject_point(meta: dict, px: float, py: float) -> tuple[float, float]:
    """Inverse of :func:`project_point`: cleaned-image -> original coords.

    Used to migrate artifacts (e.g. GT bounding boxes) between two
    generations of preprocessing: unproject through the OLD transform,
    then project through the NEW one.
    """
    import numpy as _np

    cx, cy = meta["crop"][0], meta["crop"][1]
    s = meta["scale"]
    ox, oy = meta["canvas_offset"]
    xr = (px - ox) / s + cx
    yr = (py - oy) / s + cy

    if meta["rotated90"]:
        # forward was: (xr, yr) -> (yr, (w_before - 1) - xr)
        w_before = meta["size_before_rot90"][0]
        xr, yr = (w_before - 1) - yr, xr

    m = _np.asarray(meta["rotation_matrix"], dtype=_np.float64)
    inv = _np.zeros((2, 3), dtype=_np.float64)
    a = m[:, :2]
    a_inv = _np.linalg.inv(a)
    inv[:, :2] = a_inv
    inv[:, 2] = -a_inv @ m[:, 2]
    x = inv[0, 0] * xr + inv[0, 1] * yr + inv[0, 2]
    y = inv[1, 0] * xr + inv[1, 1] * yr + inv[1, 2]
    return float(x), float(y)


def project_bbox(
    meta: dict, x: float, y: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Project a COCO-style [x, y, w, h] box (original coords) to a
    center-based (cx, cy, w, h) box in cleaned-image coordinates, by
    projecting all four corners (correct under rotation)."""
    corners = [
        project_point(meta, x, y),
        project_point(meta, x + w, y),
        project_point(meta, x, y + h),
        project_point(meta, x + w, y + h),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def _estimate_skew(ink: np.ndarray, p: dict) -> float:
    """Length-weighted median angle of long line segments, folded mod 90°.

    Schematic wires are predominantly axis-aligned, so every segment
    votes for the same page rotation once folded into [-45, 45).
    Falls back to pooled-blob minAreaRect, then to 0.
    """
    H, W = ink.shape[:2]
    min_len = max(20, int(p.get("hough_min_line_frac", 0.06) * max(H, W)))
    lines = cv2.HoughLinesP(
        ink, 1, np.pi / 180,
        threshold=p.get("hough_threshold", 80),
        minLineLength=min_len,
        maxLineGap=p.get("hough_max_gap", 6),
    )

    if lines is not None and len(lines):
        angles, weights = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = float(np.hypot(dx, dy))
            ang = np.degrees(np.arctan2(dy, dx))
            ang = ((ang + 45.0) % 90.0) - 45.0   # fold mod 90 -> [-45, 45)
            angles.append(ang)
            weights.append(length)
        angles = np.asarray(angles)
        weights = np.asarray(weights)
        order = np.argsort(angles)
        angles, weights = angles[order], weights[order]
        cum = np.cumsum(weights)
        median = float(angles[np.searchsorted(cum, cum[-1] / 2.0)])
        if abs(median) <= p.get("max_skew_deg", 20.0):
            return median

    # fallback: pool every sizeable interior blob (not just the largest)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    margin = p.get("border_margin", 2)
    pooled = np.zeros_like(ink)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < p.get("angle_min_blob_area", 200):
            continue
        if x <= margin - 1 or y <= margin - 1 or x + w >= W - margin or y + h >= H - margin:
            continue
        pooled[labels == i] = 255
    coords = cv2.findNonZero(pooled)
    if coords is None:
        return 0.0
    raw = cv2.minAreaRect(coords)[-1]
    ang = -(90 + raw) if raw < -45 else -raw
    return float(ang) if abs(ang) <= p.get("max_skew_deg", 20.0) else 0.0


def _remove_specks(binary: np.ndarray, p: dict) -> np.ndarray:
    """Drop isolated noise blobs, but KEEP thin/elongated strokes.

    A hairline wire segment has small pixel area yet large extent, so a
    pure area threshold deletes real ink. A blob survives if its area is
    large enough OR it spans a meaningful distance.
    """
    min_area = p.get("speck_min_area", 40)
    min_extent = p.get("speck_min_extent", 12)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - binary, connectivity=8
    )
    keep = np.zeros_like(binary)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a >= min_area or max(w, h) >= min_extent:
            keep[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i] = 255
    return 255 - keep


def preprocess_image_meta(
    path: str, cfg: dict, ann_boxes=None
) -> tuple[np.ndarray, dict] | None:
    """Preprocess one image; returns (canvas, transform_meta) or None.

    ``ann_boxes`` is an optional iterable of COCO ``[x, y, w, h]`` boxes
    in ORIGINAL image coordinates. When given, every box is unioned into
    the crop rectangle so no annotated component can be cropped away.
    """
    p = cfg["preprocess"]

    img = cv2.imread(path)
    if img is None:
        return None

    H0, W0 = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = p["blur_kernel"]
    gray_blur = cv2.GaussianBlur(gray, (k, k), 0)

    # --- skew estimation (robust: Hough modal angle) ---
    ink = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    angle = _estimate_skew(ink, p)

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )

    rotated90 = False
    size_before_rot90 = [img.shape[1], img.shape[0]]
    if img.shape[0] > img.shape[1] * p["landscape_ratio"]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated90 = True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- shadow normalization only when lighting is uneven ---
    if gray.std() > p["shadow_std_threshold"]:
        dk = p["shadow_dilate_kernel"]
        dilated = cv2.dilate(gray, np.ones((dk, dk), np.uint8))
        bg = cv2.medianBlur(dilated, p["shadow_median_blur"])
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    else:
        norm = gray

    binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # ------------------------------------------------------------------
    # CROP RECT — computed from the FULL ink (pre-speck-removal) and
    # unioned with every annotation box. Order matters: computing this
    # after speck removal is what previously pushed faint components
    # off-canvas.
    # ------------------------------------------------------------------
    IH, IW = binary.shape[:2]
    fg = (binary == 0).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                          iterations=1)
    coords = cv2.findNonZero(fg)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
    else:
        x1, y1, x2, y2 = 0, 0, IW, IH

    partial_meta = {
        "rotation_matrix": M.tolist(),
        "rotated90": rotated90,
        "size_before_rot90": size_before_rot90,
    }
    if ann_boxes:
        for (ax, ay, aw, ah) in ann_boxes:
            for px, py in (
                _rotate_point(partial_meta, ax, ay),
                _rotate_point(partial_meta, ax + aw, ay),
                _rotate_point(partial_meta, ax, ay + ah),
                _rotate_point(partial_meta, ax + aw, ay + ah),
            ):
                x1 = min(x1, px)
                y1 = min(y1, py)
                x2 = max(x2, px)
                y2 = max(y2, py)

    # pad scales with image size (a fixed 12 px shaves ~2000 px photos)
    pad = max(p.get("crop_pad", 12),
              int(round(p.get("crop_pad_frac", 0.02) * max(IH, IW))))
    x = int(max(0, np.floor(x1) - pad))
    y = int(max(0, np.floor(y1) - pad))
    w = int(min(IW - x, np.ceil(x2 - x1) + 2 * pad))
    h = int(min(IH - y, np.ceil(y2 - y1) + 2 * pad))

    # --- speck removal affects the OUTPUT pixels only, never the crop ---
    cleaned = _remove_specks(binary, p) if p.get("remove_specks", True) else binary
    cropped = cleaned[y : y + h, x : x + w]

    # --- aspect-preserving resize onto a white square canvas ---
    target = p["target_size"]
    ch, cw = cropped.shape[:2]
    scale = min(target / ch, target / cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    # INTER_AREA anti-aliases on downscale (INTER_NEAREST aliased badly);
    # INTER_CUBIC when upscaling a small crop.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(cropped, (nw, nh), interpolation=interp)
    canvas = np.full((target, target), 255, dtype=np.uint8)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    meta = {
        "original_size": [W0, H0],
        "angle_deg": float(angle),
        "rotation_matrix": M.tolist(),
        "rotated90": rotated90,
        "size_before_rot90": size_before_rot90,
        "crop": [int(x), int(y), int(w), int(h)],
        "scale": float(scale),
        "canvas_offset": [int(x0), int(y0)],
        "target_size": int(target),
        "annotation_aware_crop": bool(ann_boxes),
    }
    return canvas, meta
