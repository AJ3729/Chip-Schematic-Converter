"""Image preprocessing: deskew, shadow normalization, binarization, crop,
aspect-preserving resize onto a square canvas.

Migrated from scripts/preprocess.py (legacy). All thresholds come from the
`preprocess` section of the config.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(path: str, cfg: dict) -> np.ndarray | None:
    """Preprocess one image file; returns the binarized canvas or None."""
    p = cfg["preprocess"]

    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = p["blur_kernel"]
    gray_blur = cv2.GaussianBlur(gray, (k, k), 0)

    # Foreground mask (ink = 255) for skew-angle estimation
    bin_inv = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    bin_inv_er = cv2.erode(bin_inv, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_inv_er, connectivity=8
    )
    angle = 0.0
    if num > 1:
        H, W = bin_inv_er.shape
        margin = p["border_margin"]
        candidates = []
        for i in range(1, num):
            x, y, w, h, a = stats[i]
            if a < p["angle_min_blob_area"]:
                continue
            if x <= margin - 1 or y <= margin - 1 or x + w >= W - margin or y + h >= H - margin:
                # touches frame -> likely page-border noise
                continue
            candidates.append((a, i))
        if candidates:
            _, idx = max(candidates, key=lambda t: t[0])
            mask = (labels == idx).astype(np.uint8) * 255
            coords = np.column_stack(np.where(mask > 0))
            raw_angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + raw_angle) if raw_angle < -45 else -raw_angle

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )

    if img.shape[0] > img.shape[1] * p["landscape_ratio"]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shadow normalization only when lighting is uneven
    if gray.std() > p["shadow_std_threshold"]:
        dk = p["shadow_dilate_kernel"]
        dilated = cv2.dilate(gray, np.ones((dk, dk), np.uint8))
        bg = cv2.medianBlur(dilated, p["shadow_median_blur"])
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    else:
        norm = gray

    binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Speck removal on the inverted (ink) image
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - binary, connectivity=8
    )
    keep = np.zeros_like(binary)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < p["speck_min_area"]:
            continue
        if w <= 2 and h <= 2:
            continue
        keep[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i] = 255
    binary = 255 - keep

    # Tight crop to content
    fg = (binary == 0).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    coords = cv2.findNonZero(fg)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = p["crop_pad"]
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        cropped = binary[y : y + h, x : x + w]
    else:
        cropped = binary

    # Aspect-preserving resize onto a white square canvas
    target = p["target_size"]
    ch, cw = cropped.shape[:2]
    scale = min(target / ch, target / cw)
    nh, nw = int(round(ch * scale)), int(round(cw * scale))
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((target, target), 255, dtype=np.uint8)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas
