"""Heuristic text masking.

Handwritten labels (e.g. "10Ω", "15mH") would otherwise be picked up as
wires. This connected-component heuristic is its own module because its
value is an ablation axis (Phase E3: text masking on/off, filter
sensitivity).

Migrated verbatim from nodes_mapping_and_netlist.py (v1).
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_text_mask(gray_img: np.ndarray, cfg: dict) -> np.ndarray:
    """Binary mask (255 = text) over the grayscale image."""
    t = cfg["textmask"]

    bin_inv = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        t["adaptive_block_size"],
        t["adaptive_c"],
    )
    k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (t["dilate_kernel"], t["dilate_kernel"])
    )
    bin_inv = cv2.dilate(bin_inv, k, iterations=1)

    num, _, stats, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    text_mask = np.zeros_like(gray_img, dtype=np.uint8)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        aspect = ww / hh if hh > 0 else 0
        if (
            t["min_area"] < area < t["max_area"]
            and t["min_aspect"] < aspect < t["max_aspect"]
            and hh < t["max_height"]
            and ww < t["max_width"]
        ):
            text_mask[y : y + hh, x : x + ww] = 255

    return text_mask
