"""Skeletonization and stroke-intersection detection.

Net assembly needs to know *where* two strokes meet before it can ask
whether they connect. That is a geometric question, answered here, and
it is deliberately separate from the semantic question (junction vs
crossover) answered by :mod:`schematic2netlist.junction_model`.

Two implementation notes, both learned the hard way:

- The morphological erode/open residue "skeleton" is not adequate. It
  leaves a thick, fragmented result on which branch-point counting
  under-reports intersections badly. This uses real Zhang-Suen thinning.
- ``cv2.filter2D`` with a uint8 kernel returns wrong neighbour counts
  (capped around 2), which silently reports zero branch points on a
  skeleton that plainly has them. Neighbour counting goes through
  scipy.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int32)


def thin(mask: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Zhang-Suen thinning to a 1-pixel-wide skeleton (uint8, 0/1)."""
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, dy, 0), dx, 1) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1))]
            P2, P3, P4, P5, P6, P7, P8, P9 = p
            B = sum(p)
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                    for i in range(8))
            if step == 0:
                c1, c2 = P2 * P4 * P6, P4 * P6 * P8
            else:
                c1, c2 = P2 * P4 * P8, P2 * P6 * P8
            kill = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                    & (c1 == 0) & (c2 == 0))
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            break
    return img


def intersection_sites(mask: np.ndarray, min_sep: int = 6) -> list[tuple[int, int]]:
    """(x, y) of each place strokes meet, clustered so one physical
    intersection yields one site."""
    return [(x, y) for x, y, _deg in intersection_sites_with_degree(mask, min_sep)]


def intersection_sites_with_degree(
    mask: np.ndarray, min_sep: int = 6, window: int = 3
) -> list[tuple[int, int, int]]:
    """(x, y, degree) per site, where degree approximates how many wire
    arms meet there.

    Degree matters because the notch-and-relink repair is only *defined*
    for a 4-way crossing: you sever the middle and rejoin opposite arms.
    A 3-arm T has no opposite pair for its stem, so notching it orphans
    that stem permanently. Three wires meeting in a schematic is a
    junction by definition anyway, so a T is never a crossing candidate.
    """
    skel = thin(mask)
    neigh = ndimage.convolve(skel.astype(np.int32), _NEIGHBOURS, mode="constant")
    branch = ((skel > 0) & (neigh >= 3)).astype(np.uint8)
    if not branch.any():
        return []
    grown = cv2.dilate(branch, np.ones((min_sep, min_sep), np.uint8))
    n, _lab, _stats, cents = cv2.connectedComponentsWithStats(grown, 8)

    out = []
    H, W = skel.shape
    for x, y in cents[1:n]:
        x, y = int(round(x)), int(round(y))
        y0, y1 = max(0, y - window), min(H, y + window + 1)
        x0, x1 = max(0, x - window), min(W, x + window + 1)
        win = neigh[y0:y1, x0:x1][skel[y0:y1, x0:x1] > 0]
        out.append((x, y, int(win.max()) if win.size else 0))
    return out


def crop_site(mask: np.ndarray, x: int, y: int, half: int, size: int) -> np.ndarray:
    """Square patch centred on a site, padded at the page edge and
    resized — matching how training patches were produced."""
    H, W = mask.shape[:2]
    X1, Y1, X2, Y2 = x - half, y - half, x + half, y + half
    pad_l, pad_t = max(0, -X1), max(0, -Y1)
    pad_r, pad_b = max(0, X2 - W), max(0, Y2 - H)
    patch = mask[max(0, Y1):min(H, Y2), max(0, X1):min(W, X2)]
    if patch.size == 0:
        return np.zeros((size, size), np.uint8)
    if any((pad_l, pad_t, pad_r, pad_b)):
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
