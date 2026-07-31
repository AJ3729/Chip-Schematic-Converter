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


_PERP = {"h": "v", "v": "h", "d1": "d2", "d2": "d1"}


def _line_kernel(length: int, orient: str, force_odd: bool = True) -> np.ndarray:
    """A 1-px-wide line structuring element at 0, 90, 45 or 135 degrees.

    MIND THE PARITY. A closing is extensive -- it can only add ink -- when the
    structuring element is symmetric about its anchor. OpenCV anchors an
    even-length kernel at index length/2, which is off-centre, so dilate and
    erode are no longer adjoint and the closing DELETES ink: a lone pixel is
    annihilated outright. Measured on 25 frames at 1024 px, a closing at the
    shipped span of 18 destroys 1.44% of all wire ink, and every even span
    tested destroys between 0.87% and 2.40% while every odd span destroys
    exactly none. Deleting wire ink severs nets, so this shows up as splits.

    The length is therefore rounded up to odd. That makes an even ``span`` in
    an old config behave as span+1, which is the intended semantics of "bridge
    gaps up to this wide" anyway, and it removes a whole class of silent
    damage rather than leaving it to whoever next picks a round number.

    ``force_odd=False`` reproduces the damaging behaviour on purpose. The
    ablation needs a pre-fix arm to attribute the fix against, and a config flag
    is the only honest way to keep one once the bug is fixed in code.
    """
    length = int(length) | 1 if force_odd else int(length)
    if orient == "h":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    if orient == "v":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    k = np.eye(length, dtype=np.uint8)
    return k if orient == "d1" else np.ascontiguousarray(k[:, ::-1])


def _oriented_ink(mask: np.ndarray, orient: str, run: int,
                  thick: int) -> np.ndarray:
    """Ink that genuinely runs along ``orient``.

    Dilating PERPENDICULAR to the direction first is what makes this usable on
    hand-drawn strokes: a 1-px-wide test kernel asks for ``run`` ink pixels in
    a single row, which a pen line drifting a pixel every few px does not have.
    Thickening a vertical stroke vertically leaves it just as narrow
    horizontally, so the asymmetry that matters is preserved. The dilation is
    undone so the result is a subset of the original ink.
    """
    pk = _line_kernel(thick, _PERP[orient])
    fat = cv2.dilate(mask, pk) if thick > 1 else mask
    seed = cv2.morphologyEx(fat, cv2.MORPH_OPEN, _line_kernel(run, orient))
    return cv2.erode(seed, pk) if thick > 1 else seed


def _bridge_guarded(mask: np.ndarray, span: int, run: int, thick: int,
                    diagonal: bool = True) -> np.ndarray:
    """The ungated closing, minus the pixels that weld two parallel strokes.

    Replacing the closing with an orientation-gated one removes the welds but
    also removes bridging the closing was rightly doing: measured on real
    frames, the legacy closing adds 55% more ink than the frame contains, the
    gated version adds 7.6%, and the split rate rose 0.12 as a result. The
    closing is doing two jobs -- bridging dash gaps AND gluing wobbly strokes
    and pen lifts -- and gating throws the second away with the first.

    So keep every pixel the closing adds and subtract only those carrying the
    weld signature. The test is stated as a condition for KEEPING a fill, not
    for rejecting one, and that phrasing is what makes it correct in general: a
    fill along direction o is legitimate when ink that itself runs along o lies
    within reach on at least one side, because that is a stroke the fill is
    continuing. Anything else -- two rails flanking the gap, or two strokes
    that merely pass nearby -- is a weld.

    Rejecting on "PERPENDICULAR ink on both sides" was tried first and is too
    narrow twice over. It misses a weld between strokes that are neither
    parallel nor perpendicular to the pass, which is exactly how adding the
    diagonal passes re-welded the parallel rails they were supposed to leave
    alone: a vertical rail is neither diagonal nor anti-diagonal, so the guard
    could not see it. And requiring both sides was solving a problem the
    keep-phrasing does not have -- a horizontal wire meeting a vertical rail
    keeps its fill because the horizontal wire is aligned ink on one side,
    with no special case needed.

    Because the result starts from the original ink and only ever adds, this
    mode is extensive by construction -- which the bare closing is not, see
    ``_line_kernel``.
    """
    out = mask.copy()
    inv_mask = cv2.bitwise_not(mask)
    orients = ("h", "v", "d1", "d2") if diagonal else ("h", "v")
    for o in orients:
        k = _line_kernel(span, o)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        fill = cv2.bitwise_and(closed, inv_mask)
        if not fill.any():
            out = cv2.bitwise_or(out, closed)
            continue
        along = _oriented_ink(mask, o, run, thick)
        # anchored dilations reach one way along o, then the other, so a fill
        # pixel set in either has aligned ink on that side
        far = (k.shape[1] - 1, k.shape[0] - 1)
        legit = cv2.bitwise_or(cv2.dilate(along, k, anchor=(0, 0)),
                               cv2.dilate(along, k, anchor=far))
        keep = cv2.bitwise_or(cv2.bitwise_and(fill, legit), mask)
        out = cv2.bitwise_or(out, cv2.bitwise_and(closed, keep))
    return out


def _bridge_collinear(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Close gaps ALONG a stroke without welding parallel neighbours.

    A square closing kernel large enough to bridge a dash gap is also
    large enough to fuse two wires that merely pass near each other, so
    the bridging is done with 1-px-wide line kernels instead.

    That alone is not sufficient, and the reason is easy to get backwards:
    a HORIZONTAL closing bridges horizontal gaps between *any* ink, and
    the horizontal gap between two side-by-side VERTICAL strokes is
    exactly such a gap. Two rails 10 px apart become one solid band at
    span 18 — measured, not hypothesised — which is why lowering the span
    improved both the weld and the split rate at once: it moved the fusion
    threshold below the common rail spacing without addressing the cause.

    ``bridge_mode: directional`` addresses the cause. Each orientation is
    first OPENED with a short line kernel of its own direction, which
    keeps only ink that genuinely runs that way, and only that ink is
    allowed to bridge. A vertical stroke has no horizontal run longer
    than its own width, so it never seeds the horizontal closing and can
    no longer be welded to its neighbour. Gating by orientation also lets
    the diagonal passes be added safely, which an ungated closing could
    not afford.

    Nothing is ever removed: the result is the original ink unioned with
    the gated bridges, so a stroke too short to seed any orientation is
    left exactly as it was rather than deleted.

    ``bridge_mode: closing`` keeps the ungated behaviour for the ablation.
    """
    wcfg = cfg["wires"]
    span = wcfg.get("bridge_span", 9)
    if span < 2:
        return mask

    mode = wcfg.get("bridge_mode", "closing")
    odd = bool(wcfg.get("bridge_odd_kernel", True))
    if mode == "guarded":
        return _bridge_guarded(mask, span, int(wcfg.get("bridge_run", 7)),
                               int(wcfg.get("bridge_thick", 3)),
                               bool(wcfg.get("bridge_diagonal", True)))
    if mode != "directional":
        h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             _line_kernel(span, "h", odd))
        v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             _line_kernel(span, "v", odd))
        return cv2.bitwise_or(h, v)

    run = int(wcfg.get("bridge_run", 5))
    thick = int(wcfg.get("bridge_thick", 3))
    orients = ["h", "v"]
    if wcfg.get("bridge_diagonal", True):
        orients += ["d1", "d2"]

    # A closing extrapolates at the frame border -- erode takes a maximum
    # there, so ink bleeds outward and leaves specks the source image never
    # had. Padding with background and cropping back makes the border inert.
    # Only the directional branch pads, so ``closing`` mode stays bit-exact
    # with every benchmark already recorded against it.
    p = span + 1
    padded = cv2.copyMakeBorder(mask, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    out = padded.copy()
    for o in orients:
        # Testing orientation with a 1-px-wide kernel asks for `run` ink
        # pixels in a single row, which a hand-drawn stroke that drifts one
        # pixel every few px does not have -- gating on that alone withheld
        # bridges the ungated closing was rightly making, and measured worse
        # on both axes. Dilating PERPENDICULAR to the pass first restores the
        # tolerance without giving up the asymmetry that matters: thickening
        # a vertical stroke vertically leaves it just as narrow horizontally,
        # so it still cannot seed the horizontal pass. The dilation is undone
        # afterwards so bridging never fattens the strokes it joins.
        pk = _line_kernel(thick, _PERP[o])
        fat = cv2.dilate(padded, pk) if thick > 1 else padded
        seed = cv2.morphologyEx(fat, cv2.MORPH_OPEN, _line_kernel(run, o))
        if not seed.any():
            continue
        bridged = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, _line_kernel(span, o))
        if thick > 1:
            bridged = cv2.erode(bridged, pk)
        out = cv2.bitwise_or(out, bridged)
    return np.ascontiguousarray(out[p:-p, p:-p])


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


def _binarize_ink(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Ink mask from a grayscale frame.

    ``otsu`` picks ONE threshold for the whole page. On a photographed
    hand-drawing that is the wrong model: pen pressure and lighting vary across
    the sheet, so a single cut leaves thin strokes below it (gaps, which become
    net SPLITS) while letting near-touching strokes bleed together (WELDS). The
    measured failure profile matches that exactly -- 16.9% of GT nets are
    simultaneously welded AND split, which is the one signature a global
    threshold produces and a local one should not.

    ``adaptive`` thresholds against a Gaussian-weighted local mean.
    ``sauvola`` additionally scales the threshold by the local standard
    deviation, t = m * (1 + k * (s / r - 1)), which is the standard choice for
    document images because it does not carve up blank regions the way a plain
    local mean does -- where there is no ink there is no variance, so the
    threshold stays high. Implemented on integral images so it stays O(N)
    regardless of window size and needs no extra dependency.
    """
    mode = cfg["wires"].get("binarize", "otsu")
    if mode == "otsu":
        return cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    win = int(cfg["wires"].get("binarize_window", 31)) | 1
    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            win, float(cfg["wires"].get("binarize_c", 10)))

    if mode == "sauvola":
        k = float(cfg["wires"].get("binarize_k", 0.2))
        r = float(cfg["wires"].get("binarize_r", 128.0))
        g = gray.astype(np.float64)
        mean = cv2.boxFilter(g, -1, (win, win), normalize=True,
                             borderType=cv2.BORDER_REPLICATE)
        sq = cv2.boxFilter(g * g, -1, (win, win), normalize=True,
                           borderType=cv2.BORDER_REPLICATE)
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        thr = mean * (1.0 + k * (std / r - 1.0))
        return ((g < thr).astype(np.uint8)) * 255

    raise ValueError(f"Unknown wires.binarize: {mode!r}")


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

    # ink = dark pixels. See _binarize_ink: the choice of global vs local
    # thresholding is exactly the choice between accepting and rejecting the
    # welded-AND-split failure mode.
    ink = _binarize_ink(wire_candidate, cfg)
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
