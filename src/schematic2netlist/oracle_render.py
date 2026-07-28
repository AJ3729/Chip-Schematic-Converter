"""Ground-truth wire rendering for the oracle's mode C (C4).

Mode C asks: *if connectivity geometry were perfect, how much would
terminal snapping still lose?* The answer is only meaningful if the
synthetic wiring is at least as readable as a real drawing. The first
implementation rendered a **star** — every terminal on a net joined to
the net's centroid by a straight line — and failed measurably: those
spokes approach components from arbitrary angles, cross bounding boxes
where no terminal is, and run through unrelated component bodies.
Boundary snapping read the synthetic wires *worse* than real ones,
yielding a negative wire-error attribution (an impossible result), so
mode C was flagged invalid.

Two changes fix it.

**Orthogonal routing.** Wires are routed on a coarse grid with
4-connected Lee maze search, so every run is axis-aligned like drawn
wire, and every route avoids *foreign component bodies* — a wire that
grazes an unrelated symbol would invent a terminal there. Each pin gets
a short outward stub so its conductor crosses its own bounding box
exactly at the pin.

**A node-label map, not a binary mask.** The pipeline's net assembly
turns a binary mask into labels; mode C supplies the labels directly.
This matters because crossings are legal in schematics: two nets that
cross without connecting is the situation the crossover machinery
exists to handle, and forcing the synthetic wiring to be planar (nets
routed around each other) over-constrains a dense canvas and fails to
route at all. Painting labels lets nets cross freely — the ideal that
perfect crossing resolution would achieve — while each net keeps one
identity.

The render is **verified, not trusted**: `render_gt_node_map` reports
whether every pin carries its own net's label and whether any foreign
net intrudes into a component's box. Callers must exclude images whose
report is not ``ok`` rather than averaging an invalid render into the
attribution.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .classes import class_terminals

STUB = 9          # outward stub length (px) from the bbox edge
THICKNESS = 3     # rendered conductor thickness (px)
STRIDE = 3        # routing-grid cell size (px)
BODY_PAD = 5      # px of clearance foreign routes keep from a component


def terminal_sites(comp: dict) -> list[tuple[float, float, tuple[int, int]]]:
    """Terminal (x, y, outward-normal) sites on a component's bbox edge.

    Two-terminal parts get pins on the ends of the long axis; a
    one-terminal part (GND) gets one on the near edge; parts with three
    or more terminals put the first two on the long-axis ends and spread
    the rest along the perpendicular face. Exact pin identity is the
    subject of M3 and deliberately not modeled here — the oracle only
    needs sites a boundary reader can find.
    """
    cx, cy, w, h = comp["bbox"]
    n_t = max(1, class_terminals(comp["class"]))
    horiz = w >= h
    if n_t == 1:
        return [(cx, cy + h / 2, (0, 1))]

    sites = []
    for i in range(n_t):
        if i < 2:
            if horiz:
                sites.append(
                    (cx - w / 2, cy, (-1, 0)) if i == 0 else (cx + w / 2, cy, (1, 0))
                )
            else:
                sites.append(
                    (cx, cy - h / 2, (0, -1)) if i == 0 else (cx, cy + h / 2, (0, 1))
                )
        else:
            span = (i - 1) / (n_t - 1)
            if horiz:
                sites.append((cx - w / 2 + w * span, cy + h / 2, (0, 1)))
            else:
                sites.append((cx + w / 2, cy - h / 2 + h * span, (1, 0)))
    return sites


def _box_xyxy(comp: dict, pad: float = 0.0) -> tuple[float, float, float, float]:
    cx, cy, w, h = comp["bbox"]
    return (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)


def _cell(pt, stride: int, grid_shape) -> tuple[int, int]:
    rows, cols = grid_shape
    return (
        int(min(max(pt[1] // stride, 0), rows - 1)),
        int(min(max(pt[0] // stride, 0), cols - 1)),
    )


def _blocked_for(exclude: set[int], comps, shape, stride, pad) -> np.ndarray:
    """Grid cells covered by component bodies, excluding the components
    this net legitimately terminates on."""
    h_img, w_img = shape[:2]
    rows, cols = h_img // stride + 1, w_img // stride + 1
    blocked = np.zeros((rows, cols), dtype=bool)
    for ci, c in enumerate(comps):
        if ci in exclude:
            continue
        x1, y1, x2, y2 = _box_xyxy(c, pad)
        r0 = max(0, int(y1 // stride))
        r1 = min(rows - 1, int(y2 // stride))
        c0 = max(0, int(x1 // stride))
        c1 = min(cols - 1, int(x2 // stride))
        blocked[r0:r1 + 1, c0:c1 + 1] = True
    return blocked


def _route(start, targets: set, blocked: np.ndarray):
    """Lee maze route from ``start`` to the nearest cell of ``targets``.

    Breadth-first over 4-neighbours, so paths are axis-aligned runs like
    drawn wire. Returns the inclusive cell path, or None if unreachable.
    """
    rows, cols = blocked.shape
    if start in targets:
        return [start]
    prev: dict = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or nxt in prev:
                continue
            prev[nxt] = (r, c)
            if nxt in targets:
                path = [nxt]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            if blocked[nxt]:
                continue
            q.append(nxt)
    return None


def render_gt_node_map(
    gt: dict, shape, thickness: int = THICKNESS, stub: int = STUB,
    stride: int = STRIDE, body_pad: int = BODY_PAD,
) -> tuple[np.ndarray, dict, dict]:
    """Render GT connectivity as a node-label map; verify the result.

    Returns ``(node_map, label_of_net, report)`` where ``node_map`` is an
    int32 label image directly consumable by
    :func:`schematic2netlist.snapping.build_component_pin_nets`.

    **Background is -1, not 0** — this matches
    :func:`schematic2netlist.nodes.build_wire_nodes`, whose labels the
    snapping stage was written against. Emitting 0-background here made
    boundary snapping read the entire page background as one enormous
    node touching every component, which is what made the synthetic
    render score far worse than real wires.

    ``report["ok"]`` is True only when every net routed, every pin
    carries its own net's label, and no foreign net intrudes into a
    component's box.
    """
    h_img, w_img = shape[:2]
    comps = gt["components"]
    grid_shape = (h_img // stride + 1, w_img // stride + 1)

    by_net: dict[str, list] = {}
    for ci, c in enumerate(comps):
        sites = terminal_sites(c)
        for t in c["terminals"]:
            i = t["index"]
            if t["net"] is None or i >= len(sites):
                continue
            px, py, (nx, ny) = sites[i]
            by_net.setdefault(t["net"], []).append(
                {"pin": (px, py), "stub": (px + nx * stub, py + ny * stub), "comp": ci}
            )

    # labels start at 0 and background is -1, matching build_wire_nodes
    label_of_net = {net: i for i, net in enumerate(sorted(by_net))}
    node_map = np.full((h_img, w_img), -1, dtype=np.int32)
    unrouted: list[str] = []

    # Clearance/quantization ladder. A generous pad keeps routes away
    # from foreign symbols, but pad + grid quantization can also seal the
    # narrow channels a dense drawing genuinely uses — so a net that
    # cannot route at one level retries at a tighter one before being
    # declared unroutable.
    ladder = [(body_pad, stride), (max(body_pad // 2, 1), 2), (0, 2)]

    # trunks first, stubs afterwards: a stub must never be overpainted by
    # another net's trunk, or its pin would read the wrong label
    trunk_paint: list[tuple[int, list, int]] = []
    for net, entries in by_net.items():
        owners = {e["comp"] for e in entries}
        tree, used_stride = None, stride
        for pad_try, stride_try in ladder:
            gshape = (h_img // stride_try + 1, w_img // stride_try + 1)
            blocked = _blocked_for(owners, comps, shape, stride_try, pad_try)
            stub_cells = [_cell(e["stub"], stride_try, gshape) for e in entries]
            attempt: set = {stub_cells[0]}
            failed = False
            for target in stub_cells[1:]:
                if target in attempt:
                    continue
                path = _route(target, attempt, blocked)
                if path is None:
                    failed = True
                    break
                attempt.update(path)
            if not failed:
                tree, used_stride = attempt, stride_try
                break
        if tree is None:
            unrouted.append(net)
            tree, used_stride = set(), stride
        trunk_paint.append((label_of_net[net], sorted(tree), used_stride))

    trunk_only = np.full((h_img, w_img), -1, dtype=np.int32)
    for label, cells, cell_stride in trunk_paint:
        for r, c in cells:
            x, y = c * cell_stride, r * cell_stride
            cv2.rectangle(
                node_map, (x, y),
                (x + cell_stride - 1, y + cell_stride - 1),
                int(label), -1,
            )
            cv2.rectangle(
                trunk_only, (x, y),
                (x + cell_stride - 1, y + cell_stride - 1),
                int(label), -1,
            )
    for net, entries in by_net.items():
        label = label_of_net[net]
        for e in entries:
            cv2.line(
                node_map, (int(e["pin"][0]), int(e["pin"][1])),
                (int(e["stub"][0]), int(e["stub"][1])), int(label), thickness,
            )

    # ---- verification ----------------------------------------------------
    bad_pins, intruded = [], []
    for net, entries in by_net.items():
        label = label_of_net[net]
        for e in entries:
            x, y = int(e["stub"][0]), int(e["stub"][1])
            x = min(max(x, 0), w_img - 1)
            y = min(max(y, 0), h_img - 1)
            if node_map[y, x] != label:
                bad_pins.append((net, e["comp"]))

    # Intrusion is checked against the TRUE body (a conductor passing
    # near a symbol is normal in real drawings and is legitimately
    # snapping's problem; one crossing the body invents a terminal), and
    # only over TRUNK pixels. Stub geometry is dictated by the GT boxes:
    # where two annotated boxes overlap, a pin's own stub necessarily
    # lies inside its neighbour, which is a property of the drawing, not
    # a defect of this renderer.
    boxes = [_box_xyxy(c, 0) for c in comps]

    def _overlaps(a, b) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for ci, c in enumerate(comps):
        own = {
            label_of_net[t["net"]] for t in c["terminals"] if t["net"] is not None
        }
        # nets terminating on a component whose box overlaps this one may
        # legitimately appear inside it
        for cj, other in enumerate(comps):
            if cj != ci and _overlaps(boxes[ci], boxes[cj]):
                own |= {
                    label_of_net[t["net"]]
                    for t in other["terminals"] if t["net"] is not None
                }
        x1, y1, x2, y2 = boxes[ci]
        sub = trunk_only[
            max(0, int(y1)):min(h_img, int(y2) + 1),
            max(0, int(x1)):min(w_img, int(x2) + 1),
        ]
        foreign = (set(np.unique(sub)) - {-1}) - own
        if foreign:
            intruded.append((ci, sorted(int(f) for f in foreign)))

    report = {
        "n_nets": len(by_net),
        "unrouted_nets": sorted(set(unrouted)),
        "pins_with_wrong_label": bad_pins,
        "components_with_foreign_net": intruded,
        "ok": not unrouted and not bad_pins and not intruded,
    }
    return node_map, label_of_net, report
