"""The wire field as a graph whose PARTITION is the netlist.

Every previous attempt made a single-site decision -- split here or do not -- and
all four oracles came back null because two merged nets are joined by more than
one route: cutting the shortest leaves the rest. Erasing the ink at every correct
location outright removed only 24.6% of welds while costing 0.059 terminal-pair
F1, which is the same fact stated in the most direct possible way.

The action was wrong, not the perception. Recovering nets is a MULTICUT: partition
the wire field so that terminals of different nets land in different parts. A
multicut removes a consistent SET of connections at once, which is what a
multiply-connected weld requires, and it can pass one wire straight through a
crossing while separating the other -- a thing no erase-or-keep decision can
express.

The graph:

  pieces    the wire mask with a small disk removed at every crossing candidate.
            Candidates come from the raw ink's distance transform and cover 100%
            of weld paths, so every place two nets can meet becomes a boundary.
            The connected components that remain are the nodes.
  edges     two pieces share an edge when they enter the SAME removed disk. The
            edge asks "are these one conductor" -- and a 4-arm crossing yields
            six such questions rather than one erase-or-keep bit, which is
            exactly the expressiveness that was missing.

FEASIBILITY comes first and gates everything. A partition can only be correct if
no single piece already carries terminals of two different ground-truth nets --
if it does, the graph is too coarse and no solver, learned or exact, can recover
the answer. That is measured here before any solver is written.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np


def build_graph(wires: np.ndarray, sites, disk: int = 7):
    """(piece_labels, n_pieces, edges, site_of_edge) from a wire mask.

    ``sites`` are (y, x) crossing candidates; a disk of the given radius is
    removed at each, and pieces meeting inside one disk become graph neighbours.
    """
    cut = wires.copy()
    for (y, x) in sites:
        cv2.circle(cut, (int(x), int(y)), disk, 0, -1)
    n, lab = cv2.connectedComponents((cut > 0).astype(np.uint8), connectivity=8)

    edges = defaultdict(set)          # (a, b) -> {site indices}
    H, W = wires.shape
    for si, (y, x) in enumerate(sites):
        y0, y1 = max(0, int(y) - disk - 3), min(H, int(y) + disk + 4)
        x0, x1 = max(0, int(x) - disk - 3), min(W, int(x) + disk + 4)
        sub = lab[y0:y1, x0:x1]
        present = sorted({int(v) for v in np.unique(sub) if v > 0})
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                edges[(present[i], present[j])].add(si)
    return lab, n, dict(edges)


def piece_of_terminals(lab: np.ndarray, term_xy: dict, max_r: int = 26):
    """Nearest wire piece for each terminal, searched outward from its pin."""
    H, W = lab.shape
    out = {}
    for t, (y, x) in term_xy.items():
        best = None
        for r in range(2, max_r, 2):
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            sub = lab[y0:y1, x0:x1]
            vals = sub[sub > 0]
            if vals.size:
                ys, xs = np.nonzero(sub > 0)
                d = (ys + y0 - y) ** 2 + (xs + x0 - x) ** 2
                best = int(sub[ys[int(d.argmin())], xs[int(d.argmin())]])
                break
        if best is not None:
            out[t] = best
    return out


def feasibility(piece_of: dict, gt_net_of: dict):
    """Can the graph express the GT partition at all?

    Returns (n_pieces_used, n_pieces_carrying_two_nets, offending_pieces).
    A piece carrying terminals of two different GT nets must be split further;
    until it is, no solver can be correct on that image.
    """
    nets_on_piece = defaultdict(set)
    for t, pc in piece_of.items():
        gn = gt_net_of.get(t)
        if gn is not None:
            nets_on_piece[pc].add(gn)
    bad = {p: s for p, s in nets_on_piece.items() if len(s) > 1}
    return len(nets_on_piece), len(bad), bad
