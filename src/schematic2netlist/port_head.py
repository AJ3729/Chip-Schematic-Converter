"""Decide WHICH boundary crossing is which named pin, by looking at the symbol.

`ports.py` picks a pose from wire geometry alone. Its own docstring says it
"cannot read the arrowhead at all", and that is exactly the evidence that
separates a BJT's collector from its emitter, an op-amp's inverting input from
its non-inverting one, and a MOSFET's gate from a channel terminal. So terminal
ORDER is decided by where leads happen to leave the box, which is close to a
coin flip on one axis -- and `netlist.py` writes `Q<c> <b> <e>`,
`M<d> <g> <s>` and `E<out> 0 <in+> <in->` straight off that order, so a
reversed transistor is emitted as a reversed transistor and simulates wrongly
while every topology metric reports success.

This module loads the small per-class heatmap model trained by
`scripts/train_port_head.py` and uses it to REORDER the pins that snapping
already found.

THE INVARIANT THAT MAKES THIS SAFE: it only ever permutes the list. The set of
nodes a component connects to is decided by snapping and is passed through
untouched, so net-level topology cannot move. `tests/` asserts this, and the
benchmark must come out bit-identical on every net metric.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np

_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch  # imported lazily: the pipeline runs without it
        _TORCH = torch
    return _TORCH


PREFIX = {"BJT-NPN": "bjt_npn", "BJT-PNP": "bjt_pnp", "MOSFET-N": "mosfet_n",
          "MOSFET-P": "mosfet_p", "Op-Amp": "opamp"}


class _Net:
    """Rebuilt to match scripts/train_port_head.py PortNet exactly."""

    @staticmethod
    def build(k: int):
        torch = _torch()
        nn = torch.nn

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))
        return nn.Sequential(
            blk(1, 32), blk(32, 32), nn.MaxPool2d(2),
            blk(32, 64), blk(64, 64),
            blk(64, 128), blk(128, 128),
            nn.Conv2d(128, k, 1),
        )


@functools.lru_cache(maxsize=8)
def load_head(cls: str, weights_dir: str):
    """Load one class's head, or None if it was never trained."""
    p = Path(weights_dir) / f"{PREFIX.get(cls, cls)}.pt"
    if not p.exists():
        return None
    torch = _torch()
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    net = _Net.build(len(ck["ports"]))
    # the trainer wraps the Sequential in a Module, so every key arrives
    # prefixed "net."; loading a bare Sequential without stripping it raises,
    # and the caller's fallback would swallow that into silent no-op
    state = {k[4:] if k.startswith("net.") else k: v for k, v in ck["state"].items()}
    net.load_state_dict(state)
    net.eval()
    return {"net": net, "ports": ck["ports"], "S": ck["S"], "HS": ck["HS"],
            "margin": ck["MARGIN"]}


def _crop(gray: np.ndarray, det: dict, margin: float, S: int):
    """The box interior, expanded exactly as training did."""
    bx, by = float(det["x"]), float(det["y"])
    bw, bh = float(det["width"]), float(det["height"])
    x0, y0 = bx - bw / 2, by - bh / 2
    x1, y1 = bx + bw / 2, by + bh / 2
    mx, my = bw * margin, bh * margin
    X0, Y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    X1 = min(gray.shape[1], int(x1 + mx))
    Y1 = min(gray.shape[0], int(y1 + my))
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return None, None
    # Drawing the annotation rectangle here (to mimic the burned-in green box
    # every training crop carries) was tried and MEASURED WORSE: 0.571 -> 0.392
    # order accuracy. Keeping the crop clean.
    sub = gray[Y0:Y1, X0:X1]
    g = cv2.resize(sub, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g, (X0, Y0, X1, Y1)


def reorder(cls: str, det: dict, sites: list, nodes: list, gray: np.ndarray,
            cfg: dict) -> tuple[list, dict] | None:
    """Re-permute ``nodes`` into the head's predicted port order.

    ``sites`` are ``(node_id, x, y)`` boundary crossings in frame coordinates.
    Returns ``(nodes_reordered, info)`` or None to leave the caller's order
    alone -- when the head is absent, the crop is degenerate, a pin has no
    site, or confidence is below threshold. Falling back is always safe: the
    template order is what shipped before.
    """
    ph = cfg.get("snapping", {}).get("port_head", {})
    if not ph.get("enabled"):
        return None
    head = load_head(cls, ph.get("weights_dir", "experiments/port_head"))
    if head is None or not sites or not nodes:
        return None
    k = len(head["ports"])
    if len(nodes) != k:
        return None

    g, box = _crop(gray, det, head["margin"], head["S"])
    if g is None:
        return None
    X0, Y0, X1, Y1 = box

    torch = _torch()
    with torch.no_grad():
        hm = head["net"](torch.from_numpy(g)[None, None]).numpy()[0]
    HS = hm.shape[-1]

    # one candidate site per node currently assigned, at that node's crossing
    site_xy: dict[int, tuple[float, float]] = {}
    for nid, sx, sy in sites:
        site_xy.setdefault(int(nid), (float(sx), float(sy)))
    cand = []
    for nid in nodes:
        if nid is None or int(nid) not in site_xy:
            return None                      # no evidence: keep template order
        cand.append(site_xy[int(nid)])

    # score[p][c] = the head's belief that candidate c is port p
    score = np.zeros((k, k), np.float32)
    for c, (sx, sy) in enumerate(cand):
        fx = (sx - X0) / max(1, X1 - X0)
        fy = (sy - Y0) / max(1, Y1 - Y0)
        hx = int(np.clip(round(fx * (HS - 1)), 0, HS - 1))
        hy = int(np.clip(round(fy * (HS - 1)), 0, HS - 1))
        for p in range(k):
            score[p, c] = hm[p, hy, hx]

    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-score)
    total = float(score[rows, cols].sum())
    if total < float(ph.get("min_total_score", 0.0)):
        return None

    # The head's port order is the dataset's directory order
    # (Base, Collector, Emitter); the pipeline's terminal order is the port
    # TEMPLATE's (Collector, Base, Emitter), which is what netlist.py writes
    # as Q<c> <b> <e>. Aligning by position rather than by NAME silently
    # applies a fixed (1,0,2) permutation to every BJT -- worse than doing
    # nothing at all. Align by name; fall back only if the names do not
    # correspond.
    from schematic2netlist import ports as _ports
    target = _ports.port_names(cls)
    assign = {head["ports"][p]: nodes[c] for p, c in zip(rows, cols)}
    if target and set(target) == set(assign):
        out = [assign[nm] for nm in target]
    elif target is None:
        out = [assign[nm] for nm in head["ports"]]
    else:
        return None
    if any(o is None for o in out):
        return None
    changed = list(out) != list(nodes)
    return out, {"port_head": True, "score": round(total, 4),
                 "changed": bool(changed), "ports": head["ports"]}
