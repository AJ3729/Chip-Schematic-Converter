"""Learned net assembly (C2 headline).

These pin the contract rather than the model's accuracy: the learned
path must separate a crossing it is told about, must leave a junction
alone, and must never lose the detector's own crossover evidence.
"""

from pathlib import Path

import numpy as np
import pytest

from schematic2netlist.nodes import (
    build_wire_nodes,
    build_wire_nodes_crossover_aware,
)

WEIGHTS = Path("experiments/junction/full_cpu/best.pt")


def _crossing(n=80):
    """Two wires crossing at the centre — one CC, but two nets."""
    m = np.zeros((n, n), np.uint8)
    m[38:42, :] = 255
    m[:, 38:42] = 255
    return m


def _box(cx, cy, w, h):
    return {"x": float(cx), "y": float(cy), "width": float(w), "height": float(h)}


def test_plain_cc_merges_a_crossing():
    """The ceiling the learned path exists to break: two crossing wires
    are one connected component and therefore one net."""
    node_map, n = build_wire_nodes(_crossing(), connectivity=8)
    assert n == 1


def test_crossover_box_separates_a_crossing():
    node_map, n = build_wire_nodes_crossover_aware(
        _crossing(), [_box(40, 40, 20, 20)], connectivity=8)
    assert n >= 2, "notch + opposite-arm relink should yield two nets"


def test_no_boxes_leaves_topology_untouched():
    a, na = build_wire_nodes(_crossing(), connectivity=8)
    b, nb = build_wire_nodes_crossover_aware(_crossing(), [], connectivity=8)
    assert na == nb
    assert np.array_equal(a, b)


@pytest.mark.skipif(not WEIGHTS.exists(),
                    reason="junction classifier weights not trained yet")
def test_learned_path_reports_what_it_examined():
    from schematic2netlist.nodes import build_wire_nodes_learned

    node_map, n, info = build_wire_nodes_learned(
        _crossing(), [], str(WEIGHTS), threshold=0.4, connectivity=8)
    # the audit trail is the point: a run must be inspectable afterwards
    assert info["sites_found"] >= 1
    assert info["classified"] == info["sites_found"] - info["detector_labelled"]
    assert 0.0 <= info["threshold"] <= 1.0
    assert n >= 1


@pytest.mark.skipif(not WEIGHTS.exists(),
                    reason="junction classifier weights not trained yet")
def test_learned_path_honours_detector_boxes():
    """A detected crossover is evidence from the drawing itself and must
    be acted on regardless of what the model thinks."""
    from schematic2netlist.nodes import build_wire_nodes_learned

    _m, n, info = build_wire_nodes_learned(
        _crossing(), [_box(40, 40, 20, 20)], str(WEIGHTS),
        threshold=0.99,          # model will judge nothing a crossing
        connectivity=8)
    assert info["detector_labelled"] >= 1
    assert n >= 2, "detector's crossover box must still split the nets"
