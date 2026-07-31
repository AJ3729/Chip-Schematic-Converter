"""Constraint-triggered connectivity repair.

The stage is licensed by two facts about the verified ground truth: a component
with every pin on one net occurs at 0.60%, and a net with a single terminal at
0.00% (0 of 1509). So both are near-certain faults rather than rare-but-real
configurations. These tests pin the detector, the two repair operations, and
above all the refusals -- a repair that fires where it should not is worse than
one that never fires, because it can break an image that already scored.
"""

from __future__ import annotations

import cv2
import numpy as np

from schematic2netlist.connectivity_repair import (_bridge_fragment,
                                                   _erase_body,
                                                   find_violations,
                                                   repair_connectivity)


def comp(cid, cls, nets):
    return {"id": cid, "class": cls, "nodes": [1] * len(nets),
            "node_names": list(nets)}


def det(cls, x=50, y=50, w=30, h=30):
    return {"class": cls, "x": x, "y": y, "width": w, "height": h,
            "confidence": 0.9}


def test_finds_a_self_short():
    comps = [comp(0, "Resistor", ["n1", "n1"]), comp(1, "Capacitor", ["n2", "n3"])]
    shorts, ones = find_violations(comps, [det("Resistor"), det("Capacitor")])
    assert shorts == [0]


def test_ground_is_never_a_self_short():
    """A ground symbol legitimately stores one node twice; flagging it would
    erase the body of every GND in the drawing."""
    comps = [comp(0, "GND", ["n1", "n1"])]
    shorts, _ = find_violations(comps, [det("GND")])
    assert shorts == []


def test_finds_one_terminal_nets():
    comps = [comp(0, "Resistor", ["n1", "n2"]), comp(1, "Capacitor", ["n2", "n3"])]
    _, ones = find_violations(comps, [det("Resistor"), det("Capacitor")])
    assert set(ones) == {"n1", "n3"}          # n2 is shared, so it is fine


def test_erase_body_is_a_noop_on_empty_box():
    """Nothing to erase means no action, so the caller does not spend a rebuild
    pass or report an action it did not take."""
    wires = np.zeros((100, 100), np.uint8)
    assert _erase_body(wires, det("Resistor"), 0.5) is False
    wires[40:60, 40:60] = 255
    before = int((wires > 0).sum())
    assert _erase_body(wires, det("Resistor"), 0.5) is True
    assert int((wires > 0).sum()) < before


def test_erase_body_spares_the_leads():
    """Only an INNER band is erased: ink at the box edges, where the leads
    attach, must survive or snapping loses the terminals it needs."""
    wires = np.zeros((120, 120), np.uint8)
    wires[58:62, 20:100] = 255                # a straight body path + leads
    d = det("Capacitor", x=60, y=60, w=40, h=40)
    assert _erase_body(wires, d, 0.5) is True
    assert wires[58:62, 20:35].any(), "left lead was erased"
    assert wires[58:62, 85:100].any(), "right lead was erased"


def test_bridge_fragment_refuses_a_perpendicular_stub():
    """A lead pointing INTO a component is perpendicular to the gap it would
    cross. Bridging it would weld two different nets through the body, so the
    collinearity test must refuse."""
    wires = np.zeros((160, 160), np.uint8)
    node_map = np.full((160, 160), -1, np.int32)
    wires[40:100, 78:82] = 255                # a VERTICAL fragment
    node_map[40:100, 78:82] = 5
    wires[76:84, 110:150] = 255               # a target far to the RIGHT
    node_map[76:84, 110:150] = 6
    before = int((wires > 0).sum())
    ok = _bridge_fragment(wires, node_map, 5, max_gap=60, dir_tol_deg=40)
    assert ok is False, "bridged perpendicular to the fragment's own stroke"
    assert int((wires > 0).sum()) == before


def test_bridge_fragment_accepts_a_collinear_continuation():
    wires = np.zeros((160, 160), np.uint8)
    node_map = np.full((160, 160), -1, np.int32)
    wires[78:82, 20:70] = 255                 # horizontal fragment
    node_map[78:82, 20:70] = 5
    wires[78:82, 90:140] = 255                # collinear target, 20 px gap
    node_map[78:82, 90:140] = 6
    assert _bridge_fragment(wires, node_map, 5, max_gap=60,
                            dir_tol_deg=40) is True
    assert wires[79, 80] > 0, "the gap was not actually filled"


def test_bridge_fragment_respects_the_gap_cap():
    wires = np.zeros((300, 300), np.uint8)
    node_map = np.full((300, 300), -1, np.int32)
    wires[148:152, 20:70] = 255
    node_map[148:152, 20:70] = 5
    wires[148:152, 240:290] = 255             # ~170 px away
    node_map[148:152, 240:290] = 6
    assert _bridge_fragment(wires, node_map, 5, max_gap=60,
                            dir_tol_deg=40) is False


def test_repair_is_a_noop_when_no_constraint_is_violated():
    """A clean circuit must come back untouched, and the input wire mask must
    not be mutated -- the caller may still be holding it."""
    wires = np.zeros((100, 100), np.uint8)
    wires[48:52, 10:90] = 255
    original = wires.copy()
    node_map = np.full((100, 100), -1, np.int32)
    node_map[48:52, 10:50] = 1
    node_map[48:52, 50:90] = 2
    comps = [comp(0, "Resistor", ["n1", "n2"]), comp(1, "Capacitor", ["n2", "n1"])]
    cfg = {"connectivity_repair": {"enabled": True, "passes": 2}}
    w, nm, nn, info, cs, rep = repair_connectivity(
        wires, node_map, comps, [det("Resistor"), det("Capacitor")], cfg,
        lambda _w: (None, None, None, []))
    assert rep["applied"] is False and rep["n_actions"] == 0
    assert np.array_equal(wires, original), "input mask was mutated"
    assert cs is comps
