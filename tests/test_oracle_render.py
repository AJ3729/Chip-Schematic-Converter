"""Oracle mode-C GT wire rendering (C4).

These tests pin the two properties that made the previous star-topology
renderer produce an impossible negative wire attribution: the label
convention snapping reads, and the verification that refuses to pass an
unreadable render.
"""

import numpy as np

from schematic2netlist.oracle_render import render_gt_node_map, terminal_sites
from schematic2netlist.snapping import build_component_pin_nets


def _gt(components):
    return {"schema_version": 1, "image": "toy.jpg", "verified": True,
            "components": components}


def _res(cid, bbox, nets):
    return {
        "id": cid, "class": "Resistor", "bbox": list(bbox),
        "terminals": [{"index": i, "net": n} for i, n in enumerate(nets)],
    }


def test_background_is_minus_one_not_zero():
    """The pipeline's node maps use -1 for background; emitting 0 made
    snapping read the whole page as one giant node."""
    gt = _gt([_res(0, (100, 100, 40, 12), ["a", "b"]),
              _res(1, (200, 100, 40, 12), ["b", "c"])])
    node_map, labels, _ = render_gt_node_map(gt, (300, 300))
    assert node_map.dtype == np.int32
    assert node_map[5, 5] == -1                    # a background corner
    assert min(labels.values()) == 0               # labels start at 0
    assert set(np.unique(node_map)) - {-1} == set(labels.values())


def test_terminal_sites_respect_orientation():
    horiz = terminal_sites({"class": "Resistor", "bbox": (100, 100, 40, 10)})
    assert len(horiz) == 2
    assert horiz[0][0] < horiz[1][0]               # pins on the x extremes
    assert horiz[0][1] == horiz[1][1] == 100

    vert = terminal_sites({"class": "Resistor", "bbox": (100, 100, 10, 40)})
    assert vert[0][1] < vert[1][1]                 # pins on the y extremes
    assert vert[0][0] == vert[1][0] == 100


def test_ground_symbol_gets_one_site():
    assert len(terminal_sites({"class": "GND", "bbox": (50, 50, 20, 20)})) == 1


def test_render_verifies_and_snaps_back_to_gt():
    """A clean two-component chain must render OK and, when snapped,
    reproduce the GT nets — this is what makes mode C's claim ('perfect
    connectivity geometry') true rather than assumed."""
    gt = _gt([_res(0, (100, 150, 40, 12), ["a", "b"]),
              _res(1, (250, 150, 40, 12), ["b", "c"])])
    node_map, labels, report = render_gt_node_map(gt, (300, 400))
    assert report["ok"], report

    dets = [{"class": c["class"], "confidence": 1.0,
             "x": c["bbox"][0], "y": c["bbox"][1],
             "width": c["bbox"][2], "height": c["bbox"][3]}
            for c in gt["components"]]
    cfg = {"snapping": {"strategy": "boundary", "expand_step": 2,
                        "max_expand": 30, "window_depth": 4,
                        "ground_max_expand": 40,
                        "uniform_ground_max_expand": 26}}
    comps = build_component_pin_nets(dets, node_map, cfg)
    assert len(comps) == 2
    # both components must find two distinct nets, sharing exactly one
    nets0, nets1 = set(comps[0]["nodes"]), set(comps[1]["nodes"])
    assert None not in nets0 and None not in nets1
    assert len(nets0) == len(nets1) == 2
    assert len(nets0 & nets1) == 1                 # the shared net "b"


def test_report_flags_unroutable_net():
    """A pin walled in by foreign bodies must be reported, not silently
    rendered as a broken net."""
    walls = [
        _res(i, bbox, [None, None])
        for i, bbox in enumerate(
            [(100, 60, 200, 20), (100, 140, 200, 20),
             (10, 100, 20, 100), (190, 100, 20, 100)], start=2)
    ]
    gt = _gt([_res(0, (100, 100, 30, 10), ["a", "a2"]),
              _res(1, (100, 300, 30, 10), ["a", "a3"])] + walls)
    _node_map, _labels, report = render_gt_node_map(gt, (400, 400))
    assert not report["ok"]
    assert report["unrouted_nets"] or report["components_with_foreign_net"]
