"""Port-template terminal localization (C3).

The property under test is pin IDENTITY: terminal k must be the k-th
named port of the class, so the netlist writer's positional emission
(``M d g s``, ``D anode cathode``) means what it says.
"""

import numpy as np
import pytest

from schematic2netlist.ports import load_templates, match_ports, port_names
from schematic2netlist.snapping import _boundary_run_sites, build_component_pin_nets


@pytest.fixture(scope="module")
def templates():
    tpl = load_templates()
    if not tpl:
        pytest.skip("configs/port_templates.json not built")
    return tpl


def _det(cls, x, y, w, h):
    return {"class": cls, "confidence": 1.0, "x": x, "y": y,
            "width": w, "height": h}


def test_templates_cover_directional_classes(templates):
    for cls in ("Diode", "MOSFET-N", "BJT-NPN", "V-DC", "Op-Amp"):
        assert cls in templates, cls
        assert templates[cls]["port_names"], cls


def test_template_keys_are_canonical_and_complete(templates):
    """Template keys must be canonical class names, or they key on a
    name nothing downstream asks for and silently never apply — which
    is exactly what happened to the one-port source, whose port-data
    directory is spelled with a hyphen."""
    from schematic2netlist.classes import canonical_class, canonical_classes

    canon = set(canonical_classes())
    for key in templates:
        assert canonical_class(key) == key, f"{key!r} is not canonical"
        assert key in canon, f"{key!r} is not a known class"
    assert canon - set(templates) == set(), "some classes have no template"


def test_port_names_are_spice_argument_order(templates):
    """The netlist writer emits positionally, so template order must BE
    the SPICE order — this test is what keeps those two in sync."""
    assert port_names("Diode", templates)[:2] == ["Anode", "Cathode"]
    assert port_names("MOSFET-N", templates) == ["Drain", "Gate", "Source"]
    assert port_names("BJT-NPN", templates) == ["Collector", "Base", "Emitter"]
    assert port_names("V-DC", templates)[:2] == ["Positive", "Negative"]
    assert port_names("Resistor", templates) is None      # symmetric part


def test_match_assigns_distinct_runs_to_ports(templates):
    det = _det("Diode", 100, 100, 60, 20)
    # two crossings on the left and right edges
    run_sites = [(7, 70.0, 100.0), (9, 130.0, 100.0)]
    out = match_ports("Diode", det, run_sites, templates)
    assert out is not None
    nodes, info = out
    assert sorted(nodes) == [7, 9]                # both runs used, no reuse
    assert info["pose"].startswith("pose")
    assert info["port_names"] == ["Anode", "Cathode"]


def test_match_is_orientation_sensitive(templates):
    """A diode's two poses along one axis are mirror images; the matcher
    must not return the same assignment for both."""
    det = _det("Diode", 100, 100, 60, 20)
    left_right = [(1, 70.0, 100.0), (2, 130.0, 100.0)]
    right_left = [(2, 70.0, 100.0), (1, 130.0, 100.0)]
    a, _ = match_ports("Diode", det, left_right, templates)
    b, _ = match_ports("Diode", det, right_left, templates)
    assert a != b


def test_match_rejects_when_nothing_is_near(templates):
    det = _det("Diode", 100, 100, 20, 20)
    far = [(1, 900.0, 900.0), (2, 950.0, 950.0)]
    assert match_ports("Diode", det, far, templates) is None


def test_unknown_class_returns_none(templates):
    assert match_ports("Nonesuch", _det("Nonesuch", 5, 5, 4, 4),
                       [(1, 3.0, 3.0)], templates) is None


def test_boundary_run_sites_locate_crossings():
    node_map = np.full((60, 60), -1, dtype=np.int32)
    node_map[28:32, 0:21] = 3      # conductor reaching the left edge (x=20)
    node_map[28:32, 40:60] = 5     # and the right edge (x=40)
    sites = _boundary_run_sites(node_map, 20, 20, 40, 40)
    ids = {s[0] for s in sites}
    assert ids == {3, 5}
    by_id = {s[0]: s for s in sites}
    assert by_id[3][1] < by_id[5][1]        # left crossing is left of right


def test_ports_strategy_falls_back_without_regression():
    """With no usable wire evidence the ports strategy must still return
    the boundary result, never fewer terminals."""
    node_map = np.full((80, 80), -1, dtype=np.int32)
    node_map[38:42, 0:30] = 1
    node_map[38:42, 50:80] = 2
    dets = [_det("Resistor", 40, 40, 20, 8)]
    cfg = {"snapping": {"strategy": "ports", "expand_step": 2, "max_expand": 30,
                        "window_depth": 4, "ground_max_expand": 40,
                        "uniform_ground_max_expand": 26}}
    comps = build_component_pin_nets(dets, node_map, cfg)
    assert len(comps) == 1
    assert set(comps[0]["nodes"]) == {1, 2}
