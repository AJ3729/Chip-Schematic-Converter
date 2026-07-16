"""Unit tests for node naming and the SPICE netlist writer."""

import pytest

from schematic2netlist.netlist import (
    GroundNotFoundError,
    assign_node_names,
    build_node_name_map,
    export_spice_netlist,
)


def comp(i, cls, nodes, kind="two_terminal"):
    return {"id": i, "class": cls, "nodes": nodes, "kind": kind}


class TestBuildNodeNameMap:
    def test_ground_node_is_named_zero(self):
        comps = [
            comp(0, "resistor", [3, 5]),
            comp(1, "ground", [5, 5], kind="ground"),
        ]
        name_map = build_node_name_map(comps)
        assert name_map[5] == "0"
        assert name_map[3] == "n1"

    def test_n0_is_never_used(self):
        comps = [
            comp(0, "resistor", [1, 2]),
            comp(1, "capacitor", [2, 3]),
            comp(2, "ground", [2, 2], kind="ground"),
        ]
        name_map = build_node_name_map(comps)
        assert name_map[2] == "0"
        assert "n0" not in name_map.values()
        assert sorted(name_map.values()) == ["0", "n1", "n2"]

    def test_most_connected_fallback_without_ground(self):
        # node 7 touches three terminals, node 4 only one
        comps = [
            comp(0, "resistor", [7, 4]),
            comp(1, "capacitor", [7, 9]),
            comp(2, "inductor", [7, 9]),
        ]
        name_map = build_node_name_map(comps, ground_fallback="most_connected")
        assert name_map[7] == "0"

    def test_fail_policy_raises_without_ground(self):
        comps = [comp(0, "resistor", [1, 2])]
        with pytest.raises(GroundNotFoundError):
            build_node_name_map(comps, ground_fallback="fail")

    def test_unsnapped_ground_uses_fallback(self):
        comps = [
            comp(0, "resistor", [1, 2]),
            comp(1, "ground", [None, None], kind="ground"),
        ]
        name_map = build_node_name_map(comps, ground_fallback="most_connected")
        assert "0" in name_map.values()


class TestExportSpiceNetlist:
    def _export(self, comps, tmp_path):
        name_map = build_node_name_map(comps)
        assign_node_names(comps, name_map)
        out = tmp_path / "netlist.sp"
        info = export_spice_netlist(comps, str(out))
        return out.read_text(), info

    def test_basic_resistor_written(self, tmp_path):
        comps = [
            comp(0, "resistor", [1, 2]),
            comp(1, "ground", [2, 2], kind="ground"),
        ]
        text, info = self._export(comps, tmp_path)
        assert "R1 n1 0 1k" in text
        assert info["wrote_any"] is True
        assert text.rstrip().endswith(".end")
        assert ".op" in text

    def test_ground_symbol_not_emitted_as_element(self, tmp_path):
        comps = [
            comp(0, "resistor", [1, 2]),
            comp(1, "ground", [2, 2], kind="ground"),
        ]
        text, _ = self._export(comps, tmp_path)
        element_lines = [
            l for l in text.splitlines() if l and not l.startswith("*") and l[0] not in ".;"
        ]
        assert len(element_lines) == 1  # only the resistor

    def test_unsnapped_component_skipped_with_comment(self, tmp_path):
        comps = [
            comp(0, "resistor", [1, None]),
            comp(1, "capacitor", [1, 2]),
            comp(2, "ground", [1, 1], kind="ground"),
        ]
        text, info = self._export(comps, tmp_path)
        assert "* UNSNAPPED resistor" in text
        assert not any(
            line.startswith("R") for line in text.splitlines()
        )
        assert "C1" in text
        assert len(info["skipped"]) == 1

    def test_same_node_component_skipped(self, tmp_path):
        comps = [
            comp(0, "resistor", [1, 1]),
            comp(1, "capacitor", [1, 2]),
            comp(2, "ground", [1, 1], kind="ground"),
        ]
        text, info = self._export(comps, tmp_path)
        assert "* SAME_NODE_SKIPPED resistor" in text
        assert not any(line.startswith("R") for line in text.splitlines())
        assert len(info["skipped"]) == 1

    def test_no_valid_components_warning(self, tmp_path):
        comps = [
            comp(0, "resistor", [None, None]),
            comp(1, "ground", [1, 1], kind="ground"),
        ]
        text, info = self._export(comps, tmp_path)
        assert info["wrote_any"] is False
        assert "* WARNING: no valid components written" in text

    def test_placeholder_values_configurable(self, tmp_path):
        comps = [
            comp(0, "resistor", [1, 2]),
            comp(1, "ground", [2, 2], kind="ground"),
        ]
        name_map = build_node_name_map(comps)
        assign_node_names(comps, name_map)
        out = tmp_path / "netlist.sp"
        export_spice_netlist(comps, str(out), placeholders={"resistor": "4.7k"})
        assert "R1 n1 0 4.7k" in out.read_text()
