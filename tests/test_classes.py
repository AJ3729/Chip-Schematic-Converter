"""Unit tests for the canonical class vocabulary and role-based
netlist dispatch (published names AND legacy aliases)."""

from schematic2netlist.classes import (
    canonical_class,
    canonical_classes,
    class_role,
    class_terminals,
    is_ground,
)
from schematic2netlist.netlist import (
    assign_node_names,
    build_node_name_map,
    export_spice_netlist,
)


def comp(i, cls, nodes):
    return {"id": i, "class": cls, "nodes": nodes}


class TestVocabulary:
    def test_seventeen_canonical_classes(self):
        assert len(canonical_classes()) == 17

    def test_legacy_aliases_canonicalize(self):
        assert canonical_class("ground") == "GND"
        assert canonical_class("DC Supply") == "V-DC"
        assert canonical_class("Independent AC Current") == "I-AC"
        assert canonical_class("MOSFET Transistor") == "MOSFET-N"
        assert canonical_class("operational amplifier") == "Op-Amp"

    def test_case_insensitive(self):
        assert canonical_class("RESISTOR") == "Resistor"
        assert canonical_class("gnd") == "GND"

    def test_unknown_passes_through(self):
        assert canonical_class("flux capacitor") == "flux capacitor"
        assert class_role("flux capacitor") == "unknown"

    def test_roles_and_terminals(self):
        assert is_ground("GND") and is_ground("ground")
        assert class_role("V-AC") == "vac"
        assert class_terminals("BJT-PNP") == 3
        assert class_terminals("V-DC (one port)") == 1
        assert class_role("Wire Crossover") == "none"
        assert class_terminals("Wire Crossover") == 0


class TestRoleBasedNetlist:
    def _export(self, comps, tmp_path):
        name_map = build_node_name_map(comps)
        assign_node_names(comps, name_map)
        out = tmp_path / "netlist.sp"
        info = export_spice_netlist(comps, str(out))
        return out.read_text(), info

    def test_published_names_work(self, tmp_path):
        comps = [
            comp(0, "Resistor", [1, 2]),
            comp(1, "V-DC", [1, 2]),
            comp(2, "GND", [2, 2]),
        ]
        text, info = self._export(comps, tmp_path)
        assert "R1 n1 0 1k" in text
        assert "V1 n1 0 DC 5" in text
        assert info["wrote_any"] is True

    def test_ac_supply_no_longer_shadowed_by_dc_branch(self, tmp_path):
        # legacy quirk: "AC Supply" fell into the DC branch (emitted DC 5)
        comps = [
            comp(0, "AC Supply", [1, 2]),
            comp(1, "ground", [2, 2]),
        ]
        text, _ = self._export(comps, tmp_path)
        assert "V1 n1 0 AC 1" in text
        assert "DC 5" not in text

    def test_diode_and_zener_share_counter_no_duplicate_names(self, tmp_path):
        # legacy quirk: separate counters emitted duplicate "D1" names
        comps = [
            comp(0, "Diode", [1, 2]),
            comp(1, "Zener Diode", [2, 3]),
            comp(2, "GND", [3, 3]),
        ]
        text, _ = self._export(comps, tmp_path)
        assert "D1 " in text and "D2 " in text
        assert ".model Ddefault D" in text
        assert ".model Zdefault D(bv=5.1)" in text

    def test_mosfet_three_terminal(self, tmp_path):
        comps = [
            comp(0, "MOSFET-N", [1, 2, 3]),
            comp(1, "GND", [3, 3]),
        ]
        text, _ = self._export(comps, tmp_path)
        # drain gate source source(body) model
        assert "M1 n1 n2 0 0 NMOSdefault" in text
        assert ".model NMOSdefault NMOS" in text

    def test_mosfet_with_only_two_snapped_nodes_is_unsnapped(self, tmp_path):
        comps = [
            comp(0, "MOSFET-N", [1, 2]),
            comp(1, "GND", [2, 2]),
        ]
        text, info = self._export(comps, tmp_path)
        assert "* UNSNAPPED MOSFET-N" in text
        assert len(info["skipped"]) == 1

    def test_one_port_source_referenced_to_ground(self, tmp_path):
        comps = [
            comp(0, "V-DC (one port)", [1, None]),
            comp(1, "Resistor", [1, 2]),
            comp(2, "GND", [2, 2]),
        ]
        text, _ = self._export(comps, tmp_path)
        assert "V1 n1 0 DC 5" in text

    def test_opamp_as_vcvs(self, tmp_path):
        comps = [
            comp(0, "Op-Amp", [1, 2, 3]),
            comp(1, "GND", [1, 1]),
        ]
        text, _ = self._export(comps, tmp_path)
        # E out 0 in+ in- gain, terminals [in+, in-, out]
        assert "E1 n2 0 0 n1 100k" in text

    def test_wire_crossover_not_emitted(self, tmp_path):
        comps = [
            comp(0, "Resistor", [1, 2]),
            comp(1, "Wire Crossover", [1, 2]),
            comp(2, "GND", [2, 2]),
        ]
        text, _ = self._export(comps, tmp_path)
        assert "Crossover" not in text
