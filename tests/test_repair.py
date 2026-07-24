"""Unit tests for the ERC diagnosis and the minimal repair ledger (C5).

The load-bearing property is the integrity rule: repair adds only
explicit SPICE constraints and never changes topology. These tests pin
the taxonomy (gauge vs assumption), minimality (one shunt per floating
subnet), and the ledger schema.
"""

from schematic2netlist.config import load_config
from schematic2netlist.erc import run_erc
from schematic2netlist.repair import build_ledger, export_ledger, repair_circuit
from schematic2netlist.simulate import parse_ngspice_output


def comp(i, cls, node_names):
    return {"id": i, "class": cls, "node_names": node_names, "nodes": node_names}


CFG = load_config()


# A well-formed divider: two resistors, a source, a ground. Fully solvable.
GOOD = [
    comp(0, "V-DC", ["n1", "0"]),
    comp(1, "Resistor", ["n1", "n2"]),
    comp(2, "Resistor", ["n2", "0"]),
    comp(3, "GND", ["0"]),
]


class TestERC:
    def test_clean_circuit_has_no_structural_issues(self):
        keys = {i.issue for i in run_erc(GOOD, {}, CFG)}
        assert "no_ground_reference" not in keys
        assert "no_dc_path_to_ground" not in keys
        assert "unsnapped_terminal" not in keys

    def test_missing_ground_detected(self):
        no_gnd = [
            comp(0, "V-DC", ["n1", "n2"]),
            comp(1, "Resistor", ["n1", "n2"]),
        ]
        keys = {i.issue for i in run_erc(no_gnd, {}, CFG)}
        assert "no_ground_reference" in keys

    def test_floating_net_via_capacitor_only(self):
        # n2 hangs off n1 only through a capacitor -> open at DC, floating
        floating = [
            comp(0, "V-DC", ["n1", "0"]),
            comp(1, "Resistor", ["n1", "0"]),
            comp(2, "Capacitor", ["n1", "n2"]),
            comp(3, "GND", ["0"]),
        ]
        issues = {i.issue: i for i in run_erc(floating, {}, CFG)}
        assert "no_dc_path_to_ground" in issues
        assert "n2" in issues["no_dc_path_to_ground"].location["nets"]

    def test_unsnapped_terminal_flagged(self):
        dangling = [
            comp(0, "Resistor", ["n1", None]),
            comp(1, "GND", ["0"]),
        ]
        issues = {i.issue: i for i in run_erc(dangling, {}, CFG)}
        assert "unsnapped_terminal" in issues
        assert issues["unsnapped_terminal"].behavior_changing is False

    def test_current_source_direction_is_gauge_only(self):
        cs = [comp(0, "I-DC", ["n1", "0"]), comp(1, "GND", ["0"])]
        keys = {i.issue for i in run_erc(cs, {}, CFG)}
        assert "unset_current_direction" in keys


class TestRepair:
    def test_clean_circuit_needs_no_assumptions(self):
        r = repair_circuit(GOOD, {}, CFG)
        # gauge entries (e.g. placeholder values, ground selection) may
        # exist, but no solvability assumptions should fire
        solvability = [
            e for e in r.entries
            if e.issue in ("no_ground_reference", "no_dc_path_to_ground")
        ]
        assert solvability == []
        assert not any(l.startswith("Rshunt") for l in r.extra_lines)

    def test_floating_subnet_gets_exactly_one_shunt(self):
        # n2 and n3 form one floating subnet (joined by a resistor);
        # minimality => ONE shunt grounds both, not two
        floating = [
            comp(0, "V-DC", ["n1", "0"]),
            comp(1, "Capacitor", ["n1", "n2"]),
            comp(2, "Resistor", ["n2", "n3"]),
            comp(3, "GND", ["0"]),
        ]
        r = repair_circuit(floating, {}, CFG)
        shunts = [l for l in r.extra_lines if l.startswith("Rshunt")]
        assert len(shunts) == 1

    def test_two_independent_floating_subnets_get_two_shunts(self):
        floating = [
            comp(0, "V-DC", ["n1", "0"]),
            comp(1, "Capacitor", ["n1", "n2"]),   # subnet A: n2
            comp(2, "Capacitor", ["0", "n3"]),    # subnet B: n3
            comp(3, "GND", ["0"]),
        ]
        r = repair_circuit(floating, {}, CFG)
        shunts = [l for l in r.extra_lines if l.startswith("Rshunt")]
        assert len(shunts) == 2

    def test_no_ground_gets_reference_tie(self):
        no_gnd = [
            comp(0, "V-DC", ["n1", "n2"]),
            comp(1, "Resistor", ["n1", "n2"]),
        ]
        r = repair_circuit(no_gnd, {}, CFG)
        assert any(l.startswith("Rref") and l.endswith(" 0 0") for l in r.extra_lines)
        refs = [e for e in r.entries if e.issue == "no_ground_reference"]
        assert refs and refs[0].category == "assumption"

    def test_gauge_and_assumption_counts(self):
        r = repair_circuit(GOOD, {}, CFG)
        assert r.num_gauge >= 1          # at least ground_selection
        assert r.num_gauge + r.num_assumptions == len(r.entries)

    def test_current_direction_is_logged_as_gauge(self):
        cs = [comp(0, "I-DC", ["n1", "0"]), comp(1, "Resistor", ["n1", "0"]),
              comp(2, "GND", ["0"])]
        r = repair_circuit(cs, {}, CFG)
        cd = [e for e in r.entries if e.issue == "unset_current_direction"]
        assert cd and cd[0].category == "gauge"
        assert cd[0].behavior_changing is False


class TestLedger:
    def test_ledger_schema_and_export(self, tmp_path):
        r = repair_circuit(GOOD, {}, CFG)
        ledger = build_ledger("circuit_x.jpg", True, True, r)
        assert ledger["schema_version"] == 1
        assert ledger["image"] == "circuit_x.jpg"
        assert ledger["num_gauge"] + ledger["num_assumptions"] == len(ledger["entries"])
        for e in ledger["entries"]:
            assert e["category"] in ("gauge", "assumption")
            assert "action" in e and "confidence" in e

        out = tmp_path / "ledger.json"
        export_ledger(ledger, str(out))
        assert out.exists()
        assert out.with_suffix(".txt").exists()          # readable sidecar
        assert "ASSUMPTION LEDGER" in out.with_suffix(".txt").read_text()


class TestSimulateDiagnostics:
    def test_singular_matrix_node_extraction(self):
        ok, cat, diag = parse_ngspice_output(
            "doAnalyses: singular matrix: check nodes n1 and n2\n", "", 1
        )
        assert cat == "singular_matrix"
        assert set(diag["nodes"]) == {"n1", "n2"}

    def test_floating_node_extraction(self):
        ok, cat, diag = parse_ngspice_output("node n7 is floating", "", 0)
        assert cat == "floating_node"
        assert diag["nodes"] == ["n7"]

    def test_success_has_no_nodes(self):
        ok, cat, diag = parse_ngspice_output("No. of Data Rows : 1", "", 0)
        assert ok and diag["nodes"] == []
