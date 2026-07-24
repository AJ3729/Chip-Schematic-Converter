"""Unit tests for the ngspice output parser and failure taxonomy."""

from schematic2netlist.simulate import FAILURE_CATEGORIES, parse_ngspice_output


class TestParseNgspiceOutput:
    def test_clean_run_ok(self):
        ok, cat, _ = parse_ngspice_output(
            "Note: No compatibility mode selected!\n"
            "No. of Data Rows : 1\n",
            "",
            0,
        )
        assert ok is True
        assert cat == "ok"

    def test_singular_matrix(self):
        ok, cat, _ = parse_ngspice_output(
            "doAnalyses: singular matrix:  check nodes n1 and n2\n", "", 1
        )
        assert ok is False
        assert cat == "singular_matrix"

    def test_singular_matrix_case_insensitive_in_stderr(self):
        ok, cat, _ = parse_ngspice_output("", "Singular Matrix detected", 0)
        assert ok is False
        assert cat == "singular_matrix"

    def test_floating_node(self):
        ok, cat, _ = parse_ngspice_output(
            "Warning: node n3 is floating\n", "", 0
        )
        assert ok is False
        assert cat == "floating_node"

    def test_parse_error_on_nonzero_returncode(self):
        ok, cat, _ = parse_ngspice_output(
            "Error on line 3: unknown device\n", "", 1
        )
        assert ok is False
        assert cat == "parse_error"

    def test_singular_takes_priority_over_returncode(self):
        ok, cat, _ = parse_ngspice_output("singular matrix", "", 1)
        assert cat == "singular_matrix"

    def test_all_categories_declared(self):
        for cat in ("ok", "parse_error", "singular_matrix", "floating_node",
                    "timeout", "ngspice_missing", "ngspice_error"):
            assert cat in FAILURE_CATEGORIES
