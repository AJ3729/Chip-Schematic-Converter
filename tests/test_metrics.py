"""Unit tests for topology metrics on known toy graphs with exact
hand-computed expected values."""

import pytest

from schematic2netlist.metrics import (
    coverage_stats,
    graph_edit_distance,
    net_level_metrics,
    normalized_ged,
    per_component_connected_accuracy,
    terminal_pair_metrics,
)


def comp(i, cls, nets):
    return {"id": i, "class": cls, "nets": nets}


# GT: two resistors in parallel — nets A (terminal 0 of both) and
# B (terminal 1 of both).
GT = [
    comp(0, "resistor", ["A", "B"]),
    comp(1, "resistor", ["A", "B"]),
]

# Prediction that merges everything into one net.
PRED_MERGED = [
    comp(0, "resistor", ["X", "X"]),
    comp(1, "resistor", ["X", "X"]),
]


class TestTerminalPairMetrics:
    def test_identical_graphs_perfect(self):
        m = terminal_pair_metrics(GT, GT)
        assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_net_names_do_not_matter_only_grouping(self):
        renamed = [
            comp(0, "resistor", ["x", "y"]),
            comp(1, "resistor", ["x", "y"]),
        ]
        m = terminal_pair_metrics(renamed, GT)
        assert m["f1"] == 1.0

    def test_merged_nets_exact_values(self):
        # pred pairs: C(4,2) = 6; gt pairs: 2; intersection: 2
        m = terminal_pair_metrics(PRED_MERGED, GT)
        assert m["precision"] == pytest.approx(1 / 3)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(0.5)

    def test_empty_prediction(self):
        empty = [
            comp(0, "resistor", [None, None]),
            comp(1, "resistor", [None, None]),
        ]
        m = terminal_pair_metrics(empty, GT)
        assert m["precision"] == 1.0  # vacuous: no predicted pairs
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0


class TestNetLevelMetrics:
    def test_identical_graphs_perfect(self):
        m = net_level_metrics(GT, GT)
        assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_merged_nets_exact_values(self):
        # one predicted net of 4 terminals vs two GT nets of 2;
        # Hungarian matches the merged net to one GT net (overlap 2)
        m = net_level_metrics(PRED_MERGED, GT)
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(0.5)


class TestPerComponentAccuracy:
    def test_identical_is_one(self):
        assert per_component_connected_accuracy(GT, GT) == 1.0

    def test_merged_prediction_still_covers_gt_pairs(self):
        # merged prediction contains every GT pair, so components count
        # as connected (precision suffers in the pair metric instead)
        assert per_component_connected_accuracy(PRED_MERGED, GT) == 1.0

    def test_missing_connection(self):
        pred = [
            comp(0, "resistor", ["A", None]),
            comp(1, "resistor", ["A", "B"]),
        ]
        # component 0 is missing its pair on net B, component 1 has both
        # of its GT pairs broken/kept? GT pairs: {c0t0,c1t0} on A and
        # {c0t1,c1t1} on B. pred pairs: only {c0t0,c1t0}. c0 misses the
        # B pair, c1 misses it too -> both incomplete? c0 pairs: A-pair
        # (present) + B-pair (absent) -> incomplete. c1: same. But the
        # A-pair belongs to both. correct = 0/2.
        assert per_component_connected_accuracy(pred, GT) == 0.0


class TestGraphEditDistance:
    def test_identical_graphs_zero(self):
        assert graph_edit_distance(GT, GT) == 0.0
        assert normalized_ged(GT, GT) == 0.0

    def test_merged_nets_exact_ged(self):
        # pred graph: nodes {c0, c1, X}, edges {c0-X, c1-X} -> 3 + 2
        # gt graph:   nodes {c0, c1, A, B}, edges {c0-A, c0-B, c1-A, c1-B} -> 4 + 4
        # edit path: insert net node (1) + insert two edges (2) = 3
        assert graph_edit_distance(PRED_MERGED, GT) == 3.0
        assert normalized_ged(PRED_MERGED, GT) == pytest.approx(3 / 13)

    def test_class_mismatch_costs_substitution(self):
        pred = [
            comp(0, "capacitor", ["A", "B"]),
            comp(1, "resistor", ["A", "B"]),
        ]
        # one node relabel (resistor -> capacitor) = 1 edit
        assert graph_edit_distance(pred, GT) == 1.0

    def test_both_empty(self):
        assert normalized_ged([], []) == 0.0


class TestCoverageStats:
    def test_counts_and_rates(self):
        comps = [
            {"id": 0, "class": "resistor", "nodes": [1, 2]},
            {"id": 1, "class": "capacitor", "nodes": [1, None]},
            {"id": 2, "class": "ground", "nodes": [2, 2]},
        ]
        s = coverage_stats(comps)
        assert s["num_components"] == 3
        # 5 of 6 terminals snapped
        assert s["terminal_snap_rate"] == pytest.approx(5 / 6)
        # 2 of 3 components fully connected
        assert s["fully_connected_rate"] == pytest.approx(2 / 3)
