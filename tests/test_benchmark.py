"""Unit tests for benchmark alignment + scoring on toy graphs.

Pins the fairness properties: IoU component alignment, terminal-order
canonicalization (independent pred/GT indexing must not penalize a
correct circuit), and unmatched components penalizing rather than being
dropped.
"""

import pytest

from schematic2netlist.benchmark import (
    align_components,
    bootstrap_ci,
    canonicalize_terminals,
    iou_center,
    score_prediction,
)


def c(i, cls, nets, bbox):
    return {"id": i, "class": cls, "nets": nets, "bbox": bbox}


# A divider: source, two resistors, ground.
GT = [
    c(0, "V-DC", ["n1", "0"], [10, 10, 20, 20]),
    c(1, "Resistor", ["n1", "n2"], [50, 10, 20, 8]),
    c(2, "Resistor", ["n2", "0"], [50, 40, 20, 8]),
    c(3, "GND", ["0"], [10, 60, 12, 8]),
]


class TestIoU:
    def test_identical_boxes(self):
        assert iou_center([10, 10, 20, 20], [10, 10, 20, 20]) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert iou_center([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


class TestAlignment:
    def test_perfect_alignment_relabels_to_gt_ids(self):
        # same circuit, but predicted component ids are shuffled
        pred = [
            c(7, "Resistor", ["a", "b"], [50, 10, 20, 8]),
            c(8, "V-DC", ["a", "g"], [10, 10, 20, 20]),
            c(9, "Resistor", ["b", "g"], [50, 40, 20, 8]),
            c(5, "GND", ["g"], [10, 60, 12, 8]),
        ]
        pred_a, _, stats = align_components(pred, GT)
        assert stats["matched"] == 4
        assert stats["unmatched_gt"] == 0
        # the predicted resistor at (50,10) must take GT id 1
        r = next(x for x in pred_a if x["bbox"] == [50, 10, 20, 8])
        assert r["id"] == 1

    def test_unmatched_pred_gets_disjoint_id(self):
        pred = GT + [c(99, "Capacitor", ["x", "y"], [200, 200, 10, 10])]
        pred_a, _, stats = align_components(pred, GT)
        assert stats["unmatched_pred"] == 1
        extra = next(x for x in pred_a if x["bbox"] == [200, 200, 10, 10])
        assert extra["id"] >= 1000            # disjoint from GT ids

    def test_wrong_class_does_not_match(self):
        pred = [c(0, "Capacitor", ["n1", "0"], [10, 10, 20, 20])]  # same box, wrong class
        _, _, stats = align_components(pred, GT)
        assert stats["matched"] == 0


class TestScoring:
    def test_identical_circuit_scores_perfect(self):
        s = score_prediction([dict(x) for x in GT], GT)
        assert s["net_f1"] == 1.0
        assert s["terminal_pair_f1"] == 1.0
        assert s["strict_success"] is True
        assert s["nged"] == 0.0

    def test_flipped_2terminal_order_still_perfect(self):
        # predicted resistor 1 has its terminals in the opposite index
        # order — a correct circuit must NOT be penalized
        pred = [dict(x) for x in GT]
        pred[1] = c(1, "Resistor", ["n2", "n1"], [50, 10, 20, 8])
        s = score_prediction(pred, GT)
        assert s["net_f1"] == 1.0
        assert s["strict_success"] is True

    def test_missing_component_penalizes(self):
        pred = [dict(x) for x in GT[:3]]      # drop the ground
        s = score_prediction(pred, GT)
        assert s["unmatched_gt"] == 1
        assert s["strict_success"] is False

    def test_wrong_connection_lowers_net_f1(self):
        pred = [dict(x) for x in GT]
        # put resistor 2's bottom on a spurious net instead of ground
        pred[2] = c(2, "Resistor", ["n2", "n9"], [50, 40, 20, 8])
        s = score_prediction(pred, GT)
        assert s["net_f1"] < 1.0
        assert s["strict_success"] is False


class TestCanonicalize:
    def test_distinct_partners_give_deterministic_order(self):
        # R1's two terminals have DISTINCT partner sets (cap on one net,
        # inductor on the other), so canonicalization is deterministic
        # regardless of the input index order.
        base = [
            c(1, "Capacitor", ["na", "z"], [0, 0, 1, 1]),
            c(2, "Inductor", ["nb", "z"], [0, 0, 1, 1]),
        ]
        forward = canonicalize_terminals(
            [c(0, "Resistor", ["na", "nb"], [0, 0, 1, 1])] + base
        )
        flipped = canonicalize_terminals(
            [c(0, "Resistor", ["nb", "na"], [0, 0, 1, 1])] + base
        )
        r_fwd = next(x for x in forward if x["id"] == 0)
        r_flp = next(x for x in flipped if x["id"] == 0)
        assert r_fwd["nets"] == r_flp["nets"]     # same order despite flip


class TestBootstrap:
    def test_all_ones_ci_is_one(self):
        mean, lo, hi = bootstrap_ci([1.0] * 20)
        assert mean == 1.0 and lo == 1.0 and hi == 1.0

    def test_ci_brackets_mean(self):
        vals = [0.0, 1.0] * 25
        mean, lo, hi = bootstrap_ci(vals, seed=0)
        assert lo <= mean <= hi
        assert 0.3 < mean < 0.7
