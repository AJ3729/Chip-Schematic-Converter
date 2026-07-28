"""Repair-evaluation harness (C5 MSP): aggregation, topology proof,
ground gauge accuracy — on toy inputs with known answers."""

import csv
import json
from types import SimpleNamespace

import pytest

from schematic2netlist.repair_eval import (
    aggregate_repair,
    check_topology_preserved,
    ground_choice_accuracy,
)


# ---------------------------------------------------------------- topology

def _comps(nets_per_comp):
    return [
        {"id": i, "node_names": list(nets)} for i, nets in enumerate(nets_per_comp)
    ]


def test_topology_preserved_clean():
    comps = _comps([["n1", "0"], ["n1", "n2"]])
    before = [list(c["node_names"]) for c in comps]
    rep = SimpleNamespace(extra_lines=["Rref n2 0 0", "Rshunt_n1 n1 0 1e+09"])
    assert check_topology_preserved(comps, rep, before) == []


def test_topology_violation_net_change():
    comps = _comps([["n1", "0"]])
    before = [["n1", "n9"]]  # snapshot disagrees -> repair mutated nets
    assert check_topology_preserved(comps, None, before)


def test_topology_violation_rogue_line():
    comps = _comps([["n1", "0"]])
    before = [list(c["node_names"]) for c in comps]
    rep = SimpleNamespace(extra_lines=["V99 n1 0 5"])  # injected element!
    violations = check_topology_preserved(comps, rep, before)
    assert violations and "non-allowlisted" in violations[0]


def test_topology_no_repair_result():
    comps = _comps([["n1", "0"]])
    before = [list(c["node_names"]) for c in comps]
    assert check_topology_preserved(comps, None, before) == []


# ---------------------------------------------------------------- aggregate

@pytest.fixture()
def run_dir(tmp_path):
    rows = [
        {"image": "a.jpg", "solvable_before": 0, "solvable_after": 1,
         "num_assumptions": 3, "num_gauge": 1},
        {"image": "b.jpg", "solvable_before": 1, "solvable_after": 1,
         "num_assumptions": 1, "num_gauge": 2},
        {"image": "c.jpg", "solvable_before": 0, "solvable_after": 0,
         "num_assumptions": 9, "num_gauge": 0},
    ]
    with (tmp_path / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ledgers = tmp_path / "ledgers"
    ledgers.mkdir()
    for stem, entries in [
        ("a", [("no_dc_path_to_ground", "assumption"),
               ("ground_selection", "gauge")]),
        ("b", [("placeholder_values", "assumption")]),
    ]:
        (ledgers / f"{stem}.json").write_text(json.dumps({
            "entries": [{"issue": i, "category": c} for i, c in entries]
        }))
    return tmp_path


def test_aggregate_rates_and_lift(run_dir):
    s = aggregate_repair(run_dir, max_assumptions=8)
    assert s["n_images"] == 3
    assert s["solvable_before_rate"] == pytest.approx(1 / 3)
    assert s["solvable_after_rate"] == pytest.approx(2 / 3)
    assert s["solvability_lift"] == pytest.approx(1 / 3)
    assert s["lift_ci95_lo"] <= s["solvability_lift"] <= s["lift_ci95_hi"]
    assert s["regressed_images"] == 0
    assert s["budget_violations"] == 1  # c.jpg: 9 > 8
    assert s["per_issue"]["no_dc_path_to_ground|assumption"] == 1
    assert s["entries_by_category"] == {"assumption": 2, "gauge": 1}


# ---------------------------------------------------------------- ground

def _bench_comps(spec):
    """spec: list of (id, class, nets, bbox-center-x) — unit boxes."""
    return [
        {"id": i, "class": cls, "nets": list(nets),
         "bbox": [float(x), 0.0, 10.0, 10.0]}
        for i, cls, nets, x in spec
    ]


def test_ground_choice_correct():
    gt = _bench_comps([
        (0, "Resistor", ["n1", "0"], 0),
        (1, "V-DC", ["n1", "0"], 20),
    ])
    pred = _bench_comps([
        (5, "Resistor", ["a", "0"], 0),
        (7, "V-DC", ["a", "0"], 20),
    ])
    out = ground_choice_accuracy(pred, gt)
    assert out == {"mapped_gt_net": "0", "correct": True}


def test_ground_choice_wrong_net():
    # An asymmetric circuit: only the resistor bridges to the third net,
    # so pred's "0" unambiguously maps onto GT's n1, not GT's 0.
    gt = _bench_comps([
        (0, "Resistor", ["n1", "n2"], 0),
        (1, "V-DC", ["n1", "0"], 20),
    ])
    pred = _bench_comps([
        (5, "Resistor", ["0", "x"], 0),
        (7, "V-DC", ["0", "y"], 20),
    ])
    out = ground_choice_accuracy(pred, gt)
    assert out["mapped_gt_net"] == "n1"
    assert out["correct"] is False


def test_ground_choice_ambiguous_is_not_credited():
    # pred ties two components to its reference net, but GT puts those
    # two terminals on different nets — no GT net wins, so the harness
    # must report ambiguity rather than crediting a coin flip.
    gt = _bench_comps([
        (0, "Resistor", ["a", "b"], 0),
        (1, "Capacitor", ["c", "d"], 20),
    ])
    pred = _bench_comps([
        (5, "Resistor", ["0", "p"], 0),
        (7, "Capacitor", ["0", "q"], 20),
    ])
    out = ground_choice_accuracy(pred, gt)
    assert out["correct"] is False
    assert out["ambiguous"] is True


def test_ground_choice_unanswerable():
    gt = _bench_comps([(0, "Resistor", ["n1", "n2"], 0)])
    pred = _bench_comps([(5, "Resistor", ["a", "b"], 0)])
    assert ground_choice_accuracy(pred, gt) is None
