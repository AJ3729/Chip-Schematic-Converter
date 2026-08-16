"""The three controls the plan requires for the pin-aware scorer (D2).

Mirrors the existing operating-point protocol:
  1. self comparison returns exactly 1.0
  2. an asymmetric swap is detected
  3. a passive swap is NEVER detected -- exactly 0 of them

If any of these fails the metric is not trustworthy and the plan says to halt,
so they are tests rather than a report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.pin_aware import (load_symmetry, permutations_for,  # noqa: E402
                               score_pin_aware, swap_is_detectable)

SYM = load_symmetry()

ASYMMETRIC = ["BJT-NPN", "BJT-PNP", "MOSFET-N", "MOSFET-P", "Op-Amp",
              "Diode", "Zener Diode", "V-DC", "V-AC", "I-DC", "I-AC"]
SYMMETRIC = ["Resistor", "Capacitor", "Inductor"]


def comp(i, cls, nets):
    return {"id": i, "class": cls, "nets": list(nets)}


def circuit():
    return [comp(0, "BJT-NPN", ["nc", "nb", "ne"]),
            comp(1, "Resistor", ["nc", "vcc"]),
            comp(2, "Diode", ["ne", "gnd"]),
            comp(3, "Op-Amp", ["nb", "nc", "vout"])]


def matched(cs):
    return [(c["id"], c["id"]) for c in cs]


# ---------------------------------------------------- control 1: self compare

def test_self_comparison_is_exactly_one():
    c = circuit()
    r = score_pin_aware(c, c, matched(c))
    assert r.n_correct == r.n_components == 4
    assert r.component_accuracy == 1.0
    assert r.strict_success is True


def test_self_comparison_survives_net_renaming():
    """Net names are arbitrary; only the grouping is meaningful."""
    ref = circuit()
    ren = {"nc": "z1", "nb": "z2", "ne": "z3", "vcc": "z4",
           "gnd": "z5", "vout": "z6"}
    pred = [comp(c["id"], c["class"], [ren[n] for n in c["nets"]]) for c in ref]
    r = score_pin_aware(pred, ref, matched(ref))
    assert r.component_accuracy == 1.0 and r.strict_success


# ------------------------------------------- control 2: asymmetric detection

@pytest.mark.parametrize("cls", ASYMMETRIC)
def test_every_asymmetric_class_declares_a_swap_detectable(cls):
    assert swap_is_detectable(cls, 0, 1, SYM), f"{cls} would forgive a swap"


def test_bjt_collector_emitter_swap_is_caught():
    ref = circuit()
    pred = [dict(c) for c in ref]
    pred[0] = comp(0, "BJT-NPN", ["ne", "nb", "nc"])   # C and E exchanged
    r = score_pin_aware(pred, ref, matched(ref))
    assert r.n_correct == 3
    assert r.strict_success is False
    assert r.per_class["BJT-NPN"]["accuracy"] == 0.0


def test_opamp_input_swap_is_caught():
    ref = circuit()
    pred = [dict(c) for c in ref]
    pred[3] = comp(3, "Op-Amp", ["nc", "nb", "vout"])  # In+ / In- exchanged
    r = score_pin_aware(pred, ref, matched(ref))
    assert r.strict_success is False


def test_mosfet_is_asymmetric_per_the_authored_decision():
    """The author ruled drain/source asymmetric on 2026-08-15."""
    assert SYM["MOSFET-N"]["group"] == []
    assert SYM["MOSFET-P"]["group"] == []
    assert swap_is_detectable("MOSFET-N", 0, 2, SYM)


# --------------------------------------------- control 3: the passive control

@pytest.mark.parametrize("cls", SYMMETRIC)
def test_passive_swap_is_never_detected(cls):
    assert not swap_is_detectable(cls, 0, 1, SYM), \
        f"{cls} terminal order carries no meaning and must not be scored"


def test_reversing_every_passive_costs_nothing():
    """The plan's passive control, in miniature: exactly 0 detections."""
    ref = [comp(0, "Resistor", ["a", "b"]),
           comp(1, "Capacitor", ["b", "c"]),
           comp(2, "Inductor", ["c", "d"])]
    pred = [comp(c["id"], c["class"], list(reversed(c["nets"]))) for c in ref]
    r = score_pin_aware(pred, ref, matched(ref))
    assert r.n_correct == 3
    assert r.component_accuracy == 1.0
    assert r.strict_success is True


# ------------------------------------------------------------- other properties

def test_pin_aware_is_strictly_harder_than_pin_blind():
    """Anything pin-aware accepts, a pin-blind metric accepts too."""
    ref = circuit()
    pred = [dict(c) for c in ref]
    pred[0] = comp(0, "BJT-NPN", ["ne", "nb", "nc"])
    r = score_pin_aware(pred, ref, matched(ref))
    # the net PARTITION is unchanged by a swap -- that is exactly why the
    # published metric cannot see it
    assert sorted(map(sorted, [c["nets"] for c in pred])) == \
           sorted(map(sorted, [c["nets"] for c in ref]))
    assert r.strict_success is False


def test_unmatched_reference_component_fails_strict():
    ref = circuit()
    pred = [dict(c) for c in ref[:3]]
    r = score_pin_aware(pred, ref, matched(pred))
    assert r.n_components == 4 and r.n_correct == 3
    assert r.strict_success is False


def test_terminal_count_mismatch_is_a_failure_not_a_skip():
    ref = [comp(0, "BJT-NPN", ["a", "b", "c"])]
    pred = [comp(0, "BJT-NPN", ["a", "b"])]
    r = score_pin_aware(pred, ref, matched(ref))
    assert r.n_scored == 1 and r.n_correct == 0
    assert r.errors[0]["why"] == "terminal count differs"


def test_unknown_class_defaults_to_asymmetric():
    """Silently forgiving a swap is the failure this module prevents."""
    assert permutations_for("Nonexistent-Class", 3, SYM) == [(0, 1, 2)]
    assert swap_is_detectable("Nonexistent-Class", 0, 1, SYM)


def test_spec_covers_every_pipeline_class():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from schematic2netlist.classes import canonical_classes
    assert not set(canonical_classes()) - set(SYM)


def test_invented_component_fails_strict():
    """A false positive must fail, or this metric could accept a circuit the
    published (easier) metric rejects -- which would be incoherent."""
    ref = circuit()
    pred = [dict(c) for c in ref] + [comp(99, "Resistor", ["vcc", "gnd"])]
    r = score_pin_aware(pred, ref, matched(ref))
    assert r.n_correct == 4
    assert r.n_pred_unmatched == 1
    assert r.strict_success is False
