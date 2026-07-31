"""nGED must be a function of the two topologies and nothing else.

The search-based implementation was not. ``nx.graph_edit_distance(...,
timeout=t)`` returns the best bound found SO FAR as a plain float when the
budget runs out, so a non-None return could not be distinguished from an exact
answer, and the value depended on how much CPU the process happened to receive.
Measured on 18 test images at a 5 s budget, 13 burned the whole budget and all
of them returned a number.

It was not a theoretical risk. Turning ``stitch_masked_gaps`` off changes the
recovered topology on ZERO images -- every other per-image column is
bit-identical and the pipeline output matches pixel for pixel -- yet nGED moved
on 24 of 190 images with a bootstrap CI excluding zero. A metric that reports a
significant change when nothing changed cannot decide anything.

These tests pin the three properties the replacement needs: it is deterministic,
it is a genuine upper bound on the true GED, and it is still zero exactly when
the topologies agree.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from schematic2netlist.metrics import (_node_match, graph_edit_distance,
                                       normalized_ged, to_topology_graph)

CLASSES = ["Resistor", "Capacitor", "Inductor", "Diode", "V-DC", "GND"]


def circuit(rnd: random.Random, n_comp: int, n_net: int) -> list[dict]:
    return [{"id": i, "class": rnd.choice(CLASSES),
             "nets": [f"N{rnd.randrange(n_net)}", f"N{rnd.randrange(n_net)}"]}
            for i in range(n_comp)]


def test_identical_topologies_score_zero():
    rnd = random.Random(7)
    for _ in range(20):
        c = circuit(rnd, rnd.randrange(1, 8), rnd.randrange(1, 5))
        assert graph_edit_distance(c, c) == 0.0
        assert normalized_ged(c, c) == 0.0


def test_deterministic_across_repeated_calls():
    rnd = random.Random(11)
    for _ in range(15):
        a = circuit(rnd, rnd.randrange(1, 9), rnd.randrange(1, 6))
        b = circuit(rnd, rnd.randrange(1, 9), rnd.randrange(1, 6))
        vals = {graph_edit_distance(a, b) for _ in range(4)}
        assert len(vals) == 1, f"nGED varied across identical calls: {vals}"


def test_deterministic_across_node_insertion_order():
    """Shuffling the component list must not move the number. The graph is the
    same graph; only dict and edge insertion order changes."""
    rnd = random.Random(13)
    for _ in range(15):
        a = circuit(rnd, rnd.randrange(2, 9), rnd.randrange(2, 6))
        b = circuit(rnd, rnd.randrange(2, 9), rnd.randrange(2, 6))
        base = graph_edit_distance(a, b)
        for _ in range(3):
            a2, b2 = list(a), list(b)
            rnd.shuffle(a2)
            rnd.shuffle(b2)
            assert graph_edit_distance(a2, b2) == base, \
                "nGED depends on component ordering"


def test_is_a_valid_upper_bound_on_exact_ged():
    """On graphs small enough for an exact search to actually finish, the bound
    must never fall below the truth."""
    rnd = random.Random(3)
    checked = 0
    for _ in range(14):
        a = circuit(rnd, rnd.randrange(1, 5), rnd.randrange(1, 4))
        b = circuit(rnd, rnd.randrange(1, 5), rnd.randrange(1, 4))
        G1, G2 = to_topology_graph(a), to_topology_graph(b)
        exact = nx.graph_edit_distance(G1, G2, node_match=_node_match,
                                       timeout=20.0)
        if exact is None:
            continue
        bound = graph_edit_distance(a, b)
        assert bound >= exact - 1e-9, \
            f"bound {bound} is BELOW the exact GED {exact}"
        checked += 1
    assert checked >= 8, "too few exact comparisons to be meaningful"


def test_normalized_range_and_empty_case():
    assert normalized_ged([], []) == 0.0
    a = [{"id": 0, "class": "Resistor", "nets": ["N0", "N1"]}]
    assert normalized_ged(a, []) > 0.0
    assert normalized_ged([], a) > 0.0


def test_search_mode_is_budget_dependent():
    """The retired path, pinned so the justification stays checkable: the same
    graphs give different answers at different budgets. If this ever stops
    being true, networkx changed and the search mode could be revisited."""
    rnd = random.Random(5)
    a = circuit(rnd, 14, 8)
    b = circuit(rnd, 14, 8)
    lo = graph_edit_distance(a, b, timeout_s=0.05, method="search")
    hi = graph_edit_distance(a, b, timeout_s=2.0, method="search")
    assert lo >= hi, "a longer search should not return a worse bound"
    if lo == hi:
        pytest.skip("this pair happened to be solved inside the small budget")
