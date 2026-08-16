"""Coordinate-keyed site records: resolution, and the two refusals.

``compare_annotations.resolve_sites_xy`` turns the coordinates an independent
annotator writes down into the site indices the differ compares. The happy path
is covered end-to-end by ``compare_annotations.py --self-test``, which delivers
half its injected site flips by coordinate and demands identical recovery.

What that cannot cover is the REFUSALS, because the self-test never constructs an
ambiguous case. They matter more than the happy path: a coordinate silently
attached to the wrong intersection enters Cohen's kappa as a real disagreement
about ink neither annotator was looking at, and nothing downstream can detect it.
So each refusal is exercised here against a hand-built site table.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from compare_annotations import (  # noqa: E402
    SITE_XY_TOL_PX,
    resolve_sites_xy,
)


def evidence(*points: tuple[int, int]) -> dict:
    """A minimal site table: index -> position. Only x/y are read here."""
    return {i: {"x": x, "y": y, "degree": 4, "dot_score": 1.0,
                "hop_score": 1.0, "kind": "crossing"}
            for i, (x, y) in enumerate(points)}


# --------------------------------------------------------------- resolution

def test_exact_coordinate_resolves():
    ev = evidence((100, 100), (400, 400))
    got, rep = resolve_sites_xy({"sites_xy": [{"xy": [100, 100], "call": "junction"}]}, ev)
    assert got == {"0": "junction"}
    assert rep["matched"] == 1 and rep["unmatched"] == []


def test_within_tolerance_resolves():
    ev = evidence((100, 100), (400, 400))
    off = SITE_XY_TOL_PX - 1
    got, rep = resolve_sites_xy(
        {"sites_xy": [{"xy": [100 + off, 100], "call": "crossing"}]}, ev)
    assert got == {"0": "crossing"}
    assert rep["matched"] == 1


def test_edge_group_call_survives_resolution():
    """An edge grouping is a list, not a string; it must pass through intact."""
    ev = evidence((50, 50))
    grouping = [[1, 2], [3, 4]]
    got, _ = resolve_sites_xy({"sites_xy": [{"xy": [50, 50], "call": grouping}]}, ev)
    assert got == {"0": grouping}


def test_several_coordinates_resolve_independently():
    ev = evidence((100, 100), (400, 400), (700, 700))
    got, rep = resolve_sites_xy({"sites_xy": [
        {"xy": [100, 100], "call": "junction"},
        {"xy": [700, 702], "call": "crossing"},
    ]}, ev)
    assert got == {"0": "junction", "2": "crossing"}
    assert rep["matched"] == 2 and rep["unmatched"] == []


# ----------------------------------------------------------------- refusals

def test_out_of_tolerance_is_refused_not_snapped():
    """The nearest site is 40 px away. Refuse, and say how far away it was."""
    ev = evidence((100, 100))
    got, rep = resolve_sites_xy({"sites_xy": [{"xy": [140, 100], "call": "junction"}]}, ev)
    assert got == {}
    assert rep["matched"] == 0 and len(rep["unmatched"]) == 1
    assert "no traced site within" in rep["unmatched"][0]["why"]
    assert "40.0" in rep["unmatched"][0]["why"]


def test_cluster_with_conflicting_calls_is_refused():
    """Two sites within tolerance that the other annotator called differently:
    which one the coordinate names decides the answer, so refuse."""
    ev = evidence((100, 100), (100 + SITE_XY_TOL_PX - 2, 100))
    got, rep = resolve_sites_xy(
        {"sites_xy": [{"xy": [101, 100], "call": "crossing"}]}, ev,
        other_calls={"0": "junction", "1": "crossing"})
    assert got == {}
    assert "would decide the answer" in rep["unmatched"][0]["why"]


def test_cluster_with_one_shared_call_resolves_to_nearest():
    """The tracer split one drawn intersection in two and the other annotator
    called both the same way. The choice cannot change the comparison, so it is
    bookkeeping -- resolve it, and say it came from a cluster."""
    ev = evidence((100, 100), (100 + SITE_XY_TOL_PX - 2, 100))
    got, rep = resolve_sites_xy(
        {"sites_xy": [{"xy": [101, 100], "call": "junction"}]}, ev,
        other_calls={"0": "junction", "1": "junction"})
    assert got == {"0": "junction"}
    assert rep["matched"] == 1 and rep["matched_via_cluster"] == 1


def test_cluster_the_other_side_never_adjudicated_resolves():
    """Sites nobody else called cannot change the comparison either way."""
    ev = evidence((100, 100), (100 + SITE_XY_TOL_PX - 2, 100))
    got, rep = resolve_sites_xy(
        {"sites_xy": [{"xy": [101, 100], "call": "crossing"}]}, ev,
        other_calls={})
    assert got == {"0": "crossing"}
    assert rep["matched_via_cluster"] == 1


def test_unambiguous_match_is_not_counted_as_a_cluster():
    ev = evidence((100, 100), (400, 400))
    _, rep = resolve_sites_xy({"sites_xy": [{"xy": [100, 100], "call": "junction"}]},
                              ev, other_calls={"0": "junction"})
    assert rep["matched"] == 1 and rep["matched_via_cluster"] == 0


def test_two_coordinates_on_one_site_refuses_both():
    """Both calls are dropped, not arbitrarily reconciled to one of them."""
    ev = evidence((100, 100))
    got, rep = resolve_sites_xy({"sites_xy": [
        {"xy": [100, 100], "call": "junction"},
        {"xy": [103, 101], "call": "crossing"},
    ]}, ev)
    assert got == {}
    assert len(rep["unmatched"]) == 2
    assert all("cannot tell which call" in u["why"] for u in rep["unmatched"])


def test_one_bad_coordinate_does_not_poison_the_others():
    ev = evidence((100, 100), (400, 400))
    got, rep = resolve_sites_xy({"sites_xy": [
        {"xy": [100, 100], "call": "junction"},
        {"xy": [900, 900], "call": "crossing"},
    ]}, ev)
    assert got == {"0": "junction"}
    assert rep["matched"] == 1 and len(rep["unmatched"]) == 1


def test_no_tracer_evidence_refuses_everything():
    got, rep = resolve_sites_xy({"sites_xy": [{"xy": [1, 2], "call": "junction"}]}, None)
    assert got == {}
    assert rep["evidence_available"] is False
    assert rep["unmatched"][0]["why"] == "no tracer evidence"


def test_malformed_coordinate_is_reported_not_crashed():
    ev = evidence((100, 100))
    for bad in (None, [1], "100,100", [1, 2, 3]):
        got, rep = resolve_sites_xy({"sites_xy": [{"xy": bad, "call": "junction"}]}, ev)
        assert got == {}
        assert rep["unmatched"][0]["why"] == "malformed xy"


# ------------------------------------------------------------------ absence

def test_absent_record_is_silent():
    """No sites_xy key is the normal case for annotation A; emit no rows."""
    for dec in (None, {}, {"sites": {"3": "junction"}}, {"sites_xy": []}):
        got, rep = resolve_sites_xy(dec, evidence((10, 10)))
        assert got == {}
        assert rep["given"] == 0 and rep["unmatched"] == []


def test_tolerance_is_reported_so_the_number_is_never_implicit():
    _, rep = resolve_sites_xy({"sites_xy": [{"xy": [0, 0], "call": "none"}]},
                              evidence((500, 500)))
    assert rep["tolerance_px"] == SITE_XY_TOL_PX
