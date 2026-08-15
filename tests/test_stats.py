"""Unit tests for stats/. These back every significance claim in the paper.

Each test pins a value that can be checked by hand or against a textbook case,
because a statistics helper that is quietly wrong produces confident wrong
claims rather than crashes.
"""

from __future__ import annotations

import sys
from math import isclose
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stats.bootstrap import (bootstrap_mean, bootstrap_paired_delta,  # noqa: E402
                             bootstrap_rate)
from stats.holm import holm  # noqa: E402
from stats.kappa import cohens_kappa  # noqa: E402
from stats.mcnemar import mcnemar_exact  # noqa: E402


# ------------------------------------------------------------- McNemar

def test_mcnemar_cell_counts():
    a = [True, True, False, False]
    b = [True, False, True, False]
    r = mcnemar_exact(a, b)
    assert (r.n_both, r.n_only_a, r.n_only_b, r.n_neither) == (1, 1, 1, 1)


def test_mcnemar_no_discordant_pairs_is_p_one():
    """Identical decisions everywhere is no evidence of a difference."""
    a = [True, False, True]
    r = mcnemar_exact(a, list(a))
    assert r.n_discordant == 0
    assert r.p_value == 1.0


def test_mcnemar_exact_matches_hand_computation():
    """b=5, c=0, n=5 -> two-sided p = 2 * (1/2)^5 = 0.0625."""
    a = [True] * 5 + [True] * 3
    b = [False] * 5 + [True] * 3
    r = mcnemar_exact(a, b)
    assert (r.n_only_a, r.n_only_b) == (5, 0)
    assert isclose(r.p_value, 2 * (0.5 ** 5), rel_tol=1e-12)


def test_mcnemar_symmetric_split_is_not_significant():
    a = [True] * 10 + [False] * 10
    b = [False] * 10 + [True] * 10
    r = mcnemar_exact(a, b)
    assert r.p_value == 1.0


def test_mcnemar_rejects_unpaired_input():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


# ----------------------------------------------------------- bootstrap

def test_bootstrap_mean_brackets_the_point_estimate():
    vals = [0.0, 1.0] * 50
    ci = bootstrap_mean(vals, resamples=2000, seed=0)
    assert isclose(ci.point, 0.5, abs_tol=1e-12)
    assert ci.lo < ci.point < ci.hi


def test_bootstrap_is_deterministic_given_a_seed():
    v = [0.1, 0.9, 0.4, 0.7, 0.2]
    a = bootstrap_mean(v, resamples=1000, seed=7)
    b = bootstrap_mean(v, resamples=1000, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_bootstrap_constant_input_has_zero_width():
    ci = bootstrap_mean([0.42] * 30, resamples=500, seed=0)
    assert isclose(ci.lo, 0.42) and isclose(ci.hi, 0.42)


def test_paired_delta_is_tighter_than_independent_when_correlated():
    """The whole reason to pair: shared circuit difficulty cancels."""
    import numpy as np
    rng = np.random.default_rng(0)
    difficulty = rng.normal(0, 1, 200)
    a = difficulty + 0.10
    b = difficulty
    paired = bootstrap_paired_delta(a, b, resamples=2000, seed=1)
    width_paired = paired.hi - paired.lo
    ci_a = bootstrap_mean(a, resamples=2000, seed=1)
    ci_b = bootstrap_mean(b, resamples=2000, seed=1)
    width_naive = (ci_a.hi - ci_a.lo) + (ci_b.hi - ci_b.lo)
    assert width_paired < width_naive
    assert paired.excludes_zero


def test_paired_delta_of_identical_arms_is_zero():
    v = [0.3, 0.6, 0.9]
    d = bootstrap_paired_delta(v, v, resamples=500, seed=0)
    assert d.point == 0.0 and d.lo == 0.0 and d.hi == 0.0


def test_bootstrap_rate_matches_the_proportion():
    ci = bootstrap_rate([True] * 30 + [False] * 70, resamples=1000, seed=0)
    assert isclose(ci.point, 0.30, abs_tol=1e-12)


# ---------------------------------------------------------------- Holm

def test_holm_adjusts_by_descending_rank():
    res = {r.label: r for r in holm({"a": 0.01, "b": 0.02, "c": 0.03})}
    assert isclose(res["a"].p_adjusted, 0.03)   # 3 * 0.01
    assert isclose(res["b"].p_adjusted, 0.04)   # 2 * 0.02
    assert isclose(res["c"].p_adjusted, 0.04)   # 1 * 0.03, raised to be monotone


def test_holm_is_monotone_non_decreasing():
    out = holm({"a": 0.001, "b": 0.04, "c": 0.9, "d": 0.02})
    adj = [r.p_adjusted for r in out]
    assert adj == sorted(adj)


def test_holm_step_down_stops_at_first_failure():
    """Sorted order is a(.001), c(.002), b(.9); the family stops at b."""
    out = {r.label: r for r in holm({"a": 0.001, "b": 0.9, "c": 0.002})}
    assert out["a"].rejected
    assert out["c"].rejected            # reached before the failure
    assert not out["b"].rejected        # 1 * 0.9 = 0.9 > alpha, stop here
    assert isclose(out["a"].p_adjusted, 0.003)
    assert isclose(out["c"].p_adjusted, 0.004)


def test_holm_is_more_powerful_than_bonferroni():
    """Holm's FIRST step equals Bonferroni; its gain is on later hypotheses.

    m=2, alpha=0.05. Plain Bonferroni tests both against 0.025 and cannot
    reject b at 0.03. Holm tests the second against alpha/1 = 0.05 and does.
    """
    out = {r.label: r for r in holm({"a": 0.01, "b": 0.03}, alpha=0.05)}
    assert out["a"].rejected
    assert out["b"].rejected                       # Bonferroni would not
    assert isclose(out["b"].p_adjusted, 0.03)
    assert 0.03 > 0.05 / 2                         # the Bonferroni threshold


def test_holm_empty_family():
    assert holm({}) == []


# --------------------------------------------------------------- kappa

def test_kappa_perfect_agreement():
    a = ["junction", "crossing", "none", "junction"]
    r = cohens_kappa(a, list(a))
    assert isclose(r.kappa, 1.0)


def test_kappa_chance_agreement_is_near_zero():
    """High raw agreement on a skewed label set must not look like skill."""
    a = ["junction"] * 90 + ["crossing"] * 10
    b = ["junction"] * 90 + ["junction"] * 10
    r = cohens_kappa(a, b)
    assert r.p_observed == 0.90
    assert r.kappa < 0.05          # 90% raw agreement, essentially no skill


def test_kappa_degenerate_single_category():
    r = cohens_kappa(["junction"] * 5, ["junction"] * 5)
    assert r.kappa == 1.0 and r.p_expected == 1.0


def test_kappa_worse_than_chance_is_negative():
    a = ["x", "x", "y", "y"]
    b = ["y", "y", "x", "x"]
    assert cohens_kappa(a, b).kappa < 0


def test_kappa_rejects_unpaired_input():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])
