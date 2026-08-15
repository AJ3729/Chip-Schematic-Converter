"""Bootstrap percentile intervals, resampling over CIRCUITS.

The resampling unit matters and is easy to get wrong. Metrics like net F1 are
computed per circuit and then averaged, so the independent unit is the circuit,
not the terminal pair or the net. Resampling terminals would treat two pairs
from one drawing as independent evidence and produce an interval that is far
too narrow.

Paired deltas use the SAME resampled index set for both systems, which is what
makes the interval a statement about the difference rather than about two
independent means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    level: float
    n: int
    resamples: int

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_mean(values, level: float = 0.95,
                   resamples: int = DEFAULT_RESAMPLES,
                   seed: int = 0) -> Interval:
    """Percentile interval for the mean of a per-circuit metric."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        raise ValueError("no observations")
    rng = _rng(seed)
    idx = rng.integers(0, v.size, size=(resamples, v.size))
    means = v[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [a, 1.0 - a])
    return Interval(float(v.mean()), float(lo), float(hi), level, v.size,
                    resamples)


def bootstrap_paired_delta(a, b, level: float = 0.95,
                           resamples: int = DEFAULT_RESAMPLES,
                           seed: int = 0) -> Interval:
    """Percentile interval for mean(a) - mean(b) on PAIRED per-circuit values.

    Both arms are indexed by the same resample, so circuit-level difficulty
    cancels the way it does in the real comparison.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"unpaired inputs: {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("no observations")
    d = x - y
    rng = _rng(seed)
    idx = rng.integers(0, d.size, size=(resamples, d.size))
    means = d[idx].mean(axis=1)
    lvl = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [lvl, 1.0 - lvl])
    return Interval(float(d.mean()), float(lo), float(hi), level, d.size,
                    resamples)


def bootstrap_rate(successes, level: float = 0.95,
                   resamples: int = DEFAULT_RESAMPLES,
                   seed: int = 0) -> Interval:
    """Interval for a success RATE from per-circuit booleans."""
    return bootstrap_mean([1.0 if s else 0.0 for s in successes],
                          level=level, resamples=resamples, seed=seed)
