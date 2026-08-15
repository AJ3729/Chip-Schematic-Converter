"""Exact binomial McNemar test for paired binary outcomes.

Use this, not a chi-square approximation, when comparing two systems on the
SAME circuits with a binary outcome (strict success). The comparison is paired
-- the same 192 drawings go to both systems -- and the discordant counts here
are small enough that the asymptotic form is not trustworthy.

The test conditions on the discordant pairs only: of the n = b + c circuits
where exactly one system succeeded, is the split consistent with a fair coin?
Concordant pairs carry no information about which system is better and are
correctly ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class McNemarResult:
    n_both: int          # both systems correct
    n_only_a: int        # a correct, b wrong          (the "b" cell)
    n_only_b: int        # b correct, a wrong          (the "c" cell)
    n_neither: int       # both wrong
    p_value: float       # two-sided exact binomial
    odds_ratio: float | None   # n_only_a / n_only_b, None if denominator is 0

    @property
    def n_discordant(self) -> int:
        return self.n_only_a + self.n_only_b

    def describe(self, name_a: str = "A", name_b: str = "B",
                 alpha: float = 0.05) -> str:
        verdict = ("distinguishable" if self.p_value < alpha
                   else "NOT distinguishable")
        return (f"{name_a} vs {name_b}: {self.n_only_a} circuits only {name_a} "
                f"solves, {self.n_only_b} only {name_b}, "
                f"{self.n_both} both, {self.n_neither} neither. "
                f"exact McNemar p = {self.p_value:.4g} -> {verdict} at "
                f"alpha={alpha}.")


def mcnemar_exact(a: list[bool], b: list[bool]) -> McNemarResult:
    """Two-sided exact binomial McNemar on paired boolean outcomes.

    ``a`` and ``b`` must be aligned: element i is the same circuit for both.
    """
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no observations")

    both = only_a = only_b = neither = 0
    for x, y in zip(a, b):
        if x and y:
            both += 1
        elif x and not y:
            only_a += 1
        elif y and not x:
            only_b += 1
        else:
            neither += 1

    n = only_a + only_b
    if n == 0:
        # No discordant pairs: the systems made identical decisions on every
        # circuit. There is no evidence of a difference, and p = 1 is the
        # honest answer rather than an error.
        p = 1.0
    else:
        # Two-sided exact binomial at q = 0.5. Because the null is symmetric,
        # doubling the smaller tail is exact rather than approximate.
        k = min(only_a, only_b)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)

    return McNemarResult(
        n_both=both, n_only_a=only_a, n_only_b=only_b, n_neither=neither,
        p_value=p,
        odds_ratio=(only_a / only_b) if only_b else None,
    )
