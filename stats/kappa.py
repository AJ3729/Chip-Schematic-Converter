"""Cohen's kappa for categorical agreement between two raters.

Used for intersection adjudications (junction / crossing / edge_group / none),
where raw percent agreement is misleading: roughly 83% of sites in the existing
ground truth are junctions, so a rater who answered "junction" every time would
score about 0.83 without looking at the page. Kappa corrects for that expected
agreement.

Reported for the second-annotator study and for the self-agreement measure on
double-annotated circuits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KappaResult:
    kappa: float
    p_observed: float
    p_expected: float
    n: int
    categories: list[str]
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    def interpret(self) -> str:
        """Landis and Koch's conventional bands, named as conventions.

        These are a reporting convention, not a statistical threshold, and the
        manuscript should say so where the number appears.
        """
        k = self.kappa
        if k < 0.0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"


def cohens_kappa(rater_a: list[str], rater_b: list[str]) -> KappaResult:
    """Unweighted Cohen's kappa on aligned categorical labels."""
    if len(rater_a) != len(rater_b):
        raise ValueError(f"unpaired inputs: {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        raise ValueError("no observations")

    cats = sorted(set(rater_a) | set(rater_b))
    confusion = Counter(zip(rater_a, rater_b))
    agree = sum(confusion[(c, c)] for c in cats)
    p_o = agree / n

    ca, cb = Counter(rater_a), Counter(rater_b)
    p_e = sum((ca[c] / n) * (cb[c] / n) for c in cats)

    if p_e == 1.0:
        # Both raters used a single identical category throughout. Kappa is
        # undefined (0/0); report 1.0 with p_e exposed so a reader can see the
        # degenerate case rather than being handed a bare number.
        k = 1.0
    else:
        k = (p_o - p_e) / (1.0 - p_e)

    return KappaResult(kappa=k, p_observed=p_o, p_expected=p_e, n=n,
                       categories=cats, confusion=dict(confusion))
