"""Holm-Bonferroni step-down correction for a family of hypotheses.

Every comparison table in the manuscript tests several hypotheses at once
(pipeline vs each frontier model, per class, per corpus). Reporting raw
p-values across a family inflates the chance that at least one crosses 0.05 by
accident. Holm controls the family-wise error rate, is uniformly more powerful
than plain Bonferroni, and -- unlike Benjamini-Hochberg -- requires no
independence assumption, which matters here because the same circuits appear in
every comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HolmResult:
    label: str
    p_raw: float
    p_adjusted: float
    rejected: bool


def holm(p_values: dict[str, float], alpha: float = 0.05) -> list[HolmResult]:
    """Return one result per hypothesis, ordered from smallest raw p upward.

    ``p_adjusted`` is the standard step-down adjusted p, made monotone so a
    later hypothesis can never carry a smaller adjusted value than an earlier
    one. Compare it against ``alpha`` directly.
    """
    if not p_values:
        return []
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)

    adjusted: list[float] = []
    running = 0.0
    for i, (_, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        adjusted.append(running)

    # Step-down rejection: stop at the first hypothesis that fails.
    out: list[HolmResult] = []
    still_rejecting = True
    for (label, p), adj in zip(items, adjusted):
        if still_rejecting and adj > alpha:
            still_rejecting = False
        out.append(HolmResult(label, p, adj, still_rejecting))
    return out
