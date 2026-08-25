#!/usr/bin/env python3
"""Per-repeat paired comparison for the frontier model's variant-B run.

WHY THIS EXISTS. tab:vlm reported Claude's variant-B strict success as 0.5295
-- the mean over three repeats -- beside a paired delta of -0.0208 and p=0.672
taken from the MAJORITY VOTE of those same three repeats (106/192 = 0.5521).
Both numbers were correct and they described different systems, so the row did
not subtract: 0.5312 - 0.5295 is +0.0017, not -0.0208. Nothing caught it
because the frontier-model quantities were never in the numbers registry.

A majority vote of three queries is a three-times-more-expensive ensemble, not
the thing a user invokes, and it beats every individual repeat (0.5417, 0.5365,
0.5104). Reporting it as "Claude Opus 5" overstates a single query, and it also
contradicts this paper's own emphasis on per-query non-determinism.

So the comparison is done per repeat and reported as a range. A McNemar test
needs paired per-circuit outcomes and cannot be run against a mean, which is
the reason the majority vote was reached for in the first place; running it
three times and reporting the span answers the same question without inventing
a system that was never queried.

Usage:
    python scripts/vlm_repeat_significance.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOT = 10000          # matches the resample count stated in the protocol


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_ci(deltas: list[int], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    draws = []
    for _ in range(BOOT):
        draws.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline", default="results/final/benchmark/seed0/per_image.csv")
    ap.add_argument("--vlm", default="results/vlm/claude_b_test/scored/per_image.csv")
    ap.add_argument("--out", default="results/final/vlm_repeat_significance")
    a = ap.parse_args()

    pipe = {r["image"]: r["strict_success"] in ("True", "true", "1")
            for r in csv.DictReader((ROOT / a.pipeline).open())}
    by: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    for r in csv.DictReader((ROOT / a.vlm).open()):
        by[r["rep"]][r["image"]] = r["strict_success"] == "True"

    reps = sorted(by)
    per = {}
    for rep in reps:
        v = by[rep]
        imgs = sorted(set(pipe) & set(v))
        both = sum(pipe[i] and v[i] for i in imgs)
        b = sum(pipe[i] and not v[i] for i in imgs)
        c = sum(v[i] and not pipe[i] for i in imgs)
        lo, hi = paired_ci([int(pipe[i]) - int(v[i]) for i in imgs])
        per[rep] = {
            "vlm_strict_success": round(sum(v[i] for i in imgs) / len(imgs), 6),
            "both": both, "pipeline_only": b, "vlm_only": c,
            "neither": len(imgs) - both - b - c,
            "paired_delta": round(sum(int(pipe[i]) - int(v[i]) for i in imgs) / len(imgs), 6),
            "paired_delta_ci95": [round(lo, 6), round(hi, 6)],
            "mcnemar_p_exact": round(mcnemar_exact(b, c), 6),
        }

    imgs = sorted(set(pipe) & set(by[reps[0]]))
    vals = [per[r]["vlm_strict_success"] for r in reps]
    mean = sum(vals) / len(vals)
    ours = sum(pipe[i] for i in imgs) / len(imgs)
    ps = [per[r]["mcnemar_p_exact"] for r in reps]

    # what the manuscript used to quote, kept so the two can be told apart
    maj = {i: sum(by[r][i] for r in reps) >= 2 for i in imgs}
    mb = sum(pipe[i] and not maj[i] for i in imgs)
    mc = sum(maj[i] and not pipe[i] for i in imgs)

    summary = {
        "what_this_is": "per-repeat paired comparison, pipeline vs frontier model, variant B",
        "n_circuits": len(imgs),
        "n_repeats": len(reps),
        "pipeline_strict_success": round(ours, 6),
        "per_repeat": per,
        "single_query": {
            "mean_strict_success": round(mean, 6),
            "delta_ours_minus_theirs": round(ours - mean, 6),
            "mcnemar_p_min": min(ps),
            "mcnemar_p_max": max(ps),
            "ci_widest": [min(per[r]["paired_delta_ci95"][0] for r in reps),
                          max(per[r]["paired_delta_ci95"][1] for r in reps)],
            "every_ci_contains_zero": all(
                per[r]["paired_delta_ci95"][0] <= 0 <= per[r]["paired_delta_ci95"][1]
                for r in reps),
            "any_repeat_distinguishable_at_05": any(p < 0.05 for p in ps),
        },
        "majority_vote_NOT_REPORTED": {
            "strict_success": round(sum(maj.values()) / len(imgs), 6),
            "paired_delta": round(ours - sum(maj.values()) / len(imgs), 6),
            "mcnemar_p_exact": round(mcnemar_exact(mb, mc), 6),
            "why_not": ("an ensemble of three queries, costing 3x and scoring "
                        "above every individual repeat; recorded here only "
                        "because tab:vlm quoted its delta and p"),
        },
    }
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"pipeline {ours:.4f}   model mean {mean:.4f} "
          f"(delta {ours - mean:+.4f})")
    for rep in reps:
        d = per[rep]
        print(f"  {rep}: {d['vlm_strict_success']:.4f}  delta {d['paired_delta']:+.4f}  "
              f"CI [{d['paired_delta_ci95'][0]:+.4f}, {d['paired_delta_ci95'][1]:+.4f}]  "
              f"p {d['mcnemar_p_exact']:.4f}")
    print(f"  p range {min(ps):.4f}-{max(ps):.4f}; "
          f"any repeat distinguishable: "
          f"{summary['single_query']['any_repeat_distinguishable_at_05']}")
    print(f"  -> {a.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
