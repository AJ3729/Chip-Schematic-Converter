#!/usr/bin/env python3
"""Significance for the frontier-model comparison (manuscript Table 5).

The table currently reports point estimates side by side, which invites the
reader to rank systems that may not be distinguishable. Every comparison here
is PAIRED -- the same 192 drawings go to every system -- so the correct tests
are exact McNemar on the discordant pairs and a paired bootstrap on the
per-circuit delta. Holm corrects across the family.

The specific question the plan asks: is the assisted frontier result
(0.5295 +/- 0.0167) distinguishable from the pipeline (0.5312)?

Usage:
    python scripts/table5_significance.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stats.bootstrap import bootstrap_paired_delta, bootstrap_rate  # noqa: E402
from stats.holm import holm  # noqa: E402
from stats.mcnemar import mcnemar_exact  # noqa: E402

PIPE = "results/final/benchmark/seed0/per_image.csv"
RUNS = {
    "claude_A": "results/vlm/claude_a_test/scored/per_image.csv",
    "gpt_A": "results/vlm/openai_a_test/scored/per_image.csv",
    "claude_B": "results/vlm/claude_b_test/scored/per_image.csv",
    "gpt_B": "results/vlm/openai_b_test/scored/per_image.csv",
}


def rows(p: str) -> list[dict]:
    with (ROOT / p).open() as fh:
        return list(csv.DictReader(fh))


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def pipeline_outcomes() -> dict[str, bool]:
    r = rows(PIPE)
    k = next(c for c in r[0] if "strict" in c)
    return {x["image"].replace(".jpg", ""): truthy(x[k]) for x in r}


def vlm_outcomes(path: str) -> tuple[dict[str, bool], int]:
    """Per-circuit outcome, majority vote across repeats when there are several."""
    r = rows(path)
    k = next(c for c in r[0] if "strict" in c)
    hits: Counter = Counter()
    total: Counter = Counter()
    for x in r:
        stem = x["image"].replace(".jpg", "")
        hits[stem] += truthy(x[k])
        total[stem] += 1
    n_reps = max(total.values())
    return {s: hits[s] * 2 > total[s] for s in total}, n_reps


def main() -> None:
    pipe = pipeline_outcomes()
    out: dict = {
        "_what": "Paired significance for the frontier-model comparison. Every "
                 "system sees the same 192 drawings, so McNemar (exact "
                 "binomial, conditioning on discordant pairs) and a paired "
                 "bootstrap over circuits are the right tests.",
        "n_circuits": len(pipe),
        "pipeline_strict_success": sum(pipe.values()) / len(pipe),
        "comparisons": {},
    }
    ci = bootstrap_rate(list(pipe.values()), seed=0)
    out["pipeline_strict_success_ci95"] = [ci.lo, ci.hi]

    raw_p: dict[str, float] = {}
    for name, path in RUNS.items():
        vlm, n_reps = vlm_outcomes(path)
        stems = sorted(set(pipe) & set(vlm))
        a = [pipe[s] for s in stems]
        b = [vlm[s] for s in stems]

        mc = mcnemar_exact(a, b)
        delta = bootstrap_paired_delta([float(x) for x in a],
                                       [float(x) for x in b], seed=0)
        raw_p[name] = mc.p_value
        out["comparisons"][name] = {
            "n_paired": len(stems),
            "n_repeats_in_run": n_reps,
            "vlm_strict_success": sum(b) / len(b),
            "pipeline_minus_vlm": delta.point,
            "paired_delta_ci95": [delta.lo, delta.hi],
            "delta_ci_excludes_zero": delta.excludes_zero,
            "mcnemar": {
                "both": mc.n_both, "pipeline_only": mc.n_only_a,
                "vlm_only": mc.n_only_b, "neither": mc.n_neither,
                "n_discordant": mc.n_discordant, "p_exact": mc.p_value,
            },
            "sentence": mc.describe("pipeline", name),
        }

    corrected = holm(raw_p, alpha=0.05)
    out["holm_family"] = [
        {"comparison": r.label, "p_raw": r.p_raw, "p_adjusted": r.p_adjusted,
         "significant_after_correction": r.rejected} for r in corrected
    ]

    # The question the plan asks by name.
    cb = out["comparisons"]["claude_B"]
    out["the_assisted_frontier_question"] = {
        "question": "Is the assisted frontier result (Claude, variant B, "
                    "0.5295 +/- 0.0167) distinguishable from the pipeline "
                    "(0.5312, seed 0)?",
        "mcnemar_p_exact": cb["mcnemar"]["p_exact"],
        "holm_adjusted_p": next(r.p_adjusted for r in corrected
                                if r.label == "claude_B"),
        "paired_delta": cb["pipeline_minus_vlm"],
        "paired_delta_ci95": cb["paired_delta_ci95"],
        "answer": (
            "NO -- not distinguishable. The paired difference is "
            f"{cb['pipeline_minus_vlm']:+.4f} with 95% CI "
            f"[{cb['paired_delta_ci95'][0]:+.4f}, "
            f"{cb['paired_delta_ci95'][1]:+.4f}], which contains zero, and "
            f"exact McNemar gives p = {cb['mcnemar']['p_exact']:.3f}. The "
            "manuscript must not describe either system as ahead of the other "
            "on the assisted task."
            if not cb["delta_ci_excludes_zero"] else
            "YES -- the paired interval excludes zero; see the numbers above."),
    }

    dst = ROOT / "results/table5_significance.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")

    print(f"n = {out['n_circuits']} paired circuits\n")
    for name, c in out["comparisons"].items():
        print(f"  {c['sentence']}")
        print(f"      paired delta {c['pipeline_minus_vlm']:+.4f} "
              f"CI95 [{c['paired_delta_ci95'][0]:+.4f}, "
              f"{c['paired_delta_ci95'][1]:+.4f}]"
              f"{'  (excludes 0)' if c['delta_ci_excludes_zero'] else '  (contains 0)'}")
    print("\n  Holm-corrected family:")
    for r in corrected:
        print(f"    {r.label:10s} p_raw={r.p_raw:.4g}  p_adj={r.p_adjusted:.4g}  "
              f"{'significant' if r.rejected else 'not significant'}")
    print(f"\n{out['the_assisted_frontier_question']['answer']}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
