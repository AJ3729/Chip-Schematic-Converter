#!/usr/bin/env python3
"""Check the runtime claims that were in circulation, against this run.

Three claims were on the table. This script re-derives each from the
artifacts in this directory and the committed ablation, and writes
claims_check.json. Reads only; runs nothing.

    1. "~46 ms per image"          -- the retired headline
    2. "stitch is ~54% of downstream runtime"
    3. "stitch is an accuracy no-op, so removing it is a clean optimisation"

Usage:
    ./venv/bin/python results/runtime_test192/check_claims.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    rt = json.loads((HERE / "summary.json").read_text())
    legacy = json.loads((HERE / "legacy_scope" / "summary.json").read_text())
    old = json.loads((REPO / "results/runtime_1024/summary.json").read_text())

    cached = rt["scopes"]["cached"]["stages"]
    e2e = rt["scopes"]["e2e"]["stages"]

    # accuracy cost of removing stitch, from the committed paired comparison
    abl = {}
    p = REPO / "results/comparisons/seed0_vs_abl_nostitch.csv"
    for row in csv.DictReader(p.open()):
        abl[row["metric"]] = {
            "with_stitch": float(row["mean_a"]),
            "without_stitch": float(row["mean_b"]),
            "delta": float(row["delta"]),
            "wins": int(row["wins"]), "losses": int(row["losses"]),
            "ties": int(row["ties"]),
            "significant": row["significant"] == "True",
        }

    # what would end-to-end latency become with stitch deleted?
    rows = list(csv.DictReader((HERE / "per_image_e2e.csv").open()))
    e2e_wo = [float(r["total"]) - float(r["stitch"]) for r in rows]

    out = {
        "claim_1_46ms_headline": {
            "claim": "the pipeline runs in ~46 ms per image",
            "verdict": "RETIRED -- it is a fragment of a fragment",
            "source_of_the_claim": "results/runtime_1024/summary.json "
                                   f"(total mean {old['total']['mean_ms']} ms, "
                                   f"n={old['n_images']}, "
                                   f"detector_timed={old['detector_timed']})",
            "what_it_omitted": [
                "the detector, which was never run (detector_timed false); "
                "its 0.28 ms 'detect' stage is a JSON cache read",
                "the component class head, which is enabled in the shipped "
                "config and is the LARGEST downstream stage measured here",
                "constraint-triggered connectivity repair, also enabled",
            ],
            "sample": f"{old['n_images']} images, and on the 190-image split "
                      "that is today called val, not the test split",
            "replacement_numbers_median_ms": {
                "cached_downstream": cached["total"]["median_ms"],
                "true_end_to_end": e2e["total"]["median_ms"],
            },
        },

        "claim_2_stitch_is_54_percent": {
            "claim": "stitch is ~54% of downstream runtime",
            "verdict": "DOES NOT REPLICATE at either denominator",
            "original": {
                "value": old["stitch"]["share_of_total"],
                "denominator": "mean total of the 11 stages "
                               "scripts/benchmark_runtime.py measures",
                "n_images": old["n_images"],
            },
            "replication_same_scope_same_script": {
                "value": legacy["stitch"]["share_of_total"],
                "how": "unmodified scripts/benchmark_runtime.py, 189 timed "
                       "images of the 192-image test split -> "
                       "results/runtime_test192/legacy_scope/",
                "n_images": legacy["n_images"],
            },
            "replication_same_scope_this_harness": {
                "mean": rt["stitch_share_legacy_scope"]["stitch_share_mean"],
                "median": rt["stitch_share_legacy_scope"][
                    "stitch_share_median"],
                "how": "this harness, restricted to the same stage subset",
            },
            "share_of_the_pipeline_that_actually_runs": {
                "of_cached_downstream_mean":
                    cached["stitch"]["share_of_total_mean"],
                "of_cached_downstream_median":
                    cached["stitch"]["share_of_total_median"],
                "of_true_end_to_end_mean": e2e["stitch"]["share_of_total_mean"],
                "of_true_end_to_end_median":
                    e2e["stitch"]["share_of_total_median"],
            },
            "why_the_number_shrank": "the 54% denominator excluded the class "
                                     "head, which costs more than stitch, and "
                                     "excluded the detector entirely. Against "
                                     "what a user actually waits for, stitch "
                                     "is under a tenth of the budget.",
        },

        "claim_3_removing_stitch_is_a_clean_optimisation": {
            "claim": "stitch is an accuracy no-op, so deleting it is free "
                     "speed",
            "verdict": "the ACCURACY half holds on the split it was measured "
                       "on; the SPEED half is much smaller than advertised",
            "accuracy_evidence": {
                "source": "results/comparisons/seed0_vs_abl_nostitch.csv",
                "split": "190 images -- the split now called val, NOT the "
                         "192-image test split. The no-op has never been "
                         "checked on test.",
                "per_metric": abl,
                "reading": "identical on terminal-pair F1, net F1, "
                           "per-component, strict success and DC-solvability "
                           "(190 ties, zero wins, zero losses); nGED is "
                           "slightly BETTER without stitch",
            },
            "speed_evidence": {
                "stitch_median_ms": cached["stitch"]["median_ms"],
                "end_to_end_median_ms": e2e["total"]["median_ms"],
                "end_to_end_median_ms_if_stitch_removed": round(
                    1000 * statistics.median(e2e_wo), 2),
                "end_to_end_saving_fraction": round(
                    e2e["stitch"]["median_ms"] / e2e["total"]["median_ms"], 4),
            },
            "better_target": {
                "stage": "class_head",
                "median_ms": cached["class_head"]["median_ms"],
                "share_of_true_end_to_end_median":
                    e2e["class_head"]["share_of_total_median"],
                "note": "two CPU CNNs over every component crop, ensembled. "
                        "It costs more than every other downstream stage "
                        "combined. Unlike stitch it is NOT an accuracy no-op "
                        "(configs/default.yaml records +0.0053 strict for the "
                        "second head), so this is a real tradeoff rather than "
                        "free speed -- but it is where the time is.",
            },
        },

        "caveat": rt["concurrency_caveat"],
    }

    (HERE / "claims_check.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {HERE / 'claims_check.json'}")
    c2 = out["claim_2_stitch_is_54_percent"]
    print(f"  stitch share: claimed {c2['original']['value']:.1%} | "
          f"same scope replicated "
          f"{c2['replication_same_scope_same_script']['value']:.1%} | "
          f"of true end-to-end "
          f"{c2['share_of_the_pipeline_that_actually_runs']['of_true_end_to_end_median']:.1%}"
          " (median)")
    c3 = out["claim_3_removing_stitch_is_a_clean_optimisation"]["speed_evidence"]
    print(f"  removing stitch: {c3['end_to_end_median_ms']} -> "
          f"{c3['end_to_end_median_ms_if_stitch_removed']} ms median "
          f"({c3['end_to_end_saving_fraction']:.1%} saving)")


if __name__ == "__main__":
    main()
