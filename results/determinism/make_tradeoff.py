#!/usr/bin/env python3
"""Assemble results/determinism/tradeoff.json from committed artifacts only.

Reads, never runs: the two runtime scopes, the determinism measurement, the
pipeline's test- and validation-split benchmarks, and the two frontier-model
runs already on disk. NO API CALL IS MADE.

The framing this file enforces
------------------------------
ACCURACY and DETERMINISM are properties of the system. They are measured on
this hardware but they do not depend on it: the same inputs give the same
outputs and the same scores anywhere.

LATENCY IS NOT. A hosted model's response time is a property of the host --
its hardware, its queue, its batching mode, its load at that hour. The
frontier runs in this repository went through BATCH APIs, whose advertised
turnaround is up to 24 hours; using that as a latency baseline would compare
this laptop against a scheduling policy. So this file reports the pipeline's
own latency in both honest scopes, states the hardware, and makes NO
latency comparison against any hosted model. The reviewer asked for exactly
this and is right: a latency win over a hosted endpoint is not a result.

Usage:
    ./venv/bin/python results/determinism/make_tradeoff.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HEADLINE = ["strict_success", "terminal_pair_f1", "net_f1", "nged"]


def load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def main() -> None:
    rt = load("results/runtime_test192/summary.json")
    det = load("results/determinism/summary.json")
    bench_test = load("results/benchmark_test192/seed0/summary.json")
    bench_val = load("results/benchmark_1024_final/seed0/summary.json")
    vlm = {
        "claude-opus-5": load("results/vlm/claude_b/scored/summary.json"),
        "gpt-5.5-2026-04-23": load("results/vlm/openai_b/scored/summary.json"),
    }

    cached = rt["scopes"]["cached"]["stages"]
    e2e = rt["scopes"]["e2e"]["stages"]

    def topo(s):
        return {k: round(s["topology"][k]["mean"], 6) for k in HEADLINE}

    def vlm_metrics(v):
        m = v["per_repeat"][0]["metrics"]
        return {k: round(m[k]["mean"], 6) for k in HEADLINE}

    # does the determinism harness reproduce the committed benchmark exactly?
    det_means = {k: det["metric_spread_across_runs"][k]["mean"]
                 for k in HEADLINE if k in det["metric_spread_across_runs"]}
    reproduces = all(
        abs(det_means[k] - bench_test["topology"][k]["mean"]) < 1e-12
        for k in det_means)

    tradeoff = {
        "what_this_is":
            "An accuracy / latency / determinism tradeoff for the "
            "schematic2netlist pipeline against the frontier-model anchor "
            "already committed in this repository. Assembled from artifacts "
            "on disk; no model was called.",

        "claim_policy": {
            "host_independent_and_therefore_claimed": [
                "accuracy", "determinism"],
            "host_dependent_and_therefore_NOT_claimed": [
                "latency relative to any hosted model"],
            "why": "A hosted model's response time is a property of the host "
                   "-- hardware, queue depth, batching mode, load at that "
                   "hour -- not of the model. It can be changed by the "
                   "provider without touching the model, so a latency margin "
                   "over it is not a property of this system and is not "
                   "evidence about it. Accuracy and determinism are "
                   "reproducible from the same inputs on any machine.",
            "additionally_disqualifying_for_a_latency_comparison":
                "Both frontier runs were submitted through BATCH APIs (see "
                "results/vlm/PROVENANCE.md sections 3 and 8), whose turnaround "
                "is asynchronous and advertised in hours. Their wall time "
                "measures a scheduling policy, not inference.",
            "sentences_that_must_not_be_written": [
                "the pipeline is N times faster than <hosted model>",
                "the pipeline achieves comparable accuracy at a fraction of "
                "the latency of <hosted model>",
                "sub-50 ms per image (the retired headline; it excluded the "
                "detector, the class head and connectivity repair)",
            ],
            "sentences_the_evidence_supports": [
                "the pipeline runs on a 2020 MacBook Air CPU in roughly a "
                "third of a second per image end to end, with no network "
                "dependency and no per-image cost",
                "the pipeline is exactly reproducible: identical inputs give "
                "byte-identical netlists across independent runs, which the "
                "frontier baselines are not known to be and, being sampled "
                "with no seed or temperature set, are not expected to be",
            ],
        },

        "axis_accuracy": {
            "reportable_number_test_split_192": {
                "system": "schematic2netlist pipeline",
                "source": "results/benchmark_test192/seed0/summary.json",
                "split": "test (192 images, never used for selection)",
                "metrics": topo(bench_test),
                "note": "no frontier-model run exists on this split, so this "
                        "number stands alone and is NOT a head-to-head result",
            },
            "the_only_head_to_head_validation_split_190": {
                "split": "val (190 images) -- the split every parameter in "
                         "configs/default.yaml was selected on, so the "
                         "PIPELINE's entry here is IN-SAMPLE and a reviewer is "
                         "entitled to discount it; the models are not",
                "systems": {
                    "schematic2netlist pipeline": {
                        "source": "results/benchmark_1024_final/seed0/"
                                  "summary.json (its 'split: test' field is "
                                  "stale; the 2026-08-03 role swap renamed "
                                  "these 190 images to val)",
                        "metrics": topo(bench_val),
                    },
                    **{name: {
                        "source": f"results/vlm/{d}/scored/summary.json",
                        "runs": v["n_repeats"],
                        "metrics": vlm_metrics(v),
                    } for (name, v), d in zip(vlm.items(),
                                              ("claude_b", "openai_b"))},
                },
                "reading_supported": "the task is hard and a small "
                                     "specialised pipeline is COMPETITIVE "
                                     "with frontier general models on it",
                "reading_not_supported": "any ranking of the three. The gaps "
                                         "are small, the models have one run "
                                         "each, and the pipeline's entry is "
                                         "in-sample.",
            },
        },

        "axis_latency": {
            "measured_on": {
                "machine": rt["machine"],
                "processor": rt["processor"],
                "cpu_count": rt["cpu_count"],
                "accelerator": "none; CPU only, including both CNN heads",
                "concurrency": rt["concurrency_caveat"],
            },
            "scope_cached_downstream": {
                "what": rt["scopes"]["cached"]["what_it_measures"],
                "median_ms": cached["total"]["median_ms"],
                "mean_ms": cached["total"]["mean_ms"],
                "p90_ms": cached["total"]["p90_ms"],
                "n_images": rt["n_images"],
            },
            "scope_true_end_to_end": {
                "what": rt["scopes"]["e2e"]["what_it_measures"],
                "median_ms": e2e["total"]["median_ms"],
                "mean_ms": e2e["total"]["mean_ms"],
                "p90_ms": e2e["total"]["p90_ms"],
                "n_images": rt["n_images"],
            },
            "scope_cold_start_single_image_cli": {
                "what": "one image from a fresh interpreter: the "
                        "torch/ultralytics import, model construction and the "
                        "first forward pass, none of which the steady-state "
                        "figures pay",
                "median_ms": rt["detector_only_ms"][
                    "cold_start_one_image_cli"]["total_median_ms"],
                "note": "a user converting a SINGLE schematic waits for this, "
                        "not for the steady-state figure. It is start-up "
                        "cost, and it amortizes to nothing over a batch.",
            },
            "detector_alone_median_ms": {
                "inference_only_model_held_in_memory":
                    rt["detector_only_ms"]["inference_model_preloaded"][
                        "median"],
                "as_the_shipped_code_calls_it":
                    rt["detector_only_ms"]["as_shipped_model_rebuilt_per_call"][
                        "median"],
                "gap_is": "detect() -> detect_ultralytics([one image]) "
                          "reconstructs YOLO(weights) on every call; holding "
                          "the model is a free ~40 ms per image",
            },
            "which_to_quote": "the END-TO-END figure. The cached figure is an "
                              "experimental convenience -- detections are "
                              "computed once and reused so results are "
                              "reproducible -- and quoting it as user-facing "
                              "latency omits the detector entirely.",
            "no_baseline_comparison": "deliberately absent; see claim_policy",
            "per_image_marginal_cost_usd": 0.0,
            "frontier_per_image_cost_usd_for_context": {
                "claude-opus-5": 0.0279,
                "gpt-5.5-2026-04-23": 0.0288,
                "source": "results/vlm/PROVENANCE.md section 8, token-derived "
                          "batch-rate ESTIMATES (~$5.31 and ~$5.47 over 190 "
                          "images), not invoices",
                "status": "context only. Cost is a provider price list, not a "
                          "property of the model, and moves without notice.",
            },
        },

        "axis_determinism": {
            "pipeline": {
                "runs": det["runs"],
                "circuits": det["n_circuits"],
                "process_isolation": det["process_isolation"]["why"],
                "netlist_byte_identical_fraction":
                    det["exact_output_agreement"][
                        "netlist_base_byte_identical_fraction"],
                "n_circuits_topology_changed":
                    det["topology_changes"]["n_circuits_topology_changed"],
                "full_result_dict_identical_fraction":
                    det["full_result_agreement"]["fraction_all_runs_identical"],
                "headline_metric_variance": {
                    k: det["metric_spread_across_runs"][k]["variance"]
                    for k in HEADLINE if k in det["metric_spread_across_runs"]},
                "invalid_output_frequency":
                    det["invalid_outputs"]["invalid_frequency"],
                "source": "results/determinism/summary.json",
            },
            "frontier_models": {
                "status": "UNMEASURED, and not measurable from what exists",
                "runs_available": 1,
                "why_not_measured": "one repeat per model was run; "
                                    "re-running costs money and was not "
                                    "approved (results/vlm/PROVENANCE.md "
                                    "sections 6 and 12)",
                "why_non_determinism_is_expected": "neither provider was "
                                                   "given a seed or a "
                                                   "temperature, and the "
                                                   "Anthropic side records "
                                                   "only an undated model "
                                                   "alias, so even the model "
                                                   "identity is not pinned "
                                                   "(PROVENANCE.md sections 3, "
                                                   "4 and 13)",
                "do_not_quote": "score_vlm.py prints SD = 0.0000 for every "
                                "frontier metric. That is statistics.stdev "
                                "over ONE sample, not a determinism result.",
            },
            "the_asymmetry_that_is_the_point": {
                "claim": "the pipeline's single run IS its distribution; each "
                         "frontier number is ONE DRAW from a distribution "
                         "that was never sampled twice",
                "consequence_for_the_comparison": "every accuracy gap in "
                                                  "axis_accuracy is measured "
                                                  "against an unrepeated "
                                                  "sample, so its run-to-run "
                                                  "uncertainty is unknown and "
                                                  "is NOT captured by the "
                                                  "bootstrap CIs, which "
                                                  "resample images only",
                "why_this_is_a_result_and_not_a_footnote": "reproducibility is "
                                                           "a requirement for "
                                                           "an engineering "
                                                           "artifact that "
                                                           "feeds a simulator: "
                                                           "a netlist that "
                                                           "changes between "
                                                           "runs cannot be "
                                                           "signed off, "
                                                           "regressed against, "
                                                           "or debugged. The "
                                                           "pipeline meets "
                                                           "that bar by "
                                                           "measurement; the "
                                                           "hosted baselines "
                                                           "are not known to.",
            },
        },

        "cross_check": {
            "determinism_harness_reproduces_committed_benchmark": reproduces,
            "committed": {k: bench_test["topology"][k]["mean"]
                          for k in HEADLINE},
            "measured_here": det_means,
            "meaning": "the 5 determinism runs were scored by the same "
                       "functions scripts/benchmark.py uses; matching the "
                       "committed summary exactly means the determinism "
                       "measurement is of the SHIPPED configuration, not of a "
                       "variant",
        },

        "provenance": {
            "runtime": "results/runtime_test192/summary.json (harness: "
                       "results/runtime_test192/measure_runtime.py)",
            "runtime_legacy_scope": "results/runtime_test192/legacy_scope/"
                                    "summary.json (unmodified "
                                    "scripts/benchmark_runtime.py, for "
                                    "comparability with the retired number)",
            "determinism": "results/determinism/summary.json (harness: "
                           "scripts/measure_determinism.py)",
            "accuracy_test": "results/benchmark_test192/seed0/summary.json",
            "accuracy_val_and_frontier": "results/benchmark_1024_final/seed0/"
                                         "summary.json, results/vlm/*/scored/"
                                         "summary.json, "
                                         "results/vlm/PROVENANCE.md",
            "no_api_calls_made": True,
        },
    }

    out = REPO / "results/determinism/tradeoff.json"
    out.write_text(json.dumps(tradeoff, indent=2))
    print(f"wrote {out}")
    print(f"  determinism harness reproduces committed benchmark: {reproduces}")


if __name__ == "__main__":
    main()
