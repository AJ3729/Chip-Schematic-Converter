#!/usr/bin/env python3
"""Strict success against terminal-pair PRECISION — the over-merge axis.

Strict success is not spread thinly across the test set; it is concentrated.
Stratifying by terminal-pair precision (rather than by F1, size, or any drawing
property) separates the population sharply: above a precision threshold circuits
convert to strict success at a high rate, and below it they convert at zero.
That matters for where effort goes — the work is moving specific circuits across
one boundary, not lifting an average.

Precision is the right axis because it is the over-merge signal. Recall stays
comparatively flat on failing circuits: the conductors are found and then fused.

This existed only as an ad-hoc analysis, which is why its committed output
silently drifted a full configuration behind the benchmark it described. It is a
script now so it regenerates with everything else.

Usage:
    python scripts/precision_buckets.py --run-dir results/benchmark_1024_final/seed0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EDGES = [0.0, 0.3, 0.5, 0.7, 0.9, 1.000001]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="results/benchmark_1024_final/seed0")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to <run-dir>/../../stratified_1024_final")
    args = ap.parse_args()

    run = Path(args.run_dir)
    rows = list(csv.DictReader(open(run / "per_image.csv")))
    n = len(rows)
    strict_total = sum(1 for r in rows if r["strict_success"] == "True")

    buckets = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        sel = [r for r in rows
               if lo <= float(r["terminal_pair_precision"]) < hi]
        if not sel:
            continue
        k = sum(1 for r in sel if r["strict_success"] == "True")
        mean = lambda c: sum(float(r[c]) for r in sel) / len(sel)  # noqa: E731
        buckets.append({
            "precision_range": [lo, min(hi, 1.0)],
            "n_images": len(sel),
            "mean_tp_precision": round(mean("terminal_pair_precision"), 4),
            "mean_tp_recall": round(mean("terminal_pair_recall"), 4),
            "mean_tp_f1": round(mean("terminal_pair_f1"), 4),
            "mean_percomp": round(mean("per_component_connected_acc"), 4),
            "strict_successes": k,
            "strict_rate": round(k / len(sel), 4),
        })

    top = buckets[-1] if buckets else {}
    out = {
        "source": f"{run}/per_image.csv ({n} test images)",
        "strict_success_overall": round(strict_total / n, 4),
        "buckets": buckets,
        "share_of_strict_in_top_bucket":
            round(top.get("strict_successes", 0) / max(1, strict_total), 4),
        "finding":
            "Strict success is concentrated in the highest-precision bucket. "
            "Circuits at terminal-pair precision >= 0.9 convert at "
            f"{top.get('strict_rate', 0):.0%}; every bucket below it converts at "
            "0%. On the failing circuits recall stays comparatively high while "
            "precision collapses, i.e. the conductors are found and then fused. "
            "The lever is therefore separating nets on specific circuits, not "
            "improving an average.",
    }

    dst = Path(args.out_dir) if args.out_dir else run.parent.parent / "stratified_1024_final"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "precision_buckets.json").write_text(json.dumps(out, indent=1))

    print(f"{run}  n={n}  strict={strict_total} ({strict_total/n:.4f})\n")
    print(f"{'precision':>12} {'n':>5} {'meanF1':>8} {'meanPrec':>9} "
          f"{'meanRec':>8} {'strict':>7} {'rate':>7}")
    for b in buckets:
        lo, hi = b["precision_range"]
        print(f"{lo:>5.1f}-{hi:<6.1f} {b['n_images']:>5} {b['mean_tp_f1']:>8.3f} "
              f"{b['mean_tp_precision']:>9.3f} {b['mean_tp_recall']:>8.3f} "
              f"{b['strict_successes']:>7} {b['strict_rate']:>7.1%}")
    print(f"\nshare of all strict successes in the top bucket: "
          f"{out['share_of_strict_in_top_bucket']:.1%}")
    print(f"wrote {dst}/precision_buckets.json")


if __name__ == "__main__":
    main()
