#!/usr/bin/env python3
"""Cross-validated parameter selection, so a swept value is not a test-set number.

THE PROBLEM THIS EXISTS FOR -- AND WHAT HAS CHANGED. Net-level ground truth used
to cover one split and nothing else, so every parameter here -- bridge_span,
component_mask_pad, min_blob_area, the snapping radii -- was chosen by sweeping
the split it was then reported on. That is test-set selection, and a reviewer is
entitled to discount any number chosen that way.

The second split is now annotated (docs/GT_VAL_VERIFICATION_REPORT.md) and on
2026-08-03 the two swapped names: the 190 images every parameter was tuned on are
now the VALIDATION split, and the 192 images that never entered selection are the
TEST split. So the structural fix is in place -- sweep with --split val, report
with --split test -- and this script is no longer the only defence.

It still earns its place. A sweep peak read on val is a peak on 190 images and is
partly noise wherever it is read; the out-of-fold number says how much. For each
fold of an image-grouped K-fold:
choose the parameter on the OTHER folds only, then read this fold's score at that
choice. No fold ever scores an image that influenced its own selection, so the
aggregate is an honest estimate of what the selection PROCEDURE achieves -- and
every image still contributes to the reported number.

Three columns are printed, and the gap between the first two is the whole point:

    naive sweep peak    the best single value measured on all 190 -- what
                        reporting a sweep maximum claims
    cross-validated     the same procedure scored out-of-fold
    shipped default     what the repository currently does

If the CV number sits well below the sweep peak, the peak was partly noise. If
the chosen parameter is unstable across folds, the sweep was fitting the split
rather than finding a property of the data, and that is worth knowing before the
value goes in a paper.

Reads per_image.csv from runs that already exist, so it costs nothing to run.

Usage:
    python scripts/cv_select_param.py --metric terminal_pair_f1 \\
        9=results/benchmark_1024/span9 7=results/benchmark_1024/span7 ...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def num(v):
    """per_image.csv stores booleans as True/False; strict_success is one."""
    s = str(v).strip()
    if s in ("True", "true"):
        return 1.0
    if s in ("False", "false", ""):
        return 0.0
    return float(s)


def load(run: str) -> dict[str, dict]:
    p = Path(run) / "per_image.csv"
    if not p.exists():
        return {}
    return {r["image"]: r for r in csv.DictReader(p.open())}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="value=run_dir")
    ap.add_argument("--metric", default="terminal_pair_f1",
                    help="column selection maximizes (nged is minimized)")
    ap.add_argument("--report", nargs="*",
                    default=["terminal_pair_f1", "net_f1",
                             "per_component_connected_acc", "strict_success",
                             "nged"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--default", default=None,
                    help="value=run_dir for the shipped default, shown for "
                         "reference and never selectable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sweeps/cv_selection.json")
    args = ap.parse_args()

    table: dict[str, dict[str, dict]] = {}
    for spec in args.runs:
        val, _, run = spec.partition("=")
        rows = load(run)
        if rows:
            table[val] = rows
        else:
            print(f"[warn] no per_image.csv in {run}; skipping {val}")
    if len(table) < 2:
        raise SystemExit("need at least two runs to select between")

    common = sorted(set.intersection(*(set(v) for v in table.values())))
    vals = list(table)
    lower_better = args.metric == "nged"
    print(f"{len(vals)} candidate values over {len(common)} shared images, "
          f"selecting on {args.metric} "
          f"({'lower' if lower_better else 'higher'} is better)\n")

    def score(val: str, images, col: str) -> float:
        return float(np.mean([num(table[val][im][col]) for im in images]))

    # naive: the best single value measured on everything
    naive = min(vals, key=lambda v: score(v, common, args.metric)) if lower_better \
        else max(vals, key=lambda v: score(v, common, args.metric))

    rng = np.random.default_rng(args.seed)
    order = list(common)
    rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]

    picked: list[str] = []
    held: dict[str, list[float]] = {c: [] for c in args.report}
    for f in folds:
        train = [im for im in common if im not in set(f)]
        best = min(vals, key=lambda v: score(v, train, args.metric)) \
            if lower_better else max(vals, key=lambda v: score(v, train, args.metric))
        picked.append(best)
        for c in args.report:
            held[c].extend(num(table[best][im][c]) for im in f)

    from collections import Counter
    stability = Counter(picked)
    dflt_rows = None
    if args.default:
        dval, _, drun = args.default.partition("=")
        dflt_rows = load(drun)

    print(f"{'metric':30s} {'naive peak':>11s} {'CROSS-VAL':>11s} "
          f"{'shipped':>10s}")
    out = {"metric": args.metric, "folds": args.folds, "n_images": len(common),
           "naive_best": naive, "fold_choices": picked,
           "stability": dict(stability), "results": {}}
    for c in args.report:
        nv = score(naive, common, c)
        cv = float(np.mean(held[c]))
        ds = (float(np.mean([num(dflt_rows[im][c]) for im in common
                             if im in dflt_rows])) if dflt_rows else float("nan"))
        print(f"{c:30s} {nv:11.4f} {cv:11.4f} {ds:10.4f}")
        out["results"][c] = {"naive_peak": round(nv, 6),
                             "cross_validated": round(cv, 6),
                             "shipped_default": None if dflt_rows is None
                             else round(ds, 6)}

    print(f"\nnaive sweep peak at value {naive!r}")
    print(f"fold choices: " + ", ".join(f"{k}x{v}" for k, v in
                                        stability.most_common()))
    if len(stability) == 1:
        print("  the same value won every fold -- the choice is stable, and the")
        print("  gap between the first two columns is selection noise only")
    else:
        print("  the choice MOVES between folds, so the sweep is partly fitting")
        print("  the split; prefer the cross-validated column")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
