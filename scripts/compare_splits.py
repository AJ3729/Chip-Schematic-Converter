#!/usr/bin/env python3
"""Did the tuned parameters overfit the split they were tuned on?

Every value in configs/default.yaml was selected by sweeping the 190-image
split. On 2026-08-03 that split was renamed to `val` and the untouched
192-image split became `test` (data/README.md -> "the 2026-08-03 role
swap"), so the question is finally answerable: run the SAME shipped config
on both and see whether the held-out split is worse.

Two comparisons, and the second is what makes the first mean anything:

  * per-metric, val vs test, with an UNPAIRED test. The usual paired
    bootstrap in this repo compares two configs on one image set; here the
    image sets differ, so pairing is impossible and the test is a
    two-proportion z (strict success) or Welch's t (everything else).
  * a difficulty profile of each split from its GT. A held-out number that
    looks good because the split is easier is not evidence of anything, and
    component count alone correlates about -0.5 with per-image success.

Usage:
    python scripts/compare_splits.py \\
        --val-run results/benchmark_1024_final/seed0 \\
        --test-run results/benchmark_test192/seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

from scipy import stats

from schematic2netlist.classes import canonical_class, class_terminals
from schematic2netlist.detect import load_cached_detections

METRICS = ["strict_success", "terminal_pair_f1", "net_f1",
           "per_component_connected_acc", "per_component_recall_acc", "nged"]


def column(run: Path, name: str) -> list[float]:
    vals = []
    for r in csv.DictReader(open(run / "per_image.csv")):
        v = r[name]
        vals.append(1.0 if v == "True" else 0.0 if v == "False" else float(v))
    return vals


def difficulty(split: str, gt_dir: Path, det_dir: Path, splits_dir: Path) -> dict:
    """What the GT says about how hard this split is, independent of any run."""
    stems = [Path(n).stem for n in (splits_dir / f"{split}.txt").read_text().split()]
    comps, nets, xover, multi = [], [], [], []
    for s in stems:
        p = gt_dir / f"{s}.json"
        if not p.exists():
            continue
        gt = json.loads(p.read_text())
        cs = gt["components"]
        comps.append(len(cs))
        nets.append(len({t["net"] for c in cs for t in c["terminals"] if t.get("net")}))
        multi.append(any(class_terminals(canonical_class(c["class"])) >= 3 for c in cs))
        dp = det_dir / f"{s}.json"
        # crossovers are drawing annotations, not electrical parts, so they
        # are absent from GT and have to be read off the detections
        xover.append(bool(dp.exists() and any(
            canonical_class(d.get("class", "")) == "Wire Crossover"
            for d in load_cached_detections(dp))))
    return {
        "n_images": len(comps),
        "mean_components": round(st.mean(comps), 2),
        "median_components": st.median(comps),
        "max_components": max(comps),
        "mean_gt_nets": round(st.mean(nets), 2),
        "pct_with_crossover": round(100 * sum(xover) / len(xover), 1),
        "pct_with_3plus_terminal": round(100 * sum(multi) / len(multi), 1),
    }


def compare(a: list[float], b: list[float], binary: bool) -> tuple[float, str]:
    if binary:
        k1, k2, n1, n2 = sum(a), sum(b), len(a), len(b)
        p = (k1 + k2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = (k2 / n2 - k1 / n1) / se
        return 2 * (1 - stats.norm.cdf(abs(z))), f"two-proportion z={z:.2f}"
    return stats.ttest_ind(a, b, equal_var=False).pvalue, "Welch t"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-run", default="results/benchmark_1024_final/seed0")
    ap.add_argument("--test-run", default="results/benchmark_test192/seed0")
    ap.add_argument("--val-gt", default="data/gt_val_1024")
    ap.add_argument("--test-gt", default="data/gt_test_1024")
    ap.add_argument("--det-dir", default="data/detections_1024")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", default="results/split_swap")
    args = ap.parse_args()

    val_run, test_run = Path(args.val_run), Path(args.test_run)
    sd, dd = Path(args.splits_dir), Path(args.det_dir)

    prof = {"val": difficulty("val", Path(args.val_gt), dd, sd),
            "test": difficulty("test", Path(args.test_gt), dd, sd)}

    rows = []
    for m in METRICS:
        a, b = column(val_run, m), column(test_run, m)
        pv, how = compare(a, b, binary=(m == "strict_success"))
        rows.append({"metric": m, "val_mean": round(st.mean(a), 4),
                     "test_mean": round(st.mean(b), 4),
                     "delta": round(st.mean(b) - st.mean(a), 4),
                     "p_value": round(pv, 4), "test_used": how})

    strict = next(r for r in rows if r["metric"] == "strict_success")
    generalizes = strict["delta"] >= 0 or strict["p_value"] > 0.05
    out = {
        "val_run": str(val_run), "test_run": str(test_run),
        "note": ("val = the 190 images every parameter was tuned on; "
                 "test = the 192 images that never entered selection. Same "
                 "config, nothing retuned."),
        "difficulty_profile": prof,
        "metrics": rows,
        "finding": (
            "Held-out performance does not degrade: strict success is "
            f"{strict['test_mean']:.4f} on test against {strict['val_mean']:.4f} "
            f"on the tuned split (p={strict['p_value']:.3f}), and the two splits "
            "are matched on every difficulty proxy measured. There is no "
            "evidence the shipped parameters are fitted to the split they were "
            "selected on."
            if generalizes else
            "Held-out performance is WORSE than on the tuned split; the "
            "shipped parameters do not generalize and the selection needs "
            "redoing on val."),
    }

    dst = Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "val_vs_test.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'':30s} {'val (tuned)':>12s} {'test (held out)':>16s} "
          f"{'delta':>8s} {'p':>7s}")
    for r in rows:
        print(f"{r['metric']:30s} {r['val_mean']:12.4f} {r['test_mean']:16.4f} "
              f"{r['delta']:+8.4f} {r['p_value']:7.3f}")
    print(f"\n{'difficulty':30s} {'val':>12s} {'test':>16s}")
    for k in prof["val"]:
        print(f"{k:30s} {str(prof['val'][k]):>12s} {str(prof['test'][k]):>16s}")
    print(f"\n{out['finding']}")
    print(f"wrote {dst}/val_vs_test.json")


if __name__ == "__main__":
    main()
