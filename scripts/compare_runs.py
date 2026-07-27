#!/usr/bin/env python3
"""Paired comparison of two benchmark runs (the project's standard test).

Benchmark runs over the same images are PAIRED measurements, so
significance comes from a bootstrap over per-image deltas — never from
overlap of the two runs' independent CIs (which is far too conservative
and was the source of one wrong conclusion in this project's history).

Emits a CSV + prints a table: per metric, mean A, mean B, delta,
bootstrap 95% CI of the delta, win/loss/tie counts, significance.

Usage:
    python scripts/compare_runs.py results/v4_stitch results/v5_stitch_crossover
    python scripts/compare_runs.py A B --out results/comparisons/v4_vs_v5.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as st
from pathlib import Path

METRICS = [
    ("terminal_pair_f1", "terminal-pair F1", +1),
    ("net_f1", "net F1", +1),
    ("per_component_connected_acc", "per-component connected", +1),
    ("nged", "nGED", -1),                      # lower is better
    ("strict_success", "strict success", +1),
    ("solvable_before", "DC-solvable (pre-repair)", +1),
    ("solvable_after", "DC-solvable (post-repair)", +1),
]


def _val(x: str) -> float:
    if x in ("True", "true"):
        return 1.0
    if x in ("False", "false"):
        return 0.0
    return float(x)


def load(run_dir: str) -> dict[str, dict]:
    p = Path(run_dir) / "per_image.csv"
    return {r["image"]: r for r in csv.DictReader(open(p))}


def compare(a_dir: str, b_dir: str, n_boot: int = 2000, seed: int = 0) -> list[dict]:
    a, b = load(a_dir), load(b_dir)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common images between the two runs")
    rng = random.Random(seed)

    rows = []
    for key, label, _sign in METRICS:
        if key not in next(iter(a.values())) or key not in next(iter(b.values())):
            continue
        d = [_val(b[i][key]) - _val(a[i][key]) for i in common]
        boots = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
        rows.append({
            "metric": key,
            "label": label,
            "n_images": len(common),
            "mean_a": round(st.mean(_val(a[i][key]) for i in common), 4),
            "mean_b": round(st.mean(_val(b[i][key]) for i in common), 4),
            "delta": round(st.mean(d), 4),
            "ci95_lo": round(lo, 4),
            "ci95_hi": round(hi, 4),
            "wins": sum(1 for x in d if x > 1e-9),
            "losses": sum(1 for x in d if x < -1e-9),
            "ties": sum(1 for x in d if abs(x) <= 1e-9),
            "significant": bool(lo > 0 or hi < 0),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a", help="baseline run directory")
    ap.add_argument("run_b", help="candidate run directory")
    ap.add_argument("--out", default=None,
                    help="CSV path (default results/comparisons/<a>_vs_<b>.csv)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    rows = compare(args.run_a, args.run_b, n_boot=args.n_boot)

    na, nb = Path(args.run_a).name, Path(args.run_b).name
    out = Path(args.out) if args.out else Path("results/comparisons") / f"{na}_vs_{nb}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"paired comparison: {na} -> {nb}  ({rows[0]['n_images']} images, "
          f"{args.n_boot} bootstrap resamples)\n")
    print(f"{'metric':26s} {'A':>8s} {'B':>8s} {'delta':>8s} "
          f"{'95% CI':>20s} {'W/L/T':>12s}  sig")
    for r in rows:
        ci = f"[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}]"
        wlt = f"{r['wins']}/{r['losses']}/{r['ties']}"
        print(f"{r['label']:26s} {r['mean_a']:8.4f} {r['mean_b']:8.4f} "
              f"{r['delta']:+8.4f} {ci:>20s} {wlt:>12s}  "
              f"{'YES' if r['significant'] else 'no'}")
    print(f"\n[OK] wrote {out}")


if __name__ == "__main__":
    main()
