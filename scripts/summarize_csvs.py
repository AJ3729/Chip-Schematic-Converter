#!/usr/bin/env python3
"""Headline stats (n, mean, std, min, median, max) for every CSV in the repo.

The repository accumulates per-image tables, sweeps, and comparisons, but
reading a raw 190-row CSV tells you nothing at a glance. This walks every
CSV, infers which columns are numeric, and prints the summary each table
should have shipped with.

Booleans are handled: `strict_success` is written as True/False by
benchmark.py, so a naive float() drops the column entirely — here they
map to 1/0 and the mean is the success RATE.

Two output modes:
  - console: grouped by directory, one block per file
  - --out FILE.csv: one tidy row per (file, column) for further analysis

Also flags columns that are constant (std == 0) — usually a sign a knob
did nothing, which is worth noticing.

Usage:
    python scripts/summarize_csvs.py
    python scripts/summarize_csvs.py --root results/ablations_1024
    python scripts/summarize_csvs.py --out results/csv_headline_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from pathlib import Path

TRUEY = {"true", "yes"}
FALSEY = {"false", "no"}

# columns that are identifiers, not measurements
SKIP = {"image", "config", "label", "metric", "stratum", "run_dir",
        "git_sha", "config_hash", "file", "split", "class", "drafter",
        "source", "node", "significant"}


def as_number(v: str):
    s = (v or "").strip()
    if not s:
        return None
    low = s.lower()
    if low in TRUEY:
        return 1.0
    if low in FALSEY:
        return 0.0
    try:
        f = float(s)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def summarize(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    out = []
    for col in rows[0].keys():
        if col is None or col.lower() in SKIP:
            continue
        vals = [as_number(r.get(col, "")) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 1:
            continue
        out.append({
            "file": str(path),
            "column": col,
            "n": len(vals),
            "mean": round(st.mean(vals), 4),
            "std": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "median": round(st.median(vals), 4),
            "max": round(max(vals), 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="directory to walk")
    ap.add_argument("--out", default=None, help="also write a tidy CSV")
    ap.add_argument("--min-rows", type=int, default=2,
                    help="skip tables with fewer data rows than this")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(p for p in root.rglob("*.csv")
                   if ".git" not in p.parts and "venv" not in p.parts)
    all_rows: list[dict] = []
    last_dir = None
    for p in files:
        try:
            stats = summarize(p)
        except Exception as e:                     # keep walking
            print(f"  !! {p}: {type(e).__name__}: {e}")
            continue
        if not stats or stats[0]["n"] < args.min_rows:
            continue
        d = str(p.parent)
        if d != last_dir:
            print(f"\n{'=' * 78}\n{d}\n{'=' * 78}")
            last_dir = d
        print(f"\n  {p.name}   ({stats[0]['n']} rows)")
        print(f"    {'column':34s} {'mean':>9s} {'std':>9s} "
              f"{'min':>9s} {'median':>9s} {'max':>9s}")
        for s in stats:
            flag = "  <- constant" if s["std"] == 0.0 and s["n"] > 1 else ""
            print(f"    {s['column']:34s} {s['mean']:9.4f} {s['std']:9.4f} "
                  f"{s['min']:9.4f} {s['median']:9.4f} {s['max']:9.4f}{flag}")
        all_rows.extend(stats)

    print(f"\n{'=' * 78}")
    n_files = len({r["file"] for r in all_rows})
    print(f"{len(files)} CSVs found, {n_files} summarized, "
          f"{len(all_rows)} numeric columns")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
