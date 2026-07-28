#!/usr/bin/env python3
"""Stratified performance + failure-mode analysis (Week-3/4 deliverable).

Answers "where does the pipeline fail, and on what kind of drawing?"
from committed artifacts only — a benchmark run's ``per_image.csv``
joined against the verified GT files, which supply circuit size and
composition. No pipeline re-run, so this is cheap and reproducible.

Strata reported:
  * circuit size (component-count tertiles of the test split)
  * presence of a wire crossover in the drawing
  * presence of a multi-terminal device (MOSFET/BJT/Op-Amp)
  * presence of a GND symbol

and, separately, the worst images by net F1 with the diagnostic columns
(unmatched components, unsnapped-ness proxy) that say which stage to
blame.

Usage:
    python scripts/analyze_failures.py --run-dir results/v5_stitch_crossover
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from schematic2netlist.classes import canonical_class, class_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections

METRICS = ["terminal_pair_f1", "net_f1", "per_component_connected_acc",
           "nged", "strict_success"]


def as_float(v) -> float:
    if isinstance(v, str):
        if v.lower() in ("true", "false"):      # some CSVs store bools as text
            return 1.0 if v.lower() == "true" else 0.0
        return float(v) if v else 0.0
    return float(v)


def gt_features(gt_dir: Path, det_dir: Path, stem: str) -> dict | None:
    p = gt_dir / f"{stem}.json"
    if not p.exists():
        return None
    gt = json.loads(p.read_text())
    comps = gt["components"]
    classes = [canonical_class(c["class"]) for c in comps]

    # Crossovers are drawing annotations, not electrical components, so
    # they have no GT topology entry — read them from the detections.
    n_crossovers = 0
    dp = det_dir / f"{stem}.json"
    if dp.exists():
        n_crossovers = sum(
            1 for d in load_cached_detections(dp)
            if canonical_class(d.get("class", "")) == "Wire Crossover"
        )

    # GT boxes were bootstrapped and verified for TOPOLOGY, not geometry;
    # a squared-off box cannot reach the IoU threshold against a properly
    # shaped detection, which shows up as a spurious zero (see
    # scripts/threshold_sensitivity.py).
    squarish = sum(
        1 for c in comps
        if abs(c["bbox"][2] - c["bbox"][3]) / max(c["bbox"][2], c["bbox"][3]) < 0.05
    )
    return {
        "n_components": len(comps),
        "has_crossover": n_crossovers > 0,
        "n_crossovers": n_crossovers,
        "has_multiterminal": any(class_terminals(c) >= 3 for c in classes),
        "has_ground": any(c == "GND" for c in classes),
        "n_terminals": sum(len(c["terminals"]) for c in comps),
        "frac_squarish_gt_boxes": round(squarish / len(comps), 3) if comps else 0.0,
    }


def summarize(rows: list[dict], label: str) -> dict:
    out = {"stratum": label, "n_images": len(rows)}
    for m in METRICS:
        vals = [as_float(r[m]) for r in rows if r.get(m) not in (None, "")]
        out[m] = round(statistics.mean(vals), 4) if vals else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/v5_stitch_crossover")
    ap.add_argument("--gt-dir", default=None,
                    help="overrides benchmark.gt_dir from the config")
    ap.add_argument("--det-dir", default="data/detections")
    ap.add_argument("--out-dir", default="results/stratified")
    ap.add_argument("--worst", type=int, default=15)
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir or load_config(None)["benchmark"]["gt_dir"])
    run_dir = Path(args.run_dir)
    det_dir = Path(args.det_dir)
    with (run_dir / "per_image.csv").open() as fh:
        rows = list(csv.DictReader(fh))

    joined = []
    for r in rows:
        feats = gt_features(gt_dir, det_dir, Path(r["image"]).stem)
        if feats is None:
            continue
        joined.append({**r, **feats})
    if not joined:
        raise SystemExit("no rows joined to GT — check --gt-dir")

    sizes = sorted(r["n_components"] for r in joined)
    t1 = sizes[len(sizes) // 3]
    t2 = sizes[2 * len(sizes) // 3]

    strata = [
        ("all", joined),
        (f"small (<={t1} components)",
         [r for r in joined if r["n_components"] <= t1]),
        (f"medium ({t1 + 1}-{t2})",
         [r for r in joined if t1 < r["n_components"] <= t2]),
        (f"large (>{t2})", [r for r in joined if r["n_components"] > t2]),
        ("has wire crossover", [r for r in joined if r["has_crossover"]]),
        ("no wire crossover", [r for r in joined if not r["has_crossover"]]),
        ("has 3+ terminal device", [r for r in joined if r["has_multiterminal"]]),
        ("only 2-terminal devices",
         [r for r in joined if not r["has_multiterminal"]]),
        ("has GND symbol", [r for r in joined if r["has_ground"]]),
        ("no GND symbol", [r for r in joined if not r["has_ground"]]),
    ]
    table = [summarize(rs, label) for label, rs in strata if rs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stratified.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    worst = sorted(joined, key=lambda r: as_float(r["net_f1"]))[: args.worst]
    worst_rows = [{
        "image": r["image"],
        "n_components": r["n_components"],
        "net_f1": round(as_float(r["net_f1"]), 4),
        "terminal_pair_f1": round(as_float(r["terminal_pair_f1"]), 4),
        "unmatched_pred": r.get("unmatched_pred", ""),
        "unmatched_gt": r.get("unmatched_gt", ""),
        "has_crossover": r["has_crossover"],
        "has_multiterminal": r["has_multiterminal"],
        "frac_squarish_gt_boxes": r["frac_squarish_gt_boxes"],
    } for r in worst]
    with (out_dir / "worst_images.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(worst_rows[0].keys()))
        w.writeheader()
        w.writerows(worst_rows)

    print(f"stratified performance ({run_dir}, n={len(joined)})\n")
    print(f"  {'stratum':26s} {'n':>4s} {'netF1':>7s} {'tpF1':>7s} "
          f"{'perComp':>8s} {'strict':>7s}")
    for row in table:
        print(f"  {row['stratum']:26s} {row['n_images']:4d} "
              f"{row['net_f1']:7.4f} {row['terminal_pair_f1']:7.4f} "
              f"{row['per_component_connected_acc']:8.4f} "
              f"{row['strict_success']:7.4f}")

    print(f"\nworst {len(worst_rows)} images by net F1:")
    for r in worst_rows:
        print(f"  {r['image']:20s} n={r['n_components']:3d} "
              f"netF1={r['net_f1']:.3f} unmatched_gt={r['unmatched_gt']}")
    print(f"\nwrote {out_dir}/stratified.csv + worst_images.csv")


if __name__ == "__main__":
    main()
