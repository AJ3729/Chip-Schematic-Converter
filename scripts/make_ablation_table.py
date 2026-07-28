#!/usr/bin/env python3
"""Consolidate committed benchmark runs into one ablation table (BUILD-C3).

Reads each run's ``summary.json`` + ``run_meta.json`` and emits a single
CSV where every row is one configuration and every number is pulled from
the committed artifacts — never hand-typed (project rule). The default
manifest is the C2 headline progression (classical → +boundary snap →
+stitching → +crossover-aware), i.e. the story of the wire/connectivity
ablation on the 190-image verified test split.

Usage:
    python scripts/make_ablation_table.py                      # default manifest
    python scripts/make_ablation_table.py --out my.csv \
        --runs label1=results/dirA label2=results/dirB
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The canonical C2 progression. Labels match the HANDOFF/paper narrative.
DEFAULT_MANIFEST = [
    ("v1_classical_directional", "results/ablations/wire_method/classical"),
    ("v1_crossover_null", "results/ablations/wire_method/crossover_aware"),
    ("v2_newpreproc", "results/v2_newpreproc/classical"),
    ("v2_newpreproc_crossover", "results/v2_newpreproc/crossover_aware"),
    ("v3_ink_boundary_snap", "results/v3_boundary_snap"),
    ("v4_plus_stitching", "results/v4_stitch"),
    ("v5_plus_crossover_DEFAULT", "results/v5_stitch_crossover"),
]

TOPOLOGY_METRICS = [
    "terminal_pair_f1",
    "net_f1",
    "per_component_connected_acc",
    "nged",
    "strict_success",
]
REPAIR_METRICS = [
    "solvable_before_rate",
    "solvable_after_rate",
    "mean_assumptions",
    "mean_gauge",
    "spice_valid_rate",
]


def row_for(label: str, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    cfg = meta["config"]
    row = {
        "label": label,
        "run_dir": str(run_dir),
        "git_sha": meta["git_sha"][:8],
        "config_hash": summary.get("config_hash", ""),
        "n_images": summary.get("scored", ""),
        "wires_method": cfg["wires"].get("method", "canny(legacy)"),
        "stitch_masked_gaps": cfg["wires"].get("stitch_masked_gaps", False),
        "handle_crossovers": cfg["nodes"]["handle_crossovers"],
        "snapping_strategy": cfg["snapping"]["strategy"],
    }
    topo = summary.get("topology", {})
    for m in TOPOLOGY_METRICS:
        v = topo.get(m)
        if isinstance(v, dict):
            row[m] = round(v["mean"], 4)
            row[f"{m}_ci95_lo"] = round(v["ci95_lo"], 4)
            row[f"{m}_ci95_hi"] = round(v["ci95_hi"], 4)
        else:
            row[m] = ""
            row[f"{m}_ci95_lo"] = row[f"{m}_ci95_hi"] = ""
    rep = summary.get("repair", {})
    for m in REPAIR_METRICS:
        row[m] = round(rep[m], 4) if m in rep else ""
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/ablations/wire_method.csv")
    ap.add_argument(
        "--runs",
        nargs="*",
        help="label=run_dir pairs; default = the C2 headline manifest",
    )
    args = ap.parse_args()

    manifest = (
        [tuple(spec.split("=", 1)) for spec in args.runs]
        if args.runs
        else DEFAULT_MANIFEST
    )
    rows = []
    for label, run_dir in manifest:
        d = Path(run_dir)
        if not (d / "summary.json").exists():
            print(f"skip {label}: no summary.json in {run_dir}")
            continue
        rows.append(row_for(label, d))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print(
            f"  {r['label']:32s} net_f1={r['net_f1']} tpF1={r['terminal_pair_f1']} "
            f"strict={r['strict_success']}"
        )


if __name__ == "__main__":
    main()
