#!/usr/bin/env python3
"""Run the full GT benchmark over verified topology files (Phase D / C1).

For each verified GT file: run the pipeline on its image, align pred→GT,
compute the topology metric cascade, then write base and repaired
netlists and record SPICE syntactic validity + DC-solvability before and
after the design-intent repair (C5). Aggregates with bootstrap 95% CIs.

Usage:
    python scripts/benchmark.py --split test
    python scripts/benchmark.py --gt-dir data/gt_netlists --include-unverified
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

from schematic2netlist.benchmark import aggregate, score_prediction
from schematic2netlist.config import config_hash, load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.netlist import export_spice_netlist
from schematic2netlist.nodes import bbox_xyxy  # noqa: F401 (kept for parity)
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.repair import build_ledger, export_ledger
from schematic2netlist.simulate import run_ngspice_diag


def pred_components(result: dict) -> list[dict]:
    """Pipeline output -> benchmark component format with center bboxes."""
    dets = result["detections"]
    out = []
    for c in result["components"]:
        det = dets[c["id"]]
        out.append({
            "id": c["id"],
            "class": c["class"],
            "nets": list(c.get("node_names", [])),
            "bbox": [det["x"], det["y"], det["width"], det["height"]],
        })
    return out


def gt_components(gt: dict) -> list[dict]:
    comps = gt_to_components(gt)
    by_id = {c["id"]: c for c in gt["components"]}
    for c in comps:
        c["bbox"] = by_id[c["id"]]["bbox"]
    return comps


def _solvable(comps, placeholders, extra_lines, cfg) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".sp", delete=False) as f:
        path = f.name
    export_spice_netlist(comps, path, placeholders=placeholders, extra_lines=extra_lines)
    ok, cat, _ = run_ngspice_diag(path, cfg)
    Path(path).unlink(missing_ok=True)
    return ok, cat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default="data/cleaned")
    ap.add_argument("--gt-dir", default="data/gt_netlists")
    ap.add_argument("--out-dir", default="results/benchmark")
    ap.add_argument("--config", default=None)
    ap.add_argument("--include-unverified", action="store_true",
                    help="score unverified GT too (default: verified only)")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--no-spice", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, cfg, seed, extra={"split": args.split})

    det_dir = Path(cfg["detect"]["cache_dir"])
    names = (Path(args.splits_dir) / f"{args.split}.txt").read_text().split()

    rows: list[dict] = []
    skipped: dict[str, int] = {}
    for name in names:
        stem = Path(name).stem
        gt_path = Path(args.gt_dir) / (stem + ".json")
        det_path = det_dir / (stem + ".json")
        if not gt_path.exists() or not det_path.exists():
            skipped["missing_gt_or_det"] = skipped.get("missing_gt_or_det", 0) + 1
            continue
        gt = load_gt(gt_path)
        if not gt.get("verified") and not args.include_unverified:
            skipped["unverified"] = skipped.get("unverified", 0) + 1
            continue

        detections = load_cached_detections(det_path)
        result = run_pipeline(Path(args.images_dir) / name, cfg, detections=detections)

        row = {"image": name}
        row.update(score_prediction(pred_components(result), gt_components(gt),
                                    iou_threshold=args.iou_threshold))

        if not args.no_spice:
            comps = result["components"]
            ph = cfg["netlist"]["placeholders"]
            rep = result.get("repair")
            base_ok, base_cat = _solvable(comps, ph, None, cfg)
            row["spice_valid"] = int(base_ok or base_cat not in ("parse_error",))
            row["solvable_before"] = int(base_ok)
            if rep is not None:
                rep_ok, _ = _solvable(comps, ph, rep.extra_lines, cfg)
                row["solvable_after"] = int(rep_ok)
                row["num_assumptions"] = rep.num_assumptions
                row["num_gauge"] = rep.num_gauge
                ledger = build_ledger(name, bool(base_ok), bool(rep_ok), rep)
                export_ledger(ledger, str(out_dir / "ledgers" / (stem + ".json")))
            else:
                row["solvable_after"] = int(base_ok)
        rows.append(row)

    # per-image CSV
    if rows:
        fields = list(rows[0].keys())
        with open(out_dir / "per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    summary = {
        "config_hash": config_hash(cfg),
        "split": args.split,
        "scored": len(rows),
        "skipped": skipped,
        "topology": aggregate(rows, seed=seed),
    }
    if rows and "solvable_after" in rows[0]:
        n = len(rows)
        summary["repair"] = {
            "solvable_before_rate": sum(r.get("solvable_before", 0) for r in rows) / n,
            "solvable_after_rate": sum(r.get("solvable_after", 0) for r in rows) / n,
            "mean_assumptions": sum(r.get("num_assumptions", 0) for r in rows) / n,
            "mean_gauge": sum(r.get("num_gauge", 0) for r in rows) / n,
            "spice_valid_rate": sum(r.get("spice_valid", 0) for r in rows) / n,
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] scored {len(rows)} image(s); skipped {skipped}")
    if rows:
        t = summary["topology"]
        print(f"  net F1:            {t['net_f1']['mean']:.3f} "
              f"[{t['net_f1']['ci95_lo']:.3f}, {t['net_f1']['ci95_hi']:.3f}]")
        print(f"  terminal-pair F1:  {t['terminal_pair_f1']['mean']:.3f}")
        print(f"  strict success:    {t['strict_success']['mean']:.3f}")
        print(f"  nGED (lower=better): {t['nged']['mean']:.3f}")
        if "repair" in summary:
            rp = summary["repair"]
            print(f"  solvable before→after: {rp['solvable_before_rate']:.3f} "
                  f"→ {rp['solvable_after_rate']:.3f} "
                  f"(mean {rp['mean_assumptions']:.2f} assumptions/circuit)")
    print(f"[OK] wrote {out_dir}/per_image.csv + summary.json")


if __name__ == "__main__":
    main()
