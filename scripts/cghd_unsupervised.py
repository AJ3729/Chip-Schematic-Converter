#!/usr/bin/env python3
"""Everything measurable on CGHD without ground truth (task B6).

Run the frozen pipeline over the evaluable pool and record what needs no
reference netlist: SPICE validity, DC solvability before and after repair, the
declared-assumption budget, latency, and the repair-type mix -- broken down by
drafter and by capture.

These are NOT accuracy numbers. A deck that parses and solves can still be the
wrong circuit; on CGHD, where detection transfers at mAP@0.5 0.3445, it very
often is. They measure whether the pipeline *degrades gracefully* on a corpus
it was never trained for, which is a different and still useful question.

Predictions are stored for later scoring. Nothing here is inspected in a way
that could inform annotation.

Usage:
    python scripts/cghd_unsupervised.py
    python scripts/cghd_unsupervised.py --limit 40
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from schematic2netlist.config import load_config  # noqa: E402
from schematic2netlist.netlist import export_spice_netlist  # noqa: E402
from schematic2netlist.pipeline import run_pipeline  # noqa: E402
from schematic2netlist.simulate import run_ngspice_diag  # noqa: E402
from stats.bootstrap import bootstrap_rate  # noqa: E402

IMG = ROOT / "data/cghd_1024/images"
ANN = ROOT / "data/cghd_1024/annotations"
CACHE = ROOT / "data/cghd_1024/detections"
PRED = ROOT / "results/cghd_predictions"
OUT = ROOT / "results/cghd_unsupervised_metrics.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["preprocess"]["images_dir"] = "data/cghd_1024/images"
    cfg["detect"]["cache_dir"] = str(CACHE.relative_to(ROOT))
    PRED.mkdir(parents=True, exist_ok=True)

    files = sorted(ANN.glob("*.json"))
    files = files[: a.limit] if a.limit else files
    print(f"evaluable pool: {len(files)} images")

    rows: list[dict] = []
    for i, f in enumerate(files, 1):
        stem = f.stem
        meta = json.loads(f.read_text())
        dets_p = CACHE / f"{stem}.json"
        if not dets_p.exists():
            continue
        dets = json.loads(dets_p.read_text())
        t0 = time.perf_counter()
        try:
            res = run_pipeline(IMG / f"{stem}.jpg", cfg, detections=dets)
        except Exception as e:                                # noqa: BLE001
            rows.append({"stem": stem, "drafter": meta["drafter"],
                         "group": meta["drawing_group"],
                         "picture": meta["picture"],
                         "error": type(e).__name__})
            continue
        latency_ms = (time.perf_counter() - t0) * 1000.0

        comps = res.get("components") or []
        nets = {n for c in comps for n in (c.get("node_names") or [])
                if n is not None}
        # run_pipeline returns a RepairResult dataclass, not a dict.
        rep = res.get("repair")

        # write both decks and simulate each
        base_p = PRED / f"{stem}.base.sp"
        export_spice_netlist(comps, str(base_p))
        ok_b, cat_b, _ = run_ngspice_diag(str(base_p), cfg)
        extra = list(getattr(rep, "extra_lines", None) or [])
        rep_p = PRED / f"{stem}.repaired.sp"
        export_spice_netlist(comps, str(rep_p), extra_lines=extra)
        ok_a, cat_a, _ = run_ngspice_diag(str(rep_p), cfg)

        entries = list(getattr(rep, "entries", None) or [])
        rows.append({
            "stem": stem, "drafter": meta["drafter"],
            "group": meta["drawing_group"], "picture": meta["picture"],
            "n_components": len(comps), "n_nets": len(nets),
            "n_gt_boxes": len(meta["components"]),
            "latency_ms": round(latency_ms, 2),
            "spice_valid": cat_b != "parse_error",
            "solvable_before": bool(ok_b), "category_before": cat_b,
            "solvable_after": bool(ok_a), "category_after": cat_a,
            "n_assumptions": int(getattr(rep, "num_assumptions", 0) or 0),
            "n_gauge": int(getattr(rep, "num_gauge", 0) or 0),
            "issues": [getattr(e, "issue", None) if not isinstance(e, dict)
                       else e.get("issue") for e in entries],
        })
        if i % 50 == 0:
            print(f"  ...{i}/{len(files)}", flush=True)

    good = [r for r in rows if "error" not in r]
    n = len(good)
    if not n:
        sys.exit("no rows")

    def rate(k):
        return sum(1 for r in good if r[k]) / n

    by_drafter: dict[str, dict] = {}
    for d in sorted({r["drafter"] for r in good}):
        sub = [r for r in good if r["drafter"] == d]
        by_drafter[str(d)] = {
            "n": len(sub),
            "spice_valid": sum(r["spice_valid"] for r in sub) / len(sub),
            "solvable_before": sum(r["solvable_before"] for r in sub) / len(sub),
            "solvable_after": sum(r["solvable_after"] for r in sub) / len(sub),
            "mean_components": statistics.mean(r["n_components"] for r in sub),
        }
    by_picture: dict[str, dict] = {}
    for p in sorted({r["picture"] for r in good}):
        sub = [r for r in good if r["picture"] == p]
        by_picture[str(p)] = {
            "n": len(sub),
            "solvable_after": sum(r["solvable_after"] for r in sub) / len(sub),
            "mean_components": statistics.mean(r["n_components"] for r in sub),
        }

    lat = [r["latency_ms"] for r in good]
    ci_b = bootstrap_rate([r["solvable_before"] for r in good], seed=0)
    ci_a = bootstrap_rate([r["solvable_after"] for r in good], seed=0)
    hcd = json.loads((ROOT / "results/final/benchmark/seed0/summary.json").read_text())

    out = {
        "_what": "Ground-truth-free metrics on CGHD. NOT accuracy: a deck that "
                 "parses and solves can still be the wrong circuit, and with "
                 "detection at mAP@0.5 0.3445 it often is. This measures "
                 "graceful degradation, not correctness.",
        "cghd_version": 12,
        "n_images": n,
        "n_pipeline_errors": len(rows) - n,
        "spice_valid_rate": rate("spice_valid"),
        "solvable_before_rate": rate("solvable_before"),
        "solvable_before_ci95": [ci_b.lo, ci_b.hi],
        "solvable_after_rate": rate("solvable_after"),
        "solvable_after_ci95": [ci_a.lo, ci_a.hi],
        "mean_assumptions": statistics.mean(r["n_assumptions"] for r in good),
        "mean_components_predicted": statistics.mean(r["n_components"] for r in good),
        "mean_components_annotated": statistics.mean(r["n_gt_boxes"] for r in good),
        "latency_ms": {"mean": statistics.mean(lat),
                       "median": statistics.median(lat),
                       "p90": sorted(lat)[int(0.9 * len(lat))]},
        "failure_categories_before": dict(collections.Counter(
            r["category_before"] for r in good).most_common()),
        "failure_categories_after": dict(collections.Counter(
            r["category_after"] for r in good).most_common()),
        "repair_issue_mix": dict(collections.Counter(
            i for r in good for i in r["issues"]).most_common()),
        "by_drafter": by_drafter,
        "by_picture_index": by_picture,
        "digitize_hcd_reference": {
            "spice_valid_rate": hcd["repair"]["spice_valid_rate"],
            "solvable_before_rate": hcd["repair"]["solvable_before_rate"],
            "solvable_after_rate": hcd["repair"]["solvable_after_rate"],
            "mean_assumptions": hcd["repair"]["mean_assumptions"],
        },
        "per_image": rows,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"\nimages scored              {n}  (errors {len(rows)-n})")
    print(f"SPICE valid                {out['spice_valid_rate']:.4f}"
          f"   (Digitize-HCD {hcd['repair']['spice_valid_rate']:.4f})")
    print(f"DC solvable, pre-repair    {out['solvable_before_rate']:.4f}"
          f"   (Digitize-HCD {hcd['repair']['solvable_before_rate']:.4f})")
    print(f"DC solvable, post-repair   {out['solvable_after_rate']:.4f}"
          f"   (Digitize-HCD {hcd['repair']['solvable_after_rate']:.4f})")
    print(f"components predicted/annotated  "
          f"{out['mean_components_predicted']:.1f} / "
          f"{out['mean_components_annotated']:.1f}")
    print(f"latency median             {out['latency_ms']['median']:.1f} ms")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
