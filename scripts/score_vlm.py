#!/usr/bin/env python3
"""Score cached VLM responses through the pipeline's own metric cascade.

The comparison is only worth something if nothing but the method changes, so
this calls ``benchmark.score_prediction`` and ``benchmark.aggregate`` — the
exact functions ``scripts/benchmark.py`` uses — against the same verified GT at
the same IoU threshold. No metric is reimplemented here.

Variant B additionally makes component alignment a non-issue: the model is
handed our detections and returns their ids, so pred and GT align through the
identical Hungarian match the pipeline gets. Any difference in the numbers is
connectivity, not matching.

Reports mean +/- SD across repeats, and the per-image join against the
pipeline's own run so you can ask the question that matters: does the VLM fail
on the SAME images we do?

Usage:
    python scripts/score_vlm.py --run-dir results/vlm/claude_b --variant b
    python scripts/score_vlm.py --run-dir results/vlm/openai_b --variant b

Provider-agnostic: both runners write the same cached shape, so there is no
per-provider branch here and neither model gets a scoring advantage.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.benchmark import aggregate, score_prediction
from schematic2netlist.classes import canonical_class, class_terminals
from schematic2netlist.config import load_config
from schematic2netlist.gt import gt_to_components, load_gt

sys.path.insert(0, str(ROOT / "scripts"))
from vlm_task import load_detections  # same class head the pipeline applies

METRICS = ["terminal_pair_f1", "net_f1", "per_component_connected_acc",
           "nged", "strict_success"]


def pred_from_response(res: dict, variant: str, dets: list) -> list[dict] | None:
    """Turn one cached response into the {id, class, nets, bbox} shape
    score_prediction expects. Returns None if the response is unusable."""
    if "error" in res or "components" not in res:
        return None
    out = []
    if variant == "b":
        by_id = {c["id"]: c for c in res["components"] if isinstance(c.get("id"), int)}
        for i, d in enumerate(dets):
            cls = canonical_class(d["class"])
            n = class_terminals(cls)
            if n == 0:            # Wire Crossover: annotation, never in GT
                continue
            got = by_id.get(i, {}).get("terminals", [])
            # Pad or trim to the class's terminal count. A short answer is a
            # wrong answer, not a crash — score it rather than dropping it.
            nets = [str(x) for x in got[:n]] + [None] * max(0, n - len(got))
            out.append({"id": i, "class": cls, "nets": nets,
                        "bbox": [d["x"], d["y"], d["width"], d["height"]]})
    else:
        for i, c in enumerate(res["components"]):
            cls = canonical_class(c.get("class", ""))
            n = class_terminals(cls)
            bb = c.get("bbox") or [0, 0, 0, 0]
            if len(bb) != 4:
                continue
            got = c.get("terminals", [])
            nets = [str(x) for x in got[:n]] + [None] * max(0, n - len(got))
            out.append({"id": i, "class": cls, "nets": nets,
                        "bbox": [float(v) for v in bb]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", choices=["a", "b"], default="b")
    ap.add_argument("--config", default=None)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--pipeline-csv",
                    default="results/benchmark_1024_final/seed0/per_image.csv",
                    help="per-image CSV to join against for the same-images check")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = Path(args.run_dir)
    reps = sorted(p for p in run.glob("rep*") if p.is_dir())
    if not reps:
        sys.exit(f"no rep* directories under {run}")

    per_rep, per_image_rows = [], []
    for rep in reps:
        rows, n_bad = [], 0
        for f in sorted(rep.glob("*.json")):
            stem = f.stem
            gp = Path(cfg["benchmark"]["gt_dir"]) / f"{stem}.json"
            if not gp.exists():
                continue
            gt = load_gt(str(gp))
            gc = gt_to_components(gt)
            by = {c["id"]: c for c in gt["components"]}
            for c in gc:
                c["bbox"] = by[c["id"]]["bbox"]
            dets = load_detections(stem, cfg)
            pred = pred_from_response(json.loads(f.read_text()), args.variant, dets)
            if pred is None:
                n_bad += 1
                # An unusable response is a failure, not a missing sample —
                # scoring an empty prediction keeps the denominator honest.
                pred = []
            m = score_prediction(pred, gc, args.iou_threshold)
            m["image"] = f"{stem}.jpg"
            rows.append(m)
            per_image_rows.append({"rep": rep.name, **m})
        agg = aggregate(rows)
        # aggregate() returns a FLAT dict; the "topology" nesting is added by
        # scripts/benchmark.py, not by the library.
        per_rep.append({"rep": rep.name, "n": len(rows), "unusable": n_bad,
                        "metrics": agg})
        print(f"{rep.name}: n={len(rows)} unusable={n_bad}  "
              + "  ".join(f"{k}={agg[k]['mean']:.4f}"
                          for k in METRICS if k in agg))

    out = Path(args.out_dir or run / "scored")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"run_dir": str(run), "variant": args.variant,
               "n_repeats": len(per_rep), "per_repeat": per_rep,
               "across_repeats": {}}
    print(f"\nacross {len(per_rep)} repeats (mean +/- SD):")
    for k in METRICS:
        vals = [r["metrics"][k]["mean"] for r in per_rep if k in r["metrics"]]
        if not vals:
            continue
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary["across_repeats"][k] = {"mean": statistics.mean(vals), "sd": sd}
        print(f"  {k:<32} {statistics.mean(vals):.4f} +/- {sd:.4f}")

    # The comparison that matters for the ceiling claim: same images?
    pcsv = Path(args.pipeline_csv)
    if pcsv.exists():
        pipe = {r["image"]: r["strict_success"] == "True"
                for r in csv.DictReader(open(pcsv))}
        vlm: dict[str, list[bool]] = {}
        for r in per_image_rows:
            vlm.setdefault(r["image"], []).append(bool(r["strict_success"]))
        both = {k: v for k, v in vlm.items() if k in pipe}
        v_ok = {k: sum(v) > len(v) / 2 for k, v in both.items()}   # majority
        a = sum(1 for k in both if pipe[k] and v_ok[k])
        b = sum(1 for k in both if pipe[k] and not v_ok[k])
        c = sum(1 for k in both if not pipe[k] and v_ok[k])
        d = sum(1 for k in both if not pipe[k] and not v_ok[k])
        summary["agreement_vs_pipeline"] = {
            "both_strict": a, "pipeline_only": b, "vlm_only": c, "neither": d,
            "n": len(both)}
        print(f"\nstrict success vs the pipeline, {len(both)} images "
              f"(VLM by majority of repeats):")
        print(f"  both succeed      {a}")
        print(f"  pipeline only     {b}")
        print(f"  VLM only          {c}   <- headroom the pipeline is missing")
        print(f"  neither           {d}   <- corroborates an information limit")

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    with (out / "per_image.csv").open("w", newline="") as fh:
        keys = sorted({k for r in per_image_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(per_image_rows)
    print(f"\nwrote {out}/summary.json + per_image.csv")


if __name__ == "__main__":
    main()
