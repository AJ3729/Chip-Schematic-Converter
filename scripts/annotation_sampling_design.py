#!/usr/bin/env python3
"""Design the CGHD annotation subset (task B8).

Optimises evidence per hour of human effort. The target is 40-60 distinct
DRAWINGS, not images: each drawing carries up to four photographs, so one
annotated netlist scores several images through the B7 grouping.

THE RULE THAT MATTERS: selection is blind to whether the pipeline gets a
circuit right. It uses only drafter, complexity, and capture properties. If
correctness entered the selection the evaluation would be biased, and the bias
would favour us.

Stratification, in priority order:
  drafter      maximise distinct drafters -- this is the property
               Digitize-HCD cannot supply and CGHD can
  complexity   component-count deciles, taken from the ANNOTATION count, not
               from detector output (detector output is a pipeline product and
               using it would leak)
  capture      picture index within the drawing

The queue is ordered so that partial completion is still balanced: the first
10, first 20 and first 40 drawings are each independently stratified. Someone
who annotates for one evening and stops still has a usable sample.

Usage:
    python scripts/annotation_sampling_design.py
    python scripts/annotation_sampling_design.py --target 50 --seed 20260815
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ANN = ROOT / "data/cghd_1024/annotations"
COV = ROOT / "results/cghd_coverage.json"
OUT_Q = ROOT / "data/cghd/annotation_queue.json"
OUT_R = ROOT / "reports/pending_review/sampling_design.md"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260815)
    a = ap.parse_args()

    cov = json.loads(COV.read_text())
    # Netlist pool: annotating a circuit the pipeline cannot legally be scored
    # on would waste the effort.
    pool = set(cov["evaluable_images_netlist"])

    drawings: dict[str, dict] = {}
    for f in sorted(ANN.glob("*.json")):
        d = json.loads(f.read_text())
        key = f"{'drafter_' + str(d['drafter'])}/{f.stem.split('__',1)[1]}"
        if key.replace(f"drafter_{d['drafter']}/", f"drafter_{d['drafter']}/") not in pool:
            # the coverage manifest keys are drafter_N/CX_DY_PZ
            if f"drafter_{d['drafter']}/{f.stem.split('__', 1)[1]}" not in pool:
                continue
        g = d["drawing_group"]
        rec = drawings.setdefault(g, {
            "drawing_group": g, "drafter": d["drafter"],
            "captures": [], "n_components": d and len(d["components"])})
        rec["captures"].append(f.stem)
        rec["n_components"] = max(rec["n_components"], len(d["components"]))

    if not drawings:
        sys.exit("no evaluable drawings; run the adapter and coverage first")

    counts = sorted(r["n_components"] for r in drawings.values())
    deciles = [statistics.quantiles(counts, n=10)[i] for i in range(9)] \
        if len(counts) >= 10 else []

    def decile(n: int) -> int:
        return sum(1 for d in deciles if n > d)

    for r in drawings.values():
        r["complexity_decile"] = decile(r["n_components"])
        r["n_captures"] = len(r["captures"])

    rng = random.Random(a.seed)
    items = sorted(drawings.values(), key=lambda r: r["drawing_group"])
    rng.shuffle(items)

    # Round-robin over drafters, then over complexity deciles inside a drafter.
    # This is what makes every prefix of the queue balanced rather than only
    # the whole thing.
    by_drafter: dict[int, list] = collections.defaultdict(list)
    for r in items:
        by_drafter[r["drafter"]].append(r)
    for v in by_drafter.values():
        v.sort(key=lambda r: (r["complexity_decile"], r["drawing_group"]))
        rng.shuffle(v)

    queue: list[dict] = []
    drafters = sorted(by_drafter)
    rng.shuffle(drafters)
    idx = {d: 0 for d in drafters}
    while len(queue) < min(a.target, len(items)):
        progressed = False
        for d in drafters:
            if idx[d] < len(by_drafter[d]) and len(queue) < a.target:
                queue.append(by_drafter[d][idx[d]])
                idx[d] += 1
                progressed = True
        if not progressed:
            break

    def balance(prefix: list[dict]) -> dict:
        return {
            "n_drawings": len(prefix),
            "n_images_scored": sum(r["n_captures"] for r in prefix),
            "distinct_drafters": len({r["drafter"] for r in prefix}),
            "complexity_deciles_covered": len({r["complexity_decile"] for r in prefix}),
            "median_components": statistics.median(
                [r["n_components"] for r in prefix]) if prefix else None,
        }

    out = {
        "_what": "Annotation queue for CGHD. Ordered so the first 10, 20 and "
                 "40 drawings are each independently stratified.",
        "_selection_rule": "Blind to pipeline correctness. Stratified by "
                           "drafter (round robin), then complexity decile from "
                           "the ANNOTATION component count -- never from "
                           "detector output, which would leak a pipeline "
                           "product into the sample.",
        "seed": a.seed,
        "reproducible": "same seed + same coverage file -> same queue",
        "pool_drawings_available": len(drawings),
        "target": a.target,
        "queue_length": len(queue),
        "balance_at_prefixes": {
            "first_10": balance(queue[:10]),
            "first_20": balance(queue[:20]),
            "first_40": balance(queue[:40]),
            "full": balance(queue),
        },
        "double_annotation_fraction": 0.15,
        "double_annotation_drawings": [r["drawing_group"] for r in
                                       queue[::max(1, int(1 / 0.15))]],
        "queue": [{"rank": i + 1, **{k: r[k] for k in
                                     ("drawing_group", "drafter", "n_captures",
                                      "n_components", "complexity_decile",
                                      "captures")}}
                  for i, r in enumerate(queue)],
    }
    OUT_Q.parent.mkdir(parents=True, exist_ok=True)
    OUT_Q.write_text(json.dumps(out, indent=1) + "\n")

    b = out["balance_at_prefixes"]
    lines = [
        "# CGHD annotation sampling design (task B8)",
        "",
        "**For review before annotation begins.**",
        "",
        f"Queue: `data/cghd/annotation_queue.json`, seed {a.seed}, "
        f"{len(queue)} drawings from a pool of {len(drawings)}.",
        "",
        "## The rule",
        "",
        "Selection is **blind to whether the pipeline gets a circuit right**.",
        "It uses drafter, component count and capture index only. Component",
        "count comes from the *annotation*, never from detector output —",
        "using the detector would leak a pipeline product into the sample and",
        "bias the evaluation in our favour.",
        "",
        "Drawings, not images: each drawing carries up to four photographs, so",
        "one annotated netlist scores several images through the capture",
        "grouping.",
        "",
        "## Balance at every prefix",
        "",
        "| prefix | drawings | images scored | drafters | complexity deciles |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in ("first_10", "first_20", "first_40", "full"):
        v = b[k]
        lines.append(f"| {k.replace('_',' ')} | {v['n_drawings']} | "
                     f"{v['n_images_scored']} | {v['distinct_drafters']} | "
                     f"{v['complexity_deciles_covered']} |")
    lines += [
        "",
        "Stopping after any of these prefixes still leaves a stratified",
        "sample. That is the point of the ordering.",
        "",
        "## Double annotation",
        "",
        f"Every 6th drawing ({out['double_annotation_fraction']:.0%}) is",
        "re-queued after a delay for self-agreement measurement (task E4).",
        "",
        "## Caveat the reviewer should weigh",
        "",
        "Cross-corpus detection transfers at mAP@0.5 0.3445, and the pipeline",
        "localises 0.486 of components. Netlist-level scores on this sample",
        "will therefore be dominated by missing components rather than by",
        "wire-tracing errors. The sample is still worth annotating — it is the",
        "only way to put a number on cross-corpus reconstruction — but it will",
        "measure the detector more than the tracer, and the effort should be",
        "budgeted with that in mind.",
    ]
    OUT_R.write_text("\n".join(lines) + "\n")

    print(f"pool {len(drawings)} drawings -> queue {len(queue)}")
    for k in ("first_10", "first_20", "first_40", "full"):
        v = b[k]
        print(f"  {k:9s} drawings={v['n_drawings']:3d} images={v['n_images_scored']:3d} "
              f"drafters={v['distinct_drafters']:2d} deciles={v['complexity_deciles_covered']}")
    print(f"\nwrote {OUT_Q}\n      {OUT_R}")


if __name__ == "__main__":
    main()
