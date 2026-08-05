#!/usr/bin/env python3
"""The three-way hard core: circuits where the pipeline AND both VLMs miss the GT.

Builds the priority human-review queue. An image on which all three independent
systems fail strict success is worth a human's time for a reason that has
nothing to do with model quality: when three systems that share no code, no
lab and no training data all disagree with the annotation, one live hypothesis
is that the ANNOTATION is the outlier.

That is a hypothesis and this file ranks it, it does not settle it. Three other
readings of the same evidence are equally live:

  * the drawing is genuinely ambiguous, and the annotator resolved it one way
    while all three systems resolved it the other;
  * the three systems share a bias — the same bare-crossing convention, the
    same tolerance for a faint conductor — so their agreement is not
    independent evidence at all;
  * the circuit is simply hard and all three failed on their own merits.

Nothing here validates the ground truth. The union of the three systems clears
only ~57% of the split, which says the task is hard and that a specialised
pipeline is competitive with frontier general models. It says nothing about
whether the remaining annotations are right, and this queue must not be read
as an audit that found them right.

SPLIT. The VLM runs cover the 190 images that are TODAY called ``val``
(``data/splits/val.txt``); they have zero overlap with today's 192-image
``test`` split. They were submitted on 2026-08-01, before the 2026-08-03 role
swap documented in ``data/README.md``, when that same set of images was named
``test``. So every number this script emits is a VALIDATION-split number and is
labelled as such in the output. Reproducing it on ``test`` would need a fresh
paid VLM run over 192 unseen images and is not done here.

Note that ``config.benchmark.gt_dir`` now points at ``data/gt_test_1024``, so
it is the WRONG ground truth for these images. The default below is the val GT
directory, and a stem with no GT file is a hard error rather than a silent
skip.

Usage:
    python scripts/vlm_hardcore.py                        # writes the queue
    python scripts/vlm_hardcore.py --limit 5              # smoke test
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
sys.path.insert(0, str(ROOT / "scripts"))

from schematic2netlist.benchmark import (
    align_components,
    canonicalize_terminals,
    score_prediction,
)
from schematic2netlist.config import load_config
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import _terminal_pairs
from schematic2netlist.pipeline import run_pipeline

from score_vlm import pred_from_response
from vlm_task import load_detections

SYSTEMS = ["pipeline", "claude", "openai"]


def gt_components(stem: str, gt_dir: Path) -> list[dict]:
    """GT in the {id, class, nets, bbox} shape the metric cascade wants."""
    path = gt_dir / f"{stem}.json"
    if not path.exists():
        sys.exit(f"no GT for {stem} under {gt_dir}. If these images are the "
                 f"val split, pass --gt-dir data/gt_val_1024; the config's "
                 f"benchmark.gt_dir is the TEST split since the role swap.")
    gt = load_gt(str(path))
    comps = gt_to_components(gt)
    by_id = {c["id"]: c for c in gt["components"]}
    for c in comps:
        c["bbox"] = by_id[c["id"]]["bbox"]
    return comps


def pipeline_pred(stem: str, cfg, dets: list[dict]) -> list[dict]:
    """Run the pipeline on one frame and shape it like a VLM prediction."""
    frame = Path(cfg["preprocess"]["images_dir"]) / f"{stem}.jpg"
    res = run_pipeline(str(frame), cfg, detections=dets)
    out = []
    for c in res["components"]:
        d = res["detections"][c["id"]]
        out.append({"id": c["id"], "class": c["class"],
                    "nets": list(c.get("node_names", [])),
                    "bbox": [d["x"], d["y"], d["width"], d["height"]]})
    return out


def pair_set(pred: list[dict], gt: list[dict], iou: float) -> set:
    """Terminal pairs of one prediction, in GT's id space.

    Aligning to GT before taking pairs is what makes the three systems
    comparable to each other and not just to GT: after ``align_components`` a
    pair key is (gt_component_id, terminal_index), so the same key means the
    same physical pin pair for all three. The caveat is
    ``canonicalize_terminals``, which reorders terminals within a component by
    that system's OWN connectivity signature — so on a component where two
    systems disagree, their terminal indices can be permuted relative to each
    other. This is the same canonicalisation the reported metric uses, it
    applies to all three systems equally, and it can only understate agreement.
    """
    pred_a, gt_a, _ = align_components(pred, gt, iou)
    return _terminal_pairs(canonicalize_terminals(pred_a))


def pair_f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    return 2 * p * r / (p + r)


def render(pairs: set, limit: int = 6) -> str:
    """Disputed terminal pairs as 'c3.t0~c7.t1', shortest first, capped."""
    items = []
    for fs in pairs:
        (ca, ta), (cb, tb) = sorted(fs)
        items.append(f"c{ca}.t{ta}~c{cb}.t{tb}")
    items.sort()
    if len(items) > limit:
        return "; ".join(items[:limit]) + f"; +{len(items) - limit} more"
    return "; ".join(items)


def load_per_image(path: Path, rep: str | None = None) -> dict[str, dict]:
    rows = {}
    for r in csv.DictReader(path.open()):
        if rep and r.get("rep") and r["rep"] != rep:
            continue
        rows[r["image"]] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val",
                    help="label recorded in every output row. The VLM runs "
                         "cover val; nothing here re-runs a model.")
    ap.add_argument("--gt-dir", default="data/gt_val_1024",
                    help="GT for THIS split. Not config.benchmark.gt_dir, "
                         "which points at the test split.")
    ap.add_argument("--pipeline-csv",
                    default="results/benchmark_1024_final/seed0/per_image.csv")
    ap.add_argument("--claude-dir", default="results/vlm/claude_b")
    ap.add_argument("--openai-dir", default="results/vlm/openai_b")
    ap.add_argument("--rep", default="rep0",
                    help="only rep0 exists: each model was run ONCE, so no "
                         "cross-run variance is measurable here")
    ap.add_argument("--variant", choices=["a", "b"], default="b")
    ap.add_argument("--config", default=None)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--out-dir", default="results/vlm/analysis")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; smoke test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe_rows = load_per_image(Path(args.pipeline_csv))
    cl_rows = load_per_image(Path(args.claude_dir) / "scored/per_image.csv",
                             args.rep)
    op_rows = load_per_image(Path(args.openai_dir) / "scored/per_image.csv",
                             args.rep)

    images = sorted(set(pipe_rows) & set(cl_rows) & set(op_rows))
    if not images:
        sys.exit("no images common to all three per-image CSVs")

    # Guard the exact confusion this whole exercise exists to prevent: the
    # scored CSVs and the split manifest must be the same images.
    manifest = {l.strip() for l in
                (ROOT / f"data/splits/{args.split}.txt").open() if l.strip()}
    if set(images) - manifest:
        sys.exit(f"{len(set(images) - manifest)} scored images are NOT in "
                 f"data/splits/{args.split}.txt — wrong --split for these "
                 f"results. Check data/README.md, section 'the 2026-08-03 "
                 f"role swap'.")

    ok = lambda rows, img: rows[img]["strict_success"] == "True"  # noqa: E731
    groups = {
        "all_pass": [i for i in images
                     if ok(pipe_rows, i) and ok(cl_rows, i) and ok(op_rows, i)],
        "hard_core": [i for i in images
                      if not ok(pipe_rows, i) and not ok(cl_rows, i)
                      and not ok(op_rows, i)],
        "both_vlm_recoverable": [i for i in images if not ok(pipe_rows, i)
                                 and ok(cl_rows, i) and ok(op_rows, i)],
        "recoverable_by_at_least_one": [i for i in images
                                        if not ok(pipe_rows, i)
                                        and (ok(cl_rows, i) or ok(op_rows, i))],
        "union": [i for i in images if ok(pipe_rows, i) or ok(cl_rows, i)
                  or ok(op_rows, i)],
    }
    print(f"split={args.split}  n={len(images)}  "
          f"pipeline={sum(ok(pipe_rows, i) for i in images)} "
          f"claude={sum(ok(cl_rows, i) for i in images)} "
          f"openai={sum(ok(op_rows, i) for i in images)}")
    for k, v in groups.items():
        print(f"  {k:<28} {len(v)}")
    print(f"  union ceiling                {len(groups['union'])/len(images):.4f}")

    hard = groups["hard_core"]
    if args.limit:
        hard = hard[:args.limit]

    rows = []
    for n, img in enumerate(hard, 1):
        stem = Path(img).stem
        gt = gt_components(stem, gt_dir)
        dets = load_detections(stem, cfg)

        preds = {
            "pipeline": pipeline_pred(stem, cfg, dets),
            "claude": pred_from_response(
                json.loads((Path(args.claude_dir) / args.rep /
                            f"{stem}.json").read_text()), args.variant, dets),
            "openai": pred_from_response(
                json.loads((Path(args.openai_dir) / args.rep /
                            f"{stem}.json").read_text()), args.variant, dets),
        }
        # An unusable response scores as an empty prediction, exactly as
        # score_vlm.py does, so the queue and the reported metrics agree.
        preds = {k: (v if v is not None else []) for k, v in preds.items()}

        gt_pairs = pair_set(gt, gt, args.iou_threshold)
        pairs = {k: pair_set(v, gt, args.iou_threshold)
                 for k, v in preds.items()}
        scores = {k: score_prediction(v, gt, args.iou_threshold)
                  for k, v in preds.items()}
        n_nets = {k: len({net for c in v for net in c["nets"] if net})
                  for k, v in preds.items()}
        gt_nets = len({net for c in gt for net in c["nets"] if net})

        # What all three agree on, against GT. These are the pairs that carry
        # the annotation-error hypothesis: a small consensus delta is a
        # localised dispute, a large one is three systems failing at scale.
        consensus_extra = pairs["pipeline"] & pairs["claude"] & pairs["openai"]
        consensus_extra -= gt_pairs
        consensus_missing = gt_pairs - (
            pairs["pipeline"] | pairs["claude"] | pairs["openai"])

        inter = {
            "pipe_claude": pair_f1(pairs["pipeline"], pairs["claude"]),
            "pipe_openai": pair_f1(pairs["pipeline"], pairs["openai"]),
            "claude_openai": pair_f1(pairs["claude"], pairs["openai"]),
        }
        inter_mean = statistics.mean(inter.values())
        vs_gt_mean = statistics.mean(
            pair_f1(pairs[s], gt_pairs) for s in SYSTEMS)

        row = {
            "rank": 0,
            "image": img,
            "split": args.split,
            "gt_dir": str(gt_dir),
            "n_gt_components": len(gt),
            "n_gt_nets": gt_nets,
            "n_gt_terminal_pairs": len(gt_pairs),
            # The ranking key. High = the three systems agree with EACH OTHER
            # more than any of them agrees with the annotation, which is the
            # configuration in which the annotation is worth re-reading. It is
            # a pointer for a human, not a finding.
            "gt_outlier_margin": round(inter_mean - vs_gt_mean, 4),
            "inter_system_mean_f1": round(inter_mean, 4),
            "mean_vs_gt_f1": round(vs_gt_mean, 4),
            "consensus_extra": len(consensus_extra),
            "consensus_missing": len(consensus_missing),
            "consensus_delta": len(consensus_extra) + len(consensus_missing),
        }
        for s in SYSTEMS:
            row[f"{s}_tp_f1"] = round(scores[s]["terminal_pair_f1"], 4)
            row[f"{s}_net_f1"] = round(scores[s]["net_f1"], 4)
            row[f"{s}_nged"] = round(scores[s]["nged"], 4)
            row[f"{s}_unmatched_gt"] = scores[s]["unmatched_gt"]
            row[f"{s}_missing_pairs"] = len(gt_pairs - pairs[s])
            row[f"{s}_extra_pairs"] = len(pairs[s] - gt_pairs)
            row[f"{s}_n_nets"] = n_nets[s]
            row[f"{s}_net_delta"] = n_nets[s] - gt_nets
            m, e = len(gt_pairs - pairs[s]), len(pairs[s] - gt_pairs)
            row[f"{s}_mode"] = ("split" if m > e else
                                "weld" if e > m else "balanced")
        row["f1_pipe_claude"] = round(inter["pipe_claude"], 4)
        row["f1_pipe_openai"] = round(inter["pipe_openai"], 4)
        row["f1_claude_openai"] = round(inter["claude_openai"], 4)
        row["consensus_extra_pairs"] = render(consensus_extra)
        row["consensus_missing_pairs"] = render(consensus_missing)
        rows.append(row)
        print(f"  [{n}/{len(hard)}] {stem:<16} margin="
              f"{row['gt_outlier_margin']:+.3f} delta={row['consensus_delta']}",
              flush=True)

    # Most-suspicious annotation first; among equals, the smallest dispute
    # first, because a two-pair disagreement is the cheapest thing to re-read.
    rows.sort(key=lambda r: (-r["gt_outlier_margin"], r["consensus_delta"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    csv_path = out_dir / "hardcore_review_queue.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "what": "circuits where the pipeline and BOTH VLMs fail strict "
                "success — the priority human-review queue",
        "split": args.split,
        "split_manifest": f"data/splits/{args.split}.txt",
        "n_images_scored": len(images),
        "gt_dir": str(gt_dir),
        "warning": "This is a VALIDATION-split result. The VLM runs predate "
                   "the 2026-08-03 role swap (data/README.md) and cover the "
                   "190 images now called val, not the 192-image test split. "
                   "Reproducing it on test requires a new paid VLM run.",
        "reading": "A three-way disagreement is a pointer, not a verdict. It "
                   "is equally consistent with an ambiguous drawing, with a "
                   "bias the three systems share, or with three independent "
                   "failures. Nothing here validates the ground truth.",
        "n_repeats_available": 1,
        "sources": {
            "pipeline": args.pipeline_csv,
            "claude": f"{args.claude_dir}/{args.rep}",
            "openai": f"{args.openai_dir}/{args.rep}",
        },
        "groups": {k: len(v) for k, v in groups.items()},
        "union_ceiling": round(len(groups["union"]) / len(images), 4),
        "queue_size": len(rows),
        "ranking": "gt_outlier_margin = (mean pairwise terminal-pair F1 among "
                   "the three predictions) - (mean terminal-pair F1 of each "
                   "prediction against GT), descending; ties broken by the "
                   "smaller consensus_delta.",
    }
    (out_dir / "hardcore_review_queue.meta.json").write_text(
        json.dumps(meta, indent=1))

    print(f"\nwrote {csv_path}  ({len(rows)} circuits)")
    print(f"wrote {out_dir}/hardcore_review_queue.meta.json")
    print("\ntop 10 by gt_outlier_margin:")
    for r in rows[:10]:
        print(f"  {r['rank']:>2}. {r['image']:<18} margin="
              f"{r['gt_outlier_margin']:+.3f}  inter={r['inter_system_mean_f1']:.3f} "
              f"vs_gt={r['mean_vs_gt_f1']:.3f}  delta={r['consensus_delta']}")


if __name__ == "__main__":
    main()
