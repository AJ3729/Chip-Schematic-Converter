#!/usr/bin/env python3
"""HOW does connectivity fail? Welding, fragmenting, or losing terminals?

With snapping fixed, the oracle charges 0.3406 of per-component accuracy to
the wire/connectivity stage and 0.0028 to snapping, so this is where strict
success is won or lost. 'Improve connectivity' is not a plan; the failure
has to be named first.

The aggregate numbers are ambiguous on purpose-defeating grounds. Failing
images show terminal-pair precision AND recall both low (precision 0.13-0.19
with recall 0.43-0.73). Those pull in opposite directions: welding two nets
adds spurious pairs and lowers PRECISION while keeping recall high, since
every genuine pair stays inside the merged blob. Low recall means genuine
pairs are ABSENT — a net was split, or a terminal found no node at all.
Both happening at once means the two must be counted separately.

Ground-truth nets are matched to predicted nodes by Hungarian assignment on
shared terminals (the same correspondence :func:`net_level_metrics` uses),
and then every GT net is classified:

  clean        all its terminals on one predicted node, and that node
               carries nothing else
  welded       its node also carries terminals of OTHER GT nets
  split        its terminals are spread over several predicted nodes
  welded+split both at once
  lost         one or more of its terminals snapped to no node (None)

Reported per terminal-pair-precision bucket, because the population is
sharply bimodal (79 images at precision >= 0.9 supply all 67 strict
successes; the 111 below supply none) and a mean over both tells you
nothing about either.

The actionable quantity is the per-image COUNT of defects among images that
are close to the line. If a near-miss image carries one weld, one targeted
decision flips it to strict success; if it carries twelve, no local fix
will.

Usage:
    python scripts/diagnose_connectivity.py --limit 190
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import terminal_pair_metrics
from schematic2netlist.pipeline import run_pipeline


def classify(pred_comps, gt_comps):
    """Classify every GT net as clean / welded / split / lost.

    Both lists must be aligned AND canonicalized before they get here.
    A terminal is identified by (component id, terminal index), so the two
    lists have to agree on what index *k* of a component means. Raw pipeline
    order does not: roughly a quarter of components carry the right nets in a
    different arrangement, and comparing index-to-index then makes a perfectly
    predicted net look simultaneously split and welded. Passing the
    un-canonicalized lists reported 58.5% welded+split for images scoring
    0.98 terminal-pair F1, which is how the mistake was caught.
    ``canonicalize_terminals`` is what makes the indices comparable, and it is
    also exactly what the benchmark scores, so this measures the same view.
    """
    gt_terms = defaultdict(list)      # gt net -> [(cid, k), ...]
    lost = defaultdict(int)           # gt net -> terminals with no node
    pred_of = {}                      # (cid, k) -> predicted net or None
    for c in pred_comps:
        for k, n in enumerate(c["nets"]):
            pred_of[(c["id"], k)] = n
    for c in gt_comps:
        for k, net in enumerate(c["nets"]):
            if net is None:
                continue
            t = (c["id"], k)
            gt_terms[net].append(t)
            # a GT terminal with no counterpart slot in the prediction is as
            # lost as one that snapped to nothing: the pipeline can report
            # fewer terminals than the class has when the class's terminal
            # count disagrees with GT's
            if pred_of.get(t) is None:
                lost[net] += 1

    pred_terms = defaultdict(list)
    for t, n in pred_of.items():
        if n is not None:
            pred_terms[n].append(t)

    gt_names = sorted(gt_terms)
    pr_names = sorted(pred_terms)
    # Hungarian on shared-terminal overlap: which predicted node IS this net?
    corr = {}
    if gt_names and pr_names:
        cost = np.zeros((len(gt_names), len(pr_names)))
        for i, gn in enumerate(gt_names):
            gs = set(gt_terms[gn])
            for j, pn in enumerate(pr_names):
                cost[i, j] = -len(gs & set(pred_terms[pn]))
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] < 0:            # must share at least one terminal
                corr[gt_names[i]] = pr_names[j]
    owner = {v: k for k, v in corr.items()}

    out = Counter()
    detail = []
    for gn in gt_names:
        pn = corr.get(gn)
        terms = gt_terms[gn]
        nodes = {pred_of.get(t) for t in terms} - {None}
        is_lost = lost[gn] > 0
        is_split = len(nodes) > 1
        # welded: this net's matched node carries terminals belonging to a
        # DIFFERENT gt net
        is_welded = False
        n_foreign = 0
        if pn is not None:
            foreign = [t for t in pred_terms[pn] if t not in set(terms)]
            n_foreign = len(foreign)
            is_welded = n_foreign > 0
        if pn is None:
            kind = "unmatched"
        elif is_welded and is_split:
            kind = "welded+split"
        elif is_welded:
            kind = "welded"
        elif is_split:
            kind = "split"
        elif is_lost:
            kind = "lost_terminal"
        else:
            kind = "clean"
        out[kind] += 1
        detail.append({"gt_net": gn, "kind": kind, "n_terminals": len(terms),
                       "n_pred_nodes": len(nodes), "n_foreign": n_foreign,
                       "n_lost": lost[gn]})
    return out, detail


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=190)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/connectivity_diag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out, cfg, seed)

    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)

    rows = []
    for i, nm in enumerate(names, 1):
        stem = nm.rsplit(".", 1)[0]
        gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        gcomps = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gcomps:
            c["bbox"] = by[c["id"]]["bbox"]
        dets = load_cached_detections(
            f"{cfg['detect']['cache_dir']}/{stem}.json",
            min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(images_dir / nm, cfg, detections=dets)
        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [res["detections"][c["id"]]["x"],
                          res["detections"][c["id"]]["y"],
                          res["detections"][c["id"]]["width"],
                          res["detections"][c["id"]]["height"]]}
                for c in res["components"]]
        p, g, stats = align_components(pred, gcomps)
        pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)
        tp = terminal_pair_metrics(pc, gc)
        kinds, _detail = classify(pc, gc)
        n_nets = sum(kinds.values())
        rows.append({
            "image": nm, "n_gt_nets": n_nets,
            "tp_precision": round(tp["precision"], 4),
            "tp_recall": round(tp["recall"], 4),
            "tp_f1": round(tp["f1"], 4),
            "unmatched_gt": stats["unmatched_gt"],
            **{k: kinds.get(k, 0) for k in
               ("clean", "welded", "split", "welded+split",
                "lost_terminal", "unmatched")},
        })
        if i % 20 == 0:
            print(f"[{i}/{len(names)}]", flush=True)

    KINDS = ("clean", "welded", "split", "welded+split",
             "lost_terminal", "unmatched")
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]

    print("\n=== GT NETS BY FATE, per terminal-pair-precision bucket ===")
    print("(share of that bucket's GT nets; 'welded' = its node also carries")
    print(" another net's terminals, 'split' = its terminals landed on "
          "several nodes)\n")
    hdr = f"  {'precision':>11s} {'imgs':>4s} {'nets':>5s} " + \
        " ".join(f"{k[:9]:>9s}" for k in KINDS)
    print(hdr)
    bucket_rows = {}
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["tp_precision"] < hi]
        if not sel:
            continue
        tot = sum(r["n_gt_nets"] for r in sel) or 1
        bucket_rows[f"[{lo},{min(hi,1.0)})"] = {
            "n_images": len(sel), "n_gt_nets": tot,
            **{k: round(sum(r[k] for r in sel) / tot, 4) for k in KINDS},
            "mean_defects_per_image": round(
                st.mean(r["welded"] + r["split"] + r["welded+split"]
                        + r["lost_terminal"] for r in sel), 2),
        }
        print(f"  [{lo:.1f},{min(hi,1.0):.1f}) {len(sel):4d} {tot:5d} " +
              " ".join(f"{sum(r[k] for r in sel)/tot:9.1%}" for k in KINDS))

    print("\n=== HOW MANY DEFECTS DOES A NEAR-MISS IMAGE CARRY? ===")
    print("This decides whether local repair is viable. One or two defects on")
    print("an otherwise-correct image means a single decision flips it to")
    print("strict success.\n")
    print(f"  {'precision':>11s} {'imgs':>4s} {'mean defects':>13s} "
          f"{'median':>7s} {'imgs with <=2':>14s}")
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["tp_precision"] < hi]
        if not sel:
            continue
        d = [r["welded"] + r["split"] + r["welded+split"] + r["lost_terminal"]
             for r in sel]
        print(f"  [{lo:.1f},{min(hi,1.0):.1f}) {len(sel):4d} {st.mean(d):13.2f} "
              f"{st.median(d):7.1f} {sum(1 for x in d if x <= 2):8d}"
              f" ({sum(1 for x in d if x <= 2)/len(sel):5.1%})")

    tot_all = sum(r["n_gt_nets"] for r in rows) or 1
    print(f"\n=== OVERALL ({len(rows)} images, {tot_all} GT nets) ===")
    for k in KINDS:
        n = sum(r[k] for r in rows)
        print(f"  {k:14s} {n:5d}  {n/tot_all:6.1%}")
    weld = sum(r["welded"] + r["welded+split"] for r in rows)
    split = sum(r["split"] + r["welded+split"] for r in rows)
    print(f"\n  nets touched by WELDING : {weld:5d}  {weld/tot_all:6.1%}")
    print(f"  nets touched by SPLITTING: {split:5d}  {split/tot_all:6.1%}")
    print(f"  -> the larger of these is the mechanism to attack first")

    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "n_images": len(rows), "n_gt_nets": tot_all,
        "overall": {k: {"n": sum(r[k] for r in rows),
                        "rate": round(sum(r[k] for r in rows)/tot_all, 4)}
                    for k in KINDS},
        "nets_touched_by_welding": {"n": weld, "rate": round(weld/tot_all, 4)},
        "nets_touched_by_splitting": {"n": split, "rate": round(split/tot_all, 4)},
        "by_precision_bucket": bucket_rows,
    }, indent=2) + "\n")
    print(f"\nwrote {out}/per_image.csv + summary.json")


if __name__ == "__main__":
    main()
