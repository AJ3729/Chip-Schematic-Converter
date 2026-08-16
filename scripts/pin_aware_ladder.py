#!/usr/bin/env python3
"""Full structural recomputation under pin identity (task D6).

Recomputes the structural results with terminal order intact and produces the
ladder table the plan asks for:

  pin-blind strict success            as published
  pin-aware strict success            the new headline
  of pin-blind perfect, pin-aware perfect
  of pin-aware perfect, operating point agreeing

Then the central finding is re-examined. Of the circuits the published metric
scored perfect, how many survive pin-aware scoring -- and how much of the
operating-point disagreement is thereby explained STRUCTURALLY rather than
left unexplained. Circuits still unexplained after that, and after D4's
multistability flags, are listed by id for human review.

Runs from stored artifacts: the op-agreement cache holds the aligned predicted
and reference graphs for every circuit, so nothing is re-simulated.

Usage:
    python scripts/pin_aware_ladder.py
    python scripts/pin_aware_ladder.py --seeds 0,1,2
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
sys.path.insert(0, str(ROOT))

from schematic2netlist.benchmark import align_components  # noqa: E402
from metrics.pin_aware import load_symmetry, score_pin_aware  # noqa: E402
from stats.bootstrap import bootstrap_rate  # noqa: E402
from stats.mcnemar import mcnemar_exact  # noqa: E402

CACHE = ROOT / "results/final/op_agreement/cache"
OPSUM = ROOT / "results/final/op_agreement/summary.json"
MULTI = ROOT / "results/multistability.json"
OUT = ROOT / "results/pin_aware_ladder.json"


def score_one(rec: dict, sym: dict) -> dict | None:
    gt, pred = rec.get("gt_graph"), rec.get("pred_graph")
    if not gt or not pred:
        return None
    # align_components RELABELS predicted ids into GT id space. The matched
    # pairs are therefore expressed in that space, so the relabelled lists must
    # be scored -- passing the originals silently pairs unrelated components
    # and scores every circuit as a failure.
    pa, ga, _ = align_components(pred, gt, 0.3)
    gt_ids = {g["id"] for g in ga}
    matched = [(c["id"], c["id"]) for c in pa if c["id"] in gt_ids]
    r = score_pin_aware(pa, ga, matched, sym)
    top = rec.get("topology") or {}
    return {
        "stem": rec["stem"],
        "pin_blind_strict": bool(top.get("strict_success")),
        "pin_aware_strict": bool(r.strict_success),
        "n_components": r.n_components,
        "n_correct": r.n_correct,
        "component_accuracy": r.component_accuracy,
        "per_class": r.per_class,
        "errors": r.errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    sym = load_symmetry()
    files = sorted(CACHE.glob("*.json"))
    files = files[: a.limit] if a.limit else files
    if not files:
        sys.exit(f"no op-agreement cache at {CACHE}")

    rows = []
    for f in files:
        r = score_one(json.loads(f.read_text()), sym)
        if r:
            rows.append(r)
    n = len(rows)

    blind = {r["stem"] for r in rows if r["pin_blind_strict"]}
    aware = {r["stem"] for r in rows if r["pin_aware_strict"]}

    op = json.loads(OPSUM.read_text())
    op_disagree = {x["stem"] for x in op["topologically_perfect_but_op_disagrees"]}
    both_solve = {r["stem"] for r in
                  csv.DictReader((ROOT / "results/final/op_agreement/per_image.csv").open())
                  if r["population"] == "both_solve"}

    ci_b = bootstrap_rate([r["pin_blind_strict"] for r in rows], seed=0)
    ci_a = bootstrap_rate([r["pin_aware_strict"] for r in rows], seed=0)
    mc = mcnemar_exact([r["pin_blind_strict"] for r in rows],
                       [r["pin_aware_strict"] for r in rows])

    # --- the central finding, re-examined ------------------------------------
    perfect = blind & both_solve
    lost = perfect - aware                      # pin-blind perfect, pin-aware NOT
    explained = lost & op_disagree               # ...and the op DID disagree
    still_unexplained = (perfect & op_disagree) - lost

    flagged = set()
    if MULTI.exists():
        flagged = set(json.loads(MULTI.read_text()).get("flagged") or [])

    per_class_tot: dict[str, list[int]] = {}
    for r in rows:
        for k, v in r["per_class"].items():
            t = per_class_tot.setdefault(k, [0, 0])
            t[0] += v["correct"]
            t[1] += v["total"]

    out = {
        "_what": "Structural results recomputed with terminal order intact. "
                 "Pin-aware strict success is the new headline structural "
                 "metric; it is strictly harder than the published one.",
        "_symmetry_spec": "spec/pin_symmetry.yaml (MOSFET drain/source ruled "
                          "ASYMMETRIC by the author, 2026-08-15)",
        "n_circuits": n,
        "ladder": {
            "pin_blind_strict_success": len(blind) / n,
            "pin_blind_ci95": [ci_b.lo, ci_b.hi],
            "pin_aware_strict_success": len(aware) / n,
            "pin_aware_ci95": [ci_a.lo, ci_a.hi],
            "of_pin_blind_perfect_also_pin_aware_perfect":
                len(blind & aware) / len(blind) if blind else None,
            "of_pin_aware_perfect_op_agrees":
                (len(aware & both_solve) - len(aware & op_disagree))
                / len(aware & both_solve) if (aware & both_solve) else None,
        },
        "pin_aware_is_a_subset_of_pin_blind": aware <= blind,
        "mcnemar_blind_vs_aware": {
            "pin_blind_only": mc.n_only_a, "pin_aware_only": mc.n_only_b,
            "p_exact": mc.p_value,
        },
        "mean_component_accuracy": statistics.mean(
            r["component_accuracy"] for r in rows),
        "per_class_component_accuracy": {
            k: {"correct": v[0], "total": v[1],
                "accuracy": v[0] / v[1] if v[1] else 0.0}
            for k, v in sorted(per_class_tot.items())},
        "central_finding": {
            "_domain": "circuits where both decks solve AND the published "
                       "metric scored them perfect",
            "n_pin_blind_perfect": len(perfect),
            "n_op_disagreed": len(perfect & op_disagree),
            "n_now_caught_structurally": len(explained),
            "fraction_of_disagreement_explained_by_pin_order":
                len(explained) / len(perfect & op_disagree)
                if (perfect & op_disagree) else None,
            "n_still_unexplained": len(still_unexplained),
            "n_still_unexplained_excluding_multistable":
                len(still_unexplained - flagged),
            "still_unexplained_ids": sorted(still_unexplained),
            "still_unexplained_excluding_multistable_ids":
                sorted(still_unexplained - flagged),
        },
        "per_circuit": rows,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    L = out["ladder"]
    print(f"circuits {n}\n")
    print("LADDER")
    print(f"  pin-blind strict success        {L['pin_blind_strict_success']:.4f}"
          f"  [{ci_b.lo:.4f}, {ci_b.hi:.4f}]")
    print(f"  pin-aware strict success        {L['pin_aware_strict_success']:.4f}"
          f"  [{ci_a.lo:.4f}, {ci_a.hi:.4f}]")
    print(f"  of pin-blind perfect, pin-aware {L['of_pin_blind_perfect_also_pin_aware_perfect']:.4f}"
          if L['of_pin_blind_perfect_also_pin_aware_perfect'] is not None else "")
    if L["of_pin_aware_perfect_op_agrees"] is not None:
        print(f"  of pin-aware perfect, op agrees {L['of_pin_aware_perfect_op_agrees']:.4f}")
    print(f"\n  pin-aware is a strict subset of pin-blind: "
          f"{out['pin_aware_is_a_subset_of_pin_blind']}")
    c = out["central_finding"]
    print(f"\nCENTRAL FINDING re-examined")
    print(f"  pin-blind perfect & both solve      {c['n_pin_blind_perfect']}")
    print(f"  of those, operating point disagreed {c['n_op_disagreed']}")
    print(f"  now caught structurally by pin order {c['n_now_caught_structurally']}"
          + (f"  ({c['fraction_of_disagreement_explained_by_pin_order']:.1%})"
             if c['fraction_of_disagreement_explained_by_pin_order'] is not None else ""))
    print(f"  still unexplained                   {c['n_still_unexplained']}")
    print(f"  ...excluding multistable            {c['n_still_unexplained_excluding_multistable']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
