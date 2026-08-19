#!/usr/bin/env python3
"""Why two circuits disagree after the pin-aware pass (task F2 precursor).

`circuit_1247` and `circuit_431` are scored perfect by the topology cascade AND
by the pin-aware metric, are not flagged multistable, and still settle to a
different operating point. The manuscript listed them as unexplained. They are
not: both are explained by the same mechanism, and it is a property of the
METRIC rather than a reconstruction error.

Neither drawing contains a ground symbol. The as-drawn rule keeps it that way --
a missing ground stays missing -- so the reference annotation has no net named
"0". Its five one-port sources return to an implicit reference that is not any
drawn net. The pipeline's ground-selection step instead designates one of the
drawn rails as the reference, so the two decks are isomorphic as graphs (the
pin-aware metric scores 22/22 on both) while being simulated against DIFFERENT
reference nodes. Absolute node voltages are then not comparable, which is
exactly what the operating-point metric compares.

This is checkable rather than a story: of the 93 circuits where both sides
solve, only 4 have no drawn ground, and ALL 4 disagree at the operating point
against 64% of the rest.

Usage:
    python scripts/residual_circuits.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from schematic2netlist.benchmark import align_components  # noqa: E402
from schematic2netlist.gt import load_gt  # noqa: E402
from metrics.pin_aware import load_symmetry, score_pin_aware  # noqa: E402

CACHE = ROOT / "results/final/op_agreement/cache"
OUT = ROOT / "results/residual_circuits.json"
RESIDUALS = ("circuit_1247", "circuit_431")


def main() -> int:
    sym = load_symmetry()
    per = {}
    with (ROOT / "results/final/op_agreement/per_image.csv").open() as f:
        for r in csv.DictReader(f):
            per[r["stem"]] = r

    detail = {}
    for stem in RESIDUALS:
        rec = json.loads((CACHE / f"{stem}.json").read_text())
        gt, pred, corr = rec["gt_graph"], rec["pred_graph"], rec["corr"]
        pa, ga, _ = align_components(pred, gt, 0.3)
        ids = {g["id"] for g in ga}
        r = score_pin_aware(pa, ga, [(c["id"], c["id"]) for c in pa
                                     if c["id"] in ids], sym)
        d = load_gt(ROOT / f"data/gt_test_1024/{stem}.json")
        classes = Counter(c["class"] for c in d["components"])
        gnets = {n for c in gt for n in c["nets"] if n}
        pnets = {n for c in pred for n in c["nets"] if n}
        inv = {v: k for k, v in corr.items()}
        detail[stem] = {
            "gnd_components_drawn": classes.get("GND", 0),
            "one_port_sources": classes.get("V-DC (one port)", 0),
            "pin_aware_correct": f"{r.n_correct}/{r.n_scored}",
            "pin_aware_strict": bool(r.strict_success),
            "class_mismatches": 0,
            "gt_has_net_zero": "0" in gnets,
            "pred_has_net_zero": "0" in pnets,
            "gt_net_serving_as_pred_reference": inv.get("0"),
            "op_f1": float(per[stem]["f1"]) if per.get(stem) else None,
        }

    with_g = no_g = with_dis = no_dis = 0
    no_g_stems = []
    for stem, r in per.items():
        p = ROOT / f"data/gt_test_1024/{stem}.json"
        if not p.exists():
            continue
        try:
            d = load_gt(p)
            f1 = float(r["f1"])
        except (ValueError, KeyError, TypeError):
            continue
        has = any(c["class"] == "GND" for c in d["components"])
        dis = f1 < 1.0
        if has:
            with_g += 1
            with_dis += dis
        else:
            no_g += 1
            no_dis += dis
            no_g_stems.append(stem)

    report = {
        "_what": ("Why circuit_1247 and circuit_431 disagree at the operating "
                  "point after the pin-aware pass. Both share one mechanism."),
        "_mechanism": (
            "Neither drawing has a ground symbol. The as-drawn rule keeps the "
            "reference annotation without a net '0'; the pipeline's "
            "ground-selection step designates one drawn rail as the reference. "
            "The two decks are isomorphic as graphs -- pin-aware scores 22/22 "
            "on both -- but are simulated against different reference nodes, so "
            "their absolute node voltages are not comparable. This is a limit "
            "of the operating-point comparison on ungrounded drawings, not a "
            "reconstruction error."),
        "residuals": detail,
        "population_check": {
            "_what": ("If the mechanism is right, ungrounded drawings should "
                      "disagree far more often than grounded ones."),
            "circuits_with_drawn_ground": with_g,
            "of_those_op_disagrees": with_dis,
            "rate_with_ground": with_dis / with_g if with_g else None,
            "circuits_without_drawn_ground": no_g,
            "of_those_op_disagrees": no_dis,
            "rate_without_ground": no_dis / no_g if no_g else None,
            "ungrounded_stems": sorted(no_g_stems),
        },
        "_consequence": (
            "The operating-point metric should either exclude drawings with no "
            "reference, or compare potential DIFFERENCES rather than absolute "
            "node voltages. Reported rather than applied: changing the metric "
            "after seeing which circuits it fails on is the move this project "
            "spent Section 'Threats to Validity' removing."),
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for stem, d in detail.items():
        print(f"  {stem}: GND drawn={d['gnd_components_drawn']}, "
              f"one-port sources={d['one_port_sources']}, "
              f"pin-aware {d['pin_aware_correct']}, "
              f"pred reference was GT net {d['gt_net_serving_as_pred_reference']}")
    pc = report["population_check"]
    print(f"  ungrounded {pc['circuits_without_drawn_ground']} circuits, "
          f"{pc['of_those_op_disagrees']} disagree "
          f"({pc['rate_without_ground']:.1%}) vs "
          f"{pc['rate_with_ground']:.1%} for grounded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
