#!/usr/bin/env python3
"""The 494-swap control, re-run under the multi-condition criterion (task D5).

The published acceptance test injects one pin-pair swap at a time into the
ground truth and asks whether the operating-point metric notices. Pooled
detection is 0.7206, and the op-amp row is 0.2289 -- which the paper explains
with a theorem: if the two swapped terminals sit at the SAME POTENTIAL in the
unperturbed solution, that solution still satisfies the perturbed circuit, so no
tolerance, correspondence or placeholder policy can separate the two decks.

That theorem is about ONE operating point. Two terminals equipotential at one
bias, under one set of component values, at DC, need not be equipotential at
another bias, under other values, or at 1 kHz where reactances are finite. The
multi-condition criterion of task D5 therefore predicts a HIGHER detection rate,
and the size of the improvement is itself a result: it measures how much of the
published blind spot was a property of the probe rather than of the circuit.

DETECTION UNDER A STRICTER AGREEMENT CRITERION IS A UNION, NOT AN INTERSECTION.
D5 defines agreement as agreement under every condition. A swap therefore counts
as detected if it is caught under AT LEAST ONE condition -- the two statements
are duals, and getting them backwards would report the opposite result.

Conditions, all applied identically to both decks so component values cancel:
  op          the published probe, reproducing the published rate as a control
  values      K E24 component-value assignments
  dc_sweep    the same .op at five supply voltages
  ac          .ac at 1 kHz, node magnitude
  tran        step stimulus, three samples through the settle

CONTROLS THAT MUST HOLD. The passive control (resistor/capacitor/inductor swaps)
must stay at exactly zero detection under every condition -- those terminals
really are interchangeable, and a criterion that "detects" them is broken, not
sensitive. And the op condition must reproduce the published per-kind rates,
or the harness is measuring something other than what it claims.

Output goes to results/sim_coverage/, gitignored pending review.

Usage:
    python scripts/swap_detection_multicondition.py
    python scripts/swap_detection_multicondition.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from schematic2netlist.config import load_config  # noqa: E402
from schematic2netlist.classes import class_role  # noqa: E402
from measure_op_agreement import (  # noqa: E402
    PRIMARY_ATOL,
    PRIMARY_RTOL,
    SWAP_SPEC,
    _run_ngspice,
    build_deck,
    policy_placeholders,
    score_op,
    spice_components,
    swap_candidates,
    swapped,
    swapped_nets,
)
from multi_condition_agreement import (  # noqa: E402
    AC_ATOL,
    AC_PROBE,
    AC_RTOL,
    parse_ac_magnitudes,
    random_placeholders,
)
from simulation_coverage import (  # noqa: E402
    SWEEP_VOLTS,
    TRAN_ATOL,
    TRAN_RTOL,
    TRAN_SAMPLES,
    parse_meas,
    tran_probe,
)
from stats.bootstrap import bootstrap_rate  # noqa: E402

CACHE = ROOT / "results/final/op_agreement/cache"
OUT = ROOT / "results/sim_coverage/swap_detection_multicondition.json"

# Passive classes whose terminal order genuinely carries no meaning. The control
# population: detection here must be exactly zero under every condition.
PASSIVE_SPEC = {"resistor": (0, 1, "passive_control"),
                "capacitor": (0, 1, "passive_control"),
                "inductor": (0, 1, "passive_control")}

K_VALUES = 5          # E24 assignments; 5 keeps 494 swaps x 14 probes tractable
CONDITIONS = ("op", "values", "dc_sweep", "ac", "tran")


def _op(graph, ph, cfg):
    r = _run_ngspice(build_deck(spice_components(graph), ph), cfg)
    return (bool(r["solved"]), r["voltages"])


def _ac(graph, ph, cfg):
    r = _run_ngspice(build_deck(spice_components(graph), ph,
                                extra_lines=list(AC_PROBE)), cfg)
    v = parse_ac_magnitudes(r["stdout"])
    return (bool(v), v)


def _tran(graph, ph, cfg, nodes, t):
    tph = dict(ph)
    tph["dc_supply"] = "PULSE(0 15 0 10u 10u 1m 2m)"
    tph["ac_supply"] = tph["dc_supply"]
    r = _run_ngspice(build_deck(spice_components(graph), tph,
                                extra_lines=tran_probe(nodes, t)), cfg)
    v = parse_meas(r["stdout"])
    return (bool(v), v)


def moved(ref_ok, ref_v, per_ok, per_v, atol, rtol) -> bool | None:
    """Did the perturbation move this probe? None when the probe cannot see.

    A swap that makes the deck unsolvable counts as detected: the circuit leaves
    the comparable population, which is exactly what the published test does.
    """
    if not ref_ok:
        return None                      # reference could not be probed
    if not per_ok:
        return True                      # broke solvability
    ident = {k: k for k in ref_v}
    return score_op(ref_v, per_v, ident, atol=atol, rtol=rtol)["f1"] < 1.0 - 1e-12


def probe_swap(graph, cand, cfg, base_ph, value_phs) -> dict:
    """Per-condition detection for one injected swap."""
    per = swapped(graph, cand)
    out: dict = {}

    ok_r, v_r = _op(graph, base_ph, cfg)
    ok_p, v_p = _op(per, base_ph, cfg)
    out["op"] = moved(ok_r, v_r, ok_p, v_p, PRIMARY_ATOL, PRIMARY_RTOL)

    hits = []
    for ph in value_phs:
        a, b = _op(graph, ph, cfg), _op(per, ph, cfg)
        hits.append(moved(a[0], a[1], b[0], b[1], PRIMARY_ATOL, PRIMARY_RTOL))
    out["values"] = None if all(h is None for h in hits) else any(h for h in hits)

    hits = []
    for v in SWEEP_VOLTS:
        ph = {**base_ph, "dc_supply": f"DC {v}", "ac_supply": f"DC {v} AC 1"}
        a, b = _op(graph, ph, cfg), _op(per, ph, cfg)
        hits.append(moved(a[0], a[1], b[0], b[1], PRIMARY_ATOL, PRIMARY_RTOL))
    out["dc_sweep"] = None if all(h is None for h in hits) else any(h for h in hits)

    a, b = _ac(graph, base_ph, cfg), _ac(per, base_ph, cfg)
    out["ac"] = moved(a[0], a[1], b[0], b[1], AC_ATOL, AC_RTOL)

    nodes = sorted(v_r) if ok_r else []
    if nodes:
        hits = []
        for t in TRAN_SAMPLES:
            a = _tran(graph, base_ph, cfg, nodes, t)
            b = _tran(per, base_ph, cfg, nodes, t)
            hits.append(moved(a[0], a[1], b[0], b[1], TRAN_ATOL, TRAN_RTOL))
        out["tran"] = None if all(h is None for h in hits) else any(h for h in hits)
    else:
        out["tran"] = None

    out["any"] = any(bool(out[c]) for c in CONDITIONS)
    return out


def rate(rows, key):
    vals = [bool(r["probes"][key]) for r in rows]
    if not vals:
        return None
    iv = bootstrap_rate(vals)
    return {"rate": iv.point, "ci95": [iv.lo, iv.hi], "n": len(vals)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    a = ap.parse_args()

    cfg = load_config(a.config)
    base_ph = policy_placeholders(cfg, "hv")
    rng = random.Random(a.seed)
    value_phs = [random_placeholders(base_ph, rng) for _ in range(K_VALUES)]

    stems = sorted(CACHE.glob("*.json"))
    stems = stems[: a.limit] if a.limit else stems

    rows, ctl_rows = [], []
    ineligible: Counter = Counter()
    for n, p in enumerate(stems, 1):
        rec = json.loads(p.read_text())
        gt = rec.get("gt_graph")
        if not gt:
            continue
        ok, _ = _op(gt, base_ph, cfg)
        if not ok:
            continue                       # baseline must be a real solution
        for spec, sink in ((SWAP_SPEC, rows), (PASSIVE_SPEC, ctl_rows)):
            cands, inel = swap_candidates(gt, spec)
            ineligible.update(inel)
            for c in cands:
                sink.append({"stem": p.stem, "comp_id": c["comp_id"],
                             "class": c["class"], "swap_kind": c["swap_kind"],
                             "swapped_nets": list(swapped_nets(gt, c)),
                             "probes": probe_swap(gt, c, cfg, base_ph, value_phs)})
        if n % 20 == 0:
            print(f"  ...{n}/{len(stems)}  swaps={len(rows)} "
                  f"control={len(ctl_rows)}", flush=True)

    by_kind: dict[str, list] = defaultdict(list)
    for r in rows:
        by_kind[r["swap_kind"]].append(r)

    kinds = {}
    for kind, rs in sorted(by_kind.items()):
        kinds[kind] = {"n": len(rs),
                       **{c: rate(rs, c) for c in CONDITIONS},
                       "multi_condition": rate(rs, "any")}

    report = {
        "_what": ("The 494-swap acceptance control re-run under the D5 "
                  "multi-condition criterion. A swap counts as detected if any "
                  "condition catches it -- the dual of requiring agreement "
                  "under every condition."),
        "_status": "PENDING AUTHOR REVIEW -- results/sim_coverage/ is gitignored",
        "conditions": {
            "op": "the published probe (control: should reproduce published rates)",
            "values": f"{K_VALUES} E24 component-value assignments",
            "dc_sweep": f"supply at {list(SWEEP_VOLTS)} V",
            "ac": "1 kHz node magnitude",
            "tran": f"step stimulus, samples at {list(TRAN_SAMPLES)} s",
        },
        "n_swaps": len(rows),
        "n_passive_control": len(ctl_rows),
        "ineligible": dict(ineligible),
        "pooled": {c: rate(rows, c) for c in CONDITIONS},
        "pooled_multi_condition": rate(rows, "any"),
        "by_swap_kind": kinds,
        "passive_control": {
            "_must_be_zero": ("terminal order carries no meaning for these "
                              "classes; any detection is a broken criterion, "
                              "not a sensitive one"),
            **{c: rate(ctl_rows, c) for c in CONDITIONS},
            "multi_condition": rate(ctl_rows, "any"),
        },
        "per_swap": rows,
    }
    out_p = ROOT / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=1) + "\n")

    print(f"\nwrote {a.out}  (gitignored, pending review)")
    po, pm = report["pooled"]["op"], report["pooled_multi_condition"]
    print(f"  swaps {len(rows)}, passive control {len(ctl_rows)}")
    print(f"  pooled detection, published probe : {po['rate']:.4f}")
    print(f"  pooled detection, multi-condition : {pm['rate']:.4f}"
          f"   (+{pm['rate'] - po['rate']:.4f})")
    pc = report["passive_control"]["multi_condition"]
    print(f"  PASSIVE CONTROL under multi-condition: {pc['rate']:.4f} "
          f"(must be 0.0000)  {'OK' if pc['rate'] == 0.0 else 'FAIL'}")
    print()
    print(f"  {'swap kind':26s} {'n':>4s} {'op':>8s} {'multi':>8s} {'gain':>8s}")
    for kind, v in sorted(kinds.items(), key=lambda kv: -kv[1]['n']):
        o, m = v["op"]["rate"], v["multi_condition"]["rate"]
        print(f"  {kind:26s} {v['n']:4d} {o:8.4f} {m:8.4f} {m - o:+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
