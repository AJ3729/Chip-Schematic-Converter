#!/usr/bin/env python3
"""Agreement across the full simulation domain: DC, DC sweep, AC, transient.

Mentor fix 3b. The published claim rests on one DC operating point at one bias
under one set of placeholder values, and the honest reading of that is "DC
operating-point agreement under fixed placeholder parameters" -- not functional
equivalence. This widens the domain so the claim can be made on evidence rather
than narrowed by disclaimer. Four probes, every one applied identically to both
decks so component values cancel exactly as they do in the published metric:

  op          the published probe: .op at the frozen placeholder policy.
  dc_sweep    the SAME .op evaluated at five supply voltages spanning 1-24 V,
              with agreement required at every point. Nonlinear devices sit in
              a different region at each, so two circuits that coincide at one
              bias by arithmetic accident are unlikely to coincide at five.
  ac          .ac at 1 kHz, node magnitude. At DC an inductor is a short and a
              capacitor an open, so the reactive network is invisible to the
              first two probes and visible here.
  tran        a step stimulus and three samples through the settle. This is the
              only probe that exercises nonlinear DYNAMICS -- diode switching,
              transistor slewing -- which none of the others reach.

TWO PARSING DECISIONS, BOTH MADE TO AVOID A KNOWN FAILURE

The DC sweep is NOT ``.dc`` + ``print all``. That prints a multi-column table
split across blocks, and this metric's history already contains a ``print all``
parse that was silently wrong for every single-node deck. Five ``.op`` runs
carry the same information through the existing, cross-checked ``.op``-table
parser, and a sweep that cannot be mis-parsed is worth more than one that is
syntactically tidier.

The transient uses ``meas ... FIND v(node) AT=t``, which emits one
``name = value`` line per node, for the same reason: the raw transient table has
the same multi-column shape and the same risk.

WHAT IS DELIBERATELY NOT CLAIMED. Even all four together are not functional
equivalence. They are agreement over a declared, finite domain: one topology,
one placeholder family, five bias points, one frequency, one stimulus shape.
The report says so in the artifact rather than leaving it to the reader.

Output goes to results/sim_coverage/, which is gitignored pending review.

Usage:
    python scripts/simulation_coverage.py
    python scripts/simulation_coverage.py --limit 20
    python scripts/simulation_coverage.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from schematic2netlist.config import load_config  # noqa: E402
from measure_op_agreement import (  # noqa: E402
    PRIMARY_ATOL,
    PRIMARY_RTOL,
    _run_ngspice,
    build_deck,
    policy_placeholders,
    score_op,
    spice_components,
)
from multi_condition_agreement import (  # noqa: E402
    AC_ATOL,
    AC_RTOL,
    AC_PROBE,
    is_linear,
    parse_ac_magnitudes,
)
from stats.bootstrap import bootstrap_mean, bootstrap_rate  # noqa: E402

CACHE = ROOT / "results/final/op_agreement/cache"
OUT = ROOT / "results/sim_coverage/coverage.json"

# Supply voltages for the DC sweep. Spans the working range of the placeholder
# family; nonlinear devices are in visibly different regions across it.
SWEEP_VOLTS = (1, 5, 10, 15, 24)

# Transient: step the supply and sample through the settle. The three samples
# bracket a 1 ms-ish time constant, which is what the placeholder R and C give.
TRAN_SAMPLES = (0.2e-3, 0.5e-3, 1.5e-3)
TRAN_STOP = 2e-3
TRAN_STEP = 10e-6
TRAN_ATOL, TRAN_RTOL = 1e-3, 1e-2

_MEAS_RE = re.compile(r"^\s*m_([A-Za-z0-9_.:+\-\[\]#]+)\s*=\s*"
                      r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$", re.M)

CONDITIONS = ("op", "dc_sweep", "ac", "tran")


def parse_meas(stdout: str) -> dict[str, float]:
    """`m_<node> = value` lines from a transient meas block.

    Returns {} when nothing parsed, so a failed run stays distinguishable from
    an all-zero one.
    """
    out: dict[str, float] = {}
    for name, val in _MEAS_RE.findall(stdout):
        out[name.lower()] = float(val)
    return out


def tran_probe(nodes: list[str], t: float) -> list[str]:
    """A .control block sampling every node at one instant."""
    lines = [".control", f"tran {TRAN_STEP:g} {TRAN_STOP:g}"]
    for n in nodes:
        if n == "0":
            continue
        lines.append(f"meas tran m_{n} FIND v({n}) AT={t:g}")
    lines.append(".endc")
    return lines


def run_op(graph, ph, cfg) -> dict:
    res = _run_ngspice(build_deck(spice_components(graph), ph), cfg)
    return {"solved": bool(res["solved"]), "values": res["voltages"]}


def run_ac(graph, ph, cfg) -> dict:
    res = _run_ngspice(
        build_deck(spice_components(graph), ph, extra_lines=list(AC_PROBE)), cfg)
    vals = parse_ac_magnitudes(res["stdout"])
    return {"solved": bool(vals), "values": vals}


def run_tran(graph, ph, cfg, nodes: list[str], t: float) -> dict:
    # Step stimulus: without a time-varying source a transient simply settles to
    # the operating point and measures nothing the first probe did not.
    tph = dict(ph)
    tph["dc_supply"] = f"PULSE(0 {ph.get('_supply_v', 15)} 0 10u 10u 1m 2m)"
    tph["ac_supply"] = tph["dc_supply"]
    res = _run_ngspice(
        build_deck(spice_components(graph), tph,
                   extra_lines=tran_probe(nodes, t)), cfg)
    vals = parse_meas(res["stdout"])
    return {"solved": bool(vals), "values": vals}


def score_circuit(rec: dict, cfg: dict, base_ph: dict) -> dict:
    gt, pred, corr = rec.get("gt_graph"), rec.get("pred_graph"), rec.get("corr")
    row: dict = {"stem": rec.get("stem")}
    if not gt or not pred or not corr:
        return {**row, "skipped": "no cached graphs"}

    # ---- op ------------------------------------------------------------
    g = run_op(gt, base_ph, cfg)
    p = run_op(pred, base_ph, cfg)
    dc_ok = g["solved"] and p["solved"]
    if dc_ok:
        s = score_op(g["values"], p["values"], corr,
                     atol=PRIMARY_ATOL, rtol=PRIMARY_RTOL)
        degen = all(abs(v) < 1e-12 for v in g["values"].values())
        row["op"] = {"in_domain": True, "f1": s["f1"],
                     "exact": s["f1"] == 1.0, "degenerate": degen}
    else:
        row["op"] = {"in_domain": False}

    # ---- dc sweep ------------------------------------------------------
    per_point, ok_all = [], True
    for v in SWEEP_VOLTS:
        ph = {**base_ph, "dc_supply": f"DC {v}", "ac_supply": f"DC {v} AC 1"}
        gg, pp = run_op(gt, ph, cfg), run_op(pred, ph, cfg)
        if not (gg["solved"] and pp["solved"]):
            ok_all = False
            break
        ss = score_op(gg["values"], pp["values"], corr,
                      atol=PRIMARY_ATOL, rtol=PRIMARY_RTOL)
        per_point.append({"volts": v, "f1": ss["f1"], "exact": ss["f1"] == 1.0})
    row["dc_sweep"] = ({"in_domain": True, "points": per_point,
                        "exact_at_every_point": all(x["exact"] for x in per_point),
                        "mean_f1": sum(x["f1"] for x in per_point) / len(per_point)}
                       if ok_all and per_point else {"in_domain": False})

    # ---- ac ------------------------------------------------------------
    ga, pa = run_ac(gt, base_ph, cfg), run_ac(pred, base_ph, cfg)
    if ga["solved"] and pa["solved"] and (dc_ok or (is_linear(gt) and is_linear(pred))):
        s = score_op(ga["values"], pa["values"], corr,
                     atol=AC_ATOL, rtol=AC_RTOL)
        degen = all(abs(v) < 1e-12 for v in ga["values"].values())
        row["ac"] = {"in_domain": True, "f1": s["f1"],
                     "exact": s["f1"] == 1.0, "degenerate": degen}
    else:
        row["ac"] = {"in_domain": False,
                     **({"excluded": "nonlinear deck without a DC solution"}
                        if ga["solved"] and pa["solved"] else {})}

    # ---- transient -----------------------------------------------------
    nodes = sorted(set(g["values"]) | set(p["values"])) if dc_ok else []
    if nodes:
        pts, tok = [], True
        ref_const = True
        first_ref = None
        for t in TRAN_SAMPLES:
            gg = run_tran(gt, base_ph, cfg, nodes, t)
            pp = run_tran(pred, base_ph, cfg, nodes, t)
            if not (gg["solved"] and pp["solved"]):
                tok = False
                break
            ss = score_op(gg["values"], pp["values"], corr,
                          atol=TRAN_ATOL, rtol=TRAN_RTOL)
            pts.append({"t": t, "f1": ss["f1"], "exact": ss["f1"] == 1.0})
            sig = tuple(round(v, 9) for _, v in sorted(gg["values"].items()))
            if first_ref is None:
                first_ref = sig
            elif sig != first_ref:
                ref_const = False
        row["tran"] = ({"in_domain": True, "points": pts,
                        "exact_at_every_sample": all(x["exact"] for x in pts),
                        "mean_f1": sum(x["f1"] for x in pts) / len(pts),
                        # a reference that never moves carries no dynamics, so
                        # agreement there is the operating point again
                        "degenerate": ref_const}
                       if tok and pts else {"in_domain": False})
    else:
        row["tran"] = {"in_domain": False}
    return row


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for cond in CONDITIONS:
        live = [r for r in rows if r.get(cond, {}).get("in_domain")]
        usable = [r for r in live if not r[cond].get("degenerate")]
        key = {"op": "exact", "ac": "exact",
               "dc_sweep": "exact_at_every_point",
               "tran": "exact_at_every_sample"}[cond]
        if not usable:
            out[cond] = {"n_in_domain": len(live), "n_informative": 0}
            continue
        ex = bootstrap_rate([bool(r[cond][key]) for r in usable])
        f1s = [r[cond].get("f1", r[cond].get("mean_f1")) for r in usable]
        f1s = [float(x) for x in f1s if x is not None]
        entry = {"n_in_domain": len(live), "n_informative": len(usable),
                 "n_degenerate": len(live) - len(usable),
                 "exact_rate": {"mean": ex.point, "ci95": [ex.lo, ex.hi]}}
        if f1s:
            m = bootstrap_mean(f1s)
            entry["mean_f1"] = {"mean": m.point, "ci95": [m.lo, m.hi]}
        out[cond] = entry

    # The question the whole exercise exists to answer.
    strict = [r for r in rows
              if all(r.get(c, {}).get("in_domain") for c in CONDITIONS)
              and not any(r[c].get("degenerate") for c in CONDITIONS)]
    if strict:
        keyed = {"op": "exact", "ac": "exact",
                 "dc_sweep": "exact_at_every_point",
                 "tran": "exact_at_every_sample"}
        allfour = [all(bool(r[c][keyed[c]]) for c in CONDITIONS) for r in strict]
        oponly = [bool(r["op"]["exact"]) for r in strict]
        b = bootstrap_rate(allfour)
        o = bootstrap_rate(oponly)
        out["_all_four"] = {
            "_what": ("circuits informative under EVERY probe, and the fraction "
                      "exact under all of them versus under the published probe "
                      "alone. This is the widened claim."),
            "n": len(strict),
            "exact_under_all_four": {"mean": b.point, "ci95": [b.lo, b.hi]},
            "exact_under_op_alone": {"mean": o.point, "ci95": [o.lo, o.hi]},
            "lost_by_widening": [r["stem"] for r, a, o_ in zip(strict, allfour, oponly)
                                 if o_ and not a],
        }
    return out


def self_test(cfg: dict, base_ph: dict, n: int = 6) -> int:
    """Every probe must score a deck against ITSELF at exactly 1.000."""
    ok, checked = True, 0
    for p in sorted(CACHE.glob("*.json"))[:n]:
        rec = json.loads(p.read_text())
        gt = rec.get("gt_graph")
        if not gt:
            continue
        g = run_op(gt, base_ph, cfg)
        if not g["solved"]:
            continue
        nodes = sorted(g["values"])
        probes = {
            "op": g,
            "ac": run_ac(gt, base_ph, cfg),
            "tran": run_tran(gt, base_ph, cfg, nodes, TRAN_SAMPLES[1]),
        }
        for name, r in probes.items():
            if not r["solved"]:
                continue
            ident = {k: k for k in r["values"]}
            atol, rtol = ((AC_ATOL, AC_RTOL) if name == "ac"
                          else (TRAN_ATOL, TRAN_RTOL) if name == "tran"
                          else (PRIMARY_ATOL, PRIMARY_RTOL))
            s = score_op(r["values"], r["values"], ident, atol=atol, rtol=rtol)
            checked += 1
            if s["f1"] != 1.0:
                print(f"  FAIL {p.stem} {name}: self-comparison {s['f1']:.6f}")
                ok = False
    print(f"  self-comparison: {checked} (deck, probe) pairs exactly 1.000  "
          f"{'OK' if ok else 'FAIL'}")

    m = parse_meas("m_n1                =  1.500000e+01\nm_n2 =  9.370332e-01\n")
    pm = m == {"n1": 15.0, "n2": 0.9370332}
    ok &= pm
    print(f"  meas parse: {m}  {'OK' if pm else 'FAIL'}")
    empty = parse_meas("no measurements here") == {}
    ok &= empty
    print(f"  a failed transient parses to empty, not a false zero  "
          f"{'OK' if empty else 'FAIL'}")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    a = ap.parse_args()

    cfg = load_config(a.config)
    base_ph = policy_placeholders(cfg, "hv")
    base_ph["_supply_v"] = 15

    if a.self_test:
        return self_test(cfg, base_ph)

    stems = sorted(CACHE.glob("*.json"))
    stems = stems[: a.limit] if a.limit else stems
    print(f"{len(stems)} circuits x {len(CONDITIONS)} probes "
          f"({len(SWEEP_VOLTS)} sweep points, {len(TRAN_SAMPLES)} transient samples)")
    rows = []
    for i, p in enumerate(stems, 1):
        rows.append(score_circuit(json.loads(p.read_text()), cfg, base_ph))
        if i % 10 == 0:
            print(f"  ...{i}/{len(stems)}", flush=True)

    summary = summarise(rows)
    report = {
        "_what": "Agreement across DC, DC sweep, AC and transient (mentor fix 3b).",
        "_not_claimed": (
            "This is NOT functional equivalence. It is agreement over a "
            "declared finite domain: one topology, one placeholder family, "
            f"{len(SWEEP_VOLTS)} bias points spanning {SWEEP_VOLTS[0]}-"
            f"{SWEEP_VOLTS[-1]} V, one frequency, one stimulus shape. A circuit "
            "agreeing here can still differ under an input this domain does not "
            "contain."),
        "_status": "PENDING AUTHOR REVIEW -- results/sim_coverage/ is gitignored",
        "domain": {
            "sweep_volts": list(SWEEP_VOLTS),
            "ac_hz": 1000,
            "transient": {"stop_s": TRAN_STOP, "step_s": TRAN_STEP,
                          "samples_s": list(TRAN_SAMPLES),
                          "stimulus": "PULSE 0->supply, 10us edges"},
            "tolerances": {"op_atol_V": PRIMARY_ATOL, "op_rtol": PRIMARY_RTOL,
                           "ac_atol": AC_ATOL, "ac_rtol": AC_RTOL,
                           "tran_atol": TRAN_ATOL, "tran_rtol": TRAN_RTOL},
        },
        "n_circuits": len(rows),
        "summary": summary,
        "per_circuit": rows,
    }
    out_p = ROOT / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=1) + "\n")

    print(f"\nwrote {a.out}  (gitignored, pending review)")
    for c in CONDITIONS:
        s = summary[c]
        if not s.get("n_informative"):
            print(f"  {c:9s} no informative circuits")
            continue
        f1 = s.get("mean_f1", {}).get("mean")
        print(f"  {c:9s} n={s['n_informative']:3d}  "
              f"exact {s['exact_rate']['mean']:.4f}"
              + (f"  mean F1 {f1:.4f}" if f1 is not None else ""))
    af = summary.get("_all_four")
    if af:
        print(f"\n  informative under all four probes: {af['n']}")
        print(f"  exact under all four : {af['exact_under_all_four']['mean']:.4f}")
        print(f"  exact under .op alone: {af['exact_under_op_alone']['mean']:.4f}")
        print(f"  circuits the widening reclassifies: {len(af['lost_by_widening'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
