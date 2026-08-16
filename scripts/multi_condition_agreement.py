#!/usr/bin/env python3
"""Operating-point agreement under MORE THAN ONE probe (task D5).

``scripts/measure_op_agreement.py`` compares one DC operating point, at one bias,
under a placeholder policy chosen for its sensitivity to injected pin swaps. That
is a good probe and its selection is audited. It is still one probe, and a single
probe supports a narrower claim than the paper wants to make: "the recovered
netlist simulates like the reference" is a statement about behaviour, and a DC
operating point is one point of it.

THE BLIND SPOT THAT MOTIVATES THIS

At DC an inductor is a short and a capacitor is an open. Every reactive element
in the drawing therefore contributes NOTHING to the operating point: a circuit
whose only recovery error is in its L/C network scores a perfect 1.000 and is
counted among the successes. That is not a hypothetical -- passives dominate this
corpus. So the headline is measured on a probe that is structurally blind to a
whole class of the errors it is meant to detect, in the same way (and for the
same kind of reason) that ``canonicalize_terminals`` is blind to pin order.

THE THREE CONDITIONS

  op_primary    the published probe: .op under the `hv` policy.
  op_low_bias   the same .op at a much smaller supply. Nonlinear devices sit in
                a different region, so two circuits that coincide at one bias by
                arithmetic accident need not coincide at another.
  ac_1khz       .ac at 1 kHz, comparing node voltage MAGNITUDE. Reactances are
                finite here, so the L/C network is visible for the first time.

Both .op conditions reuse the existing, cross-checked .op-table parser
unchanged. Only AC needs a new parse, and it gets its own unit tests plus a
self-comparison control, because this file's own history contains a `print all`
parse that was silently wrong for every single-node deck.

WHAT IS NOT DONE HERE, DELIBERATELY

No condition tries to make a non-solving circuit solve. Adding rshunt or gmin
stepping would let singular decks print a full node table, and
``_run_ngspice`` already documents why those voltages are an artefact of the
fallback pinning a floating node rather than an operating point. Recovering
circuits that way would manufacture agreement out of numerical scaffolding. The
domain stays exactly what it was: circuits where both sides genuinely solve.

Usage:
    python scripts/multi_condition_agreement.py
    python scripts/multi_condition_agreement.py --limit 20
    python scripts/multi_condition_agreement.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
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
from stats.bootstrap import bootstrap_mean, bootstrap_rate  # noqa: E402
from stats.mcnemar import mcnemar_exact  # noqa: E402

CACHE = ROOT / "results/final/op_agreement/cache"
OUT = ROOT / "results/multi_condition_agreement.json"

# ngspice prints AC vectors as `name = real,imag`.
_AC_ROW_RE = re.compile(
    r"^\s*([A-Za-z0-9_.:+\-\[\]#]+)\s*=\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")

AC_PROBE = [".control", "ac lin 1 1k 1k", "print all", ".endc"]

# A magnitude tolerance, not a voltage one. AC node magnitudes here are driven by
# a 1 V AC source, so they live in [0, ~1] rather than around 15 V, and reusing
# the 1 mV absolute tolerance would be a far stricter test than the DC one rather
# than a comparable one. 1e-4 of a 1 V drive is the same relative resolution
# 1 mV is of a 15 V rail, near enough, and the relative term does the work.
AC_ATOL = 1e-4
AC_RTOL = 1e-2

# Classes the writer emits as nonlinear elements (D, Q, M with a .model). An
# op-amp is NOT here: it is written as `E out 0 in+ in-`, an ideal VCVS, which is
# linear.
NONLINEAR_CLASSES = {"Diode", "Zener Diode", "MOSFET-N", "MOSFET-P",
                     "BJT-NPN", "BJT-PNP"}


def is_linear(graph: list[dict]) -> bool:
    """True when the deck contains no element that must be linearised.

    This decides whether an AC result means anything for a circuit whose DC
    operating point does not solve. For a LINEAR deck, `.ac` is a direct
    complex-linear solve and needs no operating point at all -- a network that is
    singular at DC (a capacitor in series with the only source, say) can be
    perfectly well defined at 1 kHz, and measuring it there is legitimate. For a
    NONLINEAR deck, ngspice linearises about the DC solution, so if that solution
    failed the small-signal result is linearised about whatever the fallback
    pinned -- the same artefact this file refuses to build on elsewhere. ngspice
    prints a full node table either way, which is exactly why this has to be
    decided from the netlist rather than from whether output appeared.
    """
    return not any(c.get("class") in NONLINEAR_CLASSES for c in graph)


def parse_ac_magnitudes(stdout: str) -> dict[str, float]:
    """Node -> |V| from an `.ac` + `print all` run.

    Skips `frequency` (the sweep variable, not a node) and branch currents,
    which are element quantities and have no counterpart in the net
    correspondence. Returns {} when nothing parsed, so a failed run is
    distinguishable from an all-zero one.
    """
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        m = _AC_ROW_RE.match(line)
        if not m:
            continue
        name, re_s, im_s = m.group(1), m.group(2), m.group(3)
        low = name.lower()
        if low == "frequency" or "#branch" in low:
            continue
        out[low] = math.hypot(float(re_s), float(im_s))
    return out


def run_condition(graph: list[dict], cond: str, cfg: dict) -> dict:
    """One side of one condition -> {'solved', 'values'} in a common shape."""
    comps = spice_components(graph)
    if cond == "ac_1khz":
        ph = policy_placeholders(cfg, "hv")
        res = _run_ngspice(build_deck(comps, ph, extra_lines=list(AC_PROBE)), cfg)
        vals = parse_ac_magnitudes(res["stdout"])
        # An AC run reports its own convergence; the .op-derived `solved` flag
        # does not apply, so solvability is "it printed node magnitudes".
        return {"solved": bool(vals), "values": vals}

    ph = policy_placeholders(cfg, "hv")
    if cond == "op_low_bias":
        ph = {**ph, "dc_supply": "DC 1", "ac_supply": "DC 1 AC 1"}
    res = _run_ngspice(build_deck(comps, ph), cfg)
    return {"solved": bool(res["solved"]), "values": res["voltages"]}


# E24, the standard 5% preferred-value series. Drawing from it rather than from
# a continuous range keeps every deck a circuit an engineer could actually build,
# so a disagreement cannot be blamed on an absurd component value.
E24 = (1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1)

# (key, unit suffix, decade range) per class. Ranges are ordinary working values.
VALUE_RANGES = (
    ("resistor", "", (2, 5)),        # 100 ohm .. 100 k
    ("capacitor", "", (-9, -5)),     # 1 nF .. 10 uF
    ("inductor", "", (-5, -2)),      # 10 uH .. 10 mH
)

K_ASSIGNMENTS = 10


def _eng(value: float) -> str:
    """SPICE-friendly engineering notation, e.g. 4.7k, 220n."""
    for exp, suf in ((9, "g"), (6, "meg"), (3, "k"), (0, ""), (-3, "m"),
                     (-6, "u"), (-9, "n"), (-12, "p")):
        if abs(value) >= 10 ** exp:
            return f"{value / 10 ** exp:g}{suf}"
    return f"{value:g}"


def random_placeholders(base: dict, rng) -> dict:
    """One E24 assignment, applied identically to both decks.

    Values are OUR choice on both sides and therefore cancel, exactly as the
    single fixed assignment does. What varies here is whether a circuit's
    agreement survives being asked the same question with different components:
    two topologies that coincide at one assignment by arithmetic accident are
    unlikely to coincide at ten.
    """
    ph = dict(base)
    for key, _suf, (lo, hi) in VALUE_RANGES:
        mant = rng.choice(E24)
        dec = rng.randint(lo, hi)
        ph[key] = _eng(mant * (10 ** dec))
    supply = rng.choice((5, 9, 12, 15, 24))
    ph["dc_supply"] = f"DC {supply}"
    ph["ac_supply"] = f"DC {supply} AC 1"
    cur = rng.choice((100, 500, 1000))          # microamps
    ph["dc_current"] = f"DC {cur}u"
    ph["ac_current"] = f"DC {cur}u AC 1m"
    return ph


CONDITIONS = ("op_primary", "op_low_bias", "ac_1khz")


def score_condition(cond: str, gt_v: dict, pred_v: dict, corr: dict) -> dict:
    if cond == "ac_1khz":
        return score_op(gt_v, pred_v, corr, atol=AC_ATOL, rtol=AC_RTOL)
    return score_op(gt_v, pred_v, corr, atol=PRIMARY_ATOL, rtol=PRIMARY_RTOL)


def measure(stems: list[str], cfg: dict) -> list[dict]:
    rows = []
    for i, stem in enumerate(stems, 1):
        rec = json.loads((CACHE / f"{stem}.json").read_text())
        gt, pred, corr = rec.get("gt_graph"), rec.get("pred_graph"), rec.get("corr")
        if not gt or not pred or not corr:
            continue
        row: dict = {"stem": stem}
        dc_ok = None
        for cond in CONDITIONS:
            g = run_condition(gt, cond, cfg)
            p = run_condition(pred, cond, cfg)
            if cond == "op_primary":
                dc_ok = bool(g["solved"] and p["solved"])
            if not (g["solved"] and p["solved"]):
                row[cond] = {"in_domain": False}
                continue
            if cond == "ac_1khz" and not dc_ok:
                # AC without a DC solution is only meaningful for a linear deck;
                # see is_linear. Both sides must qualify, or the comparison is
                # between two linearisations about points that do not exist.
                if not (is_linear(gt) and is_linear(pred)):
                    row[cond] = {"in_domain": False,
                                 "excluded": "nonlinear deck whose DC operating "
                                             "point does not solve"}
                    continue
            s = score_condition(cond, g["values"], p["values"], corr)
            # A reference response that is identically zero agrees for free and
            # measures nothing -- flagged, never merged into a rate silently.
            degenerate = all(abs(v) < 1e-12 for v in g["values"].values())
            row[cond] = {"in_domain": True, "f1": s["f1"],
                         "exact": s["f1"] == 1.0, "degenerate": degenerate}
        # The stricter criterion: agreement under EVERY one of K independent
        # E24 assignments, each applied identically to both decks.
        row["multi_value"] = multi_value_agreement(gt, pred, corr, cfg)

        rows.append(row)
        if i % 20 == 0:
            print(f"  ...{i}/{len(stems)}", flush=True)
    return rows


def multi_value_agreement(gt: list[dict], pred: list[dict], corr: dict,
                          cfg: dict, k: int = K_ASSIGNMENTS,
                          seed: int = 0) -> dict:
    """Exact agreement under all K value assignments, and under each alone.

    Seeded, so the K assignments are the same for every circuit and across runs
    -- circuits must be compared under identical conditions, and a re-run must
    reproduce the table.
    """
    import random as _random

    rng = _random.Random(seed)
    base = policy_placeholders(cfg, "hv")
    per_draw, solved_all = [], True
    for _ in range(k):
        ph = random_placeholders(base, rng)
        g = _run_ngspice(build_deck(spice_components(gt), ph), cfg)
        p = _run_ngspice(build_deck(spice_components(pred), ph), cfg)
        if not (g["solved"] and p["solved"]):
            solved_all = False
            break
        s = score_op(g["voltages"], p["voltages"], corr,
                     atol=PRIMARY_ATOL, rtol=PRIMARY_RTOL)
        per_draw.append(s["f1"] == 1.0)
    if not solved_all or not per_draw:
        return {"in_domain": False}
    return {"in_domain": True, "k": len(per_draw),
            "exact_under_all": all(per_draw),
            "exact_under_any": any(per_draw),
            "n_exact": sum(per_draw)}


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for cond in CONDITIONS:
        live = [r for r in rows if r.get(cond, {}).get("in_domain")]
        usable = [r for r in live if not r[cond]["degenerate"]]
        if not usable:
            out[cond] = {"n_in_domain": len(live), "n_informative": 0}
            continue
        f1 = bootstrap_mean([r[cond]["f1"] for r in usable])
        ex = bootstrap_rate([r[cond]["exact"] for r in usable])
        out[cond] = {
            "n_in_domain": len(live),
            "n_informative": len(usable),
            "n_degenerate": len(live) - len(usable),
            "mean_f1": {"mean": f1.point, "ci95": [f1.lo, f1.hi]},
            "exact_rate": {"mean": ex.point, "ci95": [ex.lo, ex.hi]},
        }

    # The question the task exists to answer: does a second probe reclassify
    # circuits the published one calls perfect?
    cross = {}
    for cond in CONDITIONS:
        if cond == "op_primary":
            continue
        paired = [r for r in rows
                  if r.get("op_primary", {}).get("in_domain")
                  and r.get(cond, {}).get("in_domain")
                  and not r["op_primary"]["degenerate"]
                  and not r[cond]["degenerate"]]
        a = [r["op_primary"]["exact"] for r in paired]
        b = [r[cond]["exact"] for r in paired]
        only_op = [r["stem"] for r, x, y in zip(paired, a, b) if x and not y]
        only_other = [r["stem"] for r, x, y in zip(paired, a, b) if y and not x]
        entry = {
            "n_compared": len(paired),
            "exact_under_both": sum(1 for x, y in zip(a, b) if x and y),
            "exact_only_under_op_primary": only_op,
            "exact_only_under_this": only_other,
        }
        if paired:
            mc = mcnemar_exact(a, b)
            entry["mcnemar"] = {
                "n_only_op_primary": mc.n_only_a, "n_only_this": mc.n_only_b,
                "p_value": mc.p_value}
        cross[cond] = entry
    out["_reclassification"] = cross

    # The stricter criterion the plan asks for: agreement under all K draws.
    mv = [r for r in rows if r.get("multi_value", {}).get("in_domain")]
    if mv:
        strict = bootstrap_rate([r["multi_value"]["exact_under_all"] for r in mv])
        loose = bootstrap_rate([r["multi_value"]["exact_under_any"] for r in mv])
        single = [r for r in mv
                  if r.get("op_primary", {}).get("in_domain")
                  and not r["op_primary"]["degenerate"]]
        a = [r["op_primary"]["exact"] for r in single]
        b = [r["multi_value"]["exact_under_all"] for r in single]
        lost = [r["stem"] for r, x, y in zip(single, a, b) if x and not y]
        out["multi_value"] = {
            "_meaning": (
                f"agreement required under ALL {K_ASSIGNMENTS} independent E24 "
                "value assignments, each applied identically to both decks. Two "
                "topologies that coincide at one assignment by arithmetic "
                "accident are unlikely to coincide at ten"),
            "k": K_ASSIGNMENTS,
            "n_in_domain": len(mv),
            "exact_under_all": {"mean": strict.point,
                                "ci95": [strict.lo, strict.hi]},
            "exact_under_at_least_one": {"mean": loose.point,
                                         "ci95": [loose.lo, loose.hi]},
            "n_compared_with_single_assignment": len(single),
            "lost_under_the_stricter_criterion": lost,
            "n_lost": len(lost),
        }
        if single:
            mc = mcnemar_exact(a, b)
            out["multi_value"]["mcnemar_vs_single"] = {
                "n_only_single": mc.n_only_a, "n_only_all_k": mc.n_only_b,
                "p_value": mc.p_value}

    # The substantive result: how many circuits any probe can say anything about.
    def informative(cond):
        return {r["stem"] for r in rows
                if r.get(cond, {}).get("in_domain")
                and not r[cond].get("degenerate")}

    op_set = informative("op_primary")
    ac_set = informative("ac_1khz")
    out["_coverage"] = {
        "_meaning": ("the published metric is defined only where a DC operating "
                     "point exists on both sides. A linear network that is "
                     "singular at DC can still be well defined at 1 kHz, so AC "
                     "measures circuits the DC probe cannot -- these are added "
                     "circuits, not rescued ones"),
        "informative_under_op_primary": len(op_set),
        "informative_under_ac_1khz": len(ac_set),
        "informative_under_both": len(op_set & ac_set),
        "added_by_ac_alone": sorted(ac_set - op_set),
        "n_added_by_ac_alone": len(ac_set - op_set),
        "informative_under_either": len(op_set | ac_set),
        "excluded_nonlinear_without_dc": sorted(
            r["stem"] for r in rows if r.get("ac_1khz", {}).get("excluded")),
    }
    return out


# ------------------------------------------------------------------ self-test

def self_test(cfg: dict) -> int:
    """Two controls, both of which the AC parser must pass before it is used.

    1. A deck compared against ITSELF must score exactly 1.000 under every
       condition. This is the null case: a parser that returns {} for both sides
       would otherwise produce a vacuous "agreement".
    2. The AC parse must be non-empty and must exclude branch currents and the
       sweep variable -- the failure this repo has already had once is a parse
       that returned something plausible and wrong.
    """
    ok = True
    stems = sorted(p.stem for p in CACHE.glob("*.json"))[:12]
    checked = 0
    for stem in stems:
        rec = json.loads((CACHE / f"{stem}.json").read_text())
        gt = rec.get("gt_graph")
        if not gt:
            continue
        for cond in CONDITIONS:
            r = run_condition(gt, cond, cfg)
            if not r["solved"]:
                continue
            # The identity map over the nodes ngspice ACTUALLY REPORTED. Building
            # it from the stored correspondence instead would test the
            # correspondence -- which legitimately does not cover every node --
            # and this control is about the parser and the scorer.
            ident = {n: n for n in r["values"]}
            s = score_condition(cond, r["values"], r["values"], ident)
            if s["f1"] != 1.0:
                print(f"  FAIL {stem} {cond}: self-comparison scored "
                      f"{s['f1']:.6f}, must be exactly 1.0")
                ok = False
            checked += 1
    print(f"  self-comparison: {checked} (deck, condition) pairs, all exactly "
          f"1.000  {'OK' if ok else 'FAIL'}")

    sample = ("frequency = 1.000000e+03,0.000000e+00\n"
              "l1#branch = 1.000822e-03,-1.24335e-05\n"
              "n1 = 1.000000e+00,0.000000e+00\n"
              "n2 = 7.812182e-05,6.288349e-03\n")
    got = parse_ac_magnitudes(sample)
    want_keys = {"n1", "n2"}
    parse_ok = set(got) == want_keys and abs(got["n1"] - 1.0) < 1e-12
    ok &= parse_ok
    print(f"  AC parse: {sorted(got)} (frequency and #branch excluded)  "
          f"{'OK' if parse_ok else 'FAIL'}")

    mag_ok = abs(parse_ac_magnitudes("n3 = 3.0,4.0\n")["n3"] - 5.0) < 1e-12
    ok &= mag_ok
    print(f"  AC magnitude is |re+j im|  {'OK' if mag_ok else 'FAIL'}")

    empty_ok = parse_ac_magnitudes("no vectors here") == {}
    ok &= empty_ok
    print(f"  a failed run parses to {{}} not to a false zero  "
          f"{'OK' if empty_ok else 'FAIL'}")

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
    if a.self_test:
        return self_test(cfg)

    stems = sorted(p.stem for p in CACHE.glob("*.json"))
    stems = stems[: a.limit] if a.limit else stems
    print(f"measuring {len(stems)} circuits under {len(CONDITIONS)} conditions")
    rows = measure(stems, cfg)
    summary = summarise(rows)

    report = {
        "_what": ("Operating-point agreement under more than one probe. The "
                  "published number is one DC operating point at one bias, "
                  "which is structurally blind to the reactive network: at DC "
                  "an inductor is a short and a capacitor an open."),
        "_conditions": {
            "op_primary": "the published probe -- .op under the hv policy",
            "op_low_bias": "the same .op at DC 1 instead of DC 15; nonlinear "
                           "devices sit in a different region",
            "ac_1khz": ".ac at 1 kHz, comparing node voltage magnitude, where "
                       "reactances are finite and the L/C network is visible",
        },
        "_domain": ("unchanged from the published metric: circuits where both "
                    "sides genuinely solve. No condition applies rshunt or gmin "
                    "stepping to rescue a singular deck -- those voltages are "
                    "an artefact of the fallback, not an operating point."),
        "tolerances": {"op_atol_V": PRIMARY_ATOL, "op_rtol": PRIMARY_RTOL,
                       "ac_atol": AC_ATOL, "ac_rtol": AC_RTOL},
        "n_circuits": len(rows),
        "summary": summary,
        "per_circuit": rows,
    }
    out_p = ROOT / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=1) + "\n")

    print(f"\nwrote {a.out}")
    for cond in CONDITIONS:
        s = summary[cond]
        if not s.get("n_informative"):
            print(f"  {cond:14s} no informative circuits")
            continue
        print(f"  {cond:14s} n={s['n_informative']:3d}  "
              f"mean F1 {s['mean_f1']['mean']:.4f}  "
              f"exact {s['exact_rate']['mean']:.4f}")
    for cond, c in summary["_reclassification"].items():
        print(f"  vs op_primary [{cond}]: {len(c['exact_only_under_op_primary'])} "
              f"circuit(s) exact under .op only, "
              f"{len(c['exact_only_under_this'])} under {cond} only "
              f"(n={c['n_compared']})")
    cov = summary["_coverage"]
    print(f"\n  coverage: .op {cov['informative_under_op_primary']}, "
          f"AC {cov['informative_under_ac_1khz']}, "
          f"either {cov['informative_under_either']} "
          f"(+{cov['n_added_by_ac_alone']} linear circuits AC can measure and "
          f".op cannot)")
    print(f"  excluded as nonlinear without a DC solution: "
          f"{len(cov['excluded_nonlinear_without_dc'])}")
    mv = summary.get("multi_value")
    if mv:
        print(f"\n  under all {mv['k']} E24 assignments: "
              f"exact {mv['exact_under_all']['mean']:.4f} "
              f"(vs {mv['exact_under_at_least_one']['mean']:.4f} under at least "
              f"one), n={mv['n_in_domain']}")
        print(f"  circuits that pass a single assignment but fail all-{mv['k']}: "
              f"{mv['n_lost']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
