#!/usr/bin/env python3
"""Multistability control (task D4).

The central finding -- 52 of 84 topologically perfect circuits settle to a
different operating point -- assumes the REFERENCE operating point is itself
well defined. If a reference deck has more than one stable DC solution, or if
ngspice's answer depends on element ordering or on where the solver starts,
then some of those 52 disagreements are the solver's, not the pipeline's.

So perturb the reference against itself, in the two ways that move a DC solve
without changing the circuit:

  element ordering    5 permutations of the element lines. A circuit is the
                      same circuit whichever order its elements are listed in;
                      ngspice's matrix ordering and pivoting are not.
  .nodeset seeding    3 seeded initial guesses. A multistable circuit can be
                      pulled to a different stable point by its starting
                      guess; a monostable one cannot.

Any circuit whose OWN reference moves beyond the D3 threshold is flagged, and
the headline is recomputed with and without the flagged set.

Usage:
    python scripts/multistability_control.py
    python scripts/multistability_control.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.config import load_config  # noqa: E402

# Reuse the op-agreement runner rather than re-implementing deck execution and
# .op-table parsing: it is memoised, it already handles the single-vector
# ngspice quirk, and using a second parser here would let the two disagree.
sys.path.insert(0, str(ROOT / "scripts"))
from measure_op_agreement import _run_ngspice  # noqa: E402

NETLISTS = ROOT / "results/final/op_agreement/netlists"
OPSUM = ROOT / "results/final/op_agreement/summary.json"
BENCH = ROOT / "results/final/benchmark/seed0/per_image.csv"
OUT = ROOT / "results/multistability.json"

N_PERMUTATIONS = 5
NODESET_SEEDS = [0.0, 1.0, -1.0]     # volts applied to every non-ground node
D3_ATOL, D3_RTOL = 10e-3, 1e-2       # max(10 mV, 1% of circuit max |V|)

ELEMENT_RE = re.compile(r"^[A-Za-z]\w*\s+\S+\s+\S+")


def split_deck(text: str) -> tuple[list[str], list[str], list[str]]:
    """(header comments, element lines, control lines)."""
    head, elems, ctrl = [], [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("*"):
            (head if not elems else ctrl).append(ln)
        elif s.startswith("."):
            ctrl.append(ln)
        elif ELEMENT_RE.match(s):
            elems.append(ln)
        else:
            ctrl.append(ln)
    return head, elems, ctrl


def nodes_of(elems: list[str]) -> list[str]:
    out: set[str] = set()
    for ln in elems:
        parts = ln.split()
        for tok in parts[1:3]:
            if tok != "0":
                out.add(tok)
    return sorted(out)


def simulate(deck: str, cfg: dict) -> dict | None:
    """Node voltages, or None when the deck did not reach an operating point.

    `_run_ngspice` returns a record, not a voltage map, and its `solved` flag
    is stricter than "ngspice printed numbers": on a singular deck ngspice
    falls back to gmin stepping and still prints a full node table, but those
    voltages are an artefact of the fallback rather than an operating point.
    Honour that flag rather than reading `voltages` directly.
    """
    r = _run_ngspice(deck, cfg)
    if not r or not r.get("solved"):
        return None
    return r.get("voltages") or None


def disagrees(a: dict, b: dict) -> tuple[bool, float]:
    """D3 rule, scaled by the reference circuit's largest |V|."""
    scale = max((abs(x) for x in a.values()), default=0.0)
    worst = 0.0
    bad = False
    for k, va in a.items():
        if k not in b:
            bad = True
            continue
        d = abs(va - b[k])
        worst = max(worst, d)
        if d > max(D3_ATOL, D3_RTOL * scale):
            bad = True
    return bad, worst


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    op = json.loads(OPSUM.read_text())
    rows = json.loads(json.dumps(op))  # cheap deep copy for safety
    both = [r["stem"] for r in
            __import__("csv").DictReader(
                (ROOT / "results/final/op_agreement/per_image.csv").open())
            if r["population"] == "both_solve"]
    stems = both[: a.limit] if a.limit else both
    print(f"circuits where both decks solve: {len(both)} (testing {len(stems)})")

    rng = random.Random(0)
    per: dict[str, dict] = {}
    flagged: list[str] = []
    for i, stem in enumerate(stems, 1):
        deck_p = NETLISTS / f"{stem}.gt.sp"
        if not deck_p.exists():
            continue
        head, elems, ctrl = split_deck(deck_p.read_text())
        base = simulate("\n".join(head + elems + ctrl) + "\n", cfg)
        if not base:
            per[stem] = {"baseline_solved": False}
            continue

        variants: list[tuple[str, dict | None]] = []
        for k in range(N_PERMUTATIONS):
            e = elems[:]
            rng.shuffle(e)
            variants.append((f"perm{k}",
                             simulate("\n".join(head + e + ctrl) + "\n", cfg)))
        ns = nodes_of(elems)
        for v in NODESET_SEEDS:
            if not ns:
                continue
            line = ".nodeset " + " ".join(f"v({n})={v}" for n in ns)
            deck = "\n".join(head + elems + [line] + ctrl) + "\n"
            variants.append((f"nodeset{v:+.0f}V", simulate(deck, cfg)))

        moved, worst, failed = [], 0.0, []
        for name, res in variants:
            if res is None:
                failed.append(name)
                continue
            bad, w = disagrees(base, res)
            worst = max(worst, w)
            if bad:
                moved.append(name)
        per[stem] = {
            "baseline_solved": True,
            "n_variants": len(variants),
            "n_failed_to_solve": len(failed),
            "variants_moved": moved,
            "max_abs_dv_within_reference": worst,
            "flagged": bool(moved),
        }
        if moved:
            flagged.append(stem)
        if i % 20 == 0:
            print(f"  ...{i}/{len(stems)}  flagged so far {len(flagged)}",
                  flush=True)

    # recompute the headline with and without the flagged circuits
    import csv as _csv
    bench = {r["image"].replace(".jpg", ""): r
             for r in _csv.DictReader(BENCH.open())}
    skey = next(c for c in next(iter(bench.values())) if "strict" in c)
    perfect = {s for s in per
               if str(bench.get(s, {}).get(skey, "")).strip().lower() in ("true", "1")}
    dis = {x["stem"] for x in op["topologically_perfect_but_op_disagrees"]}
    p_all, d_all = len(perfect), len(perfect & dis)
    keep = perfect - set(flagged)
    p_kept, d_kept = len(keep), len(keep & dis)

    out = {
        "_what": "Does the REFERENCE operating point depend on element "
                 "ordering or on the solver's starting guess? If it does for a "
                 "circuit, that circuit's contribution to the headline is the "
                 "solver's, not the pipeline's.",
        "protocol": {
            "permutations": N_PERMUTATIONS,
            "nodeset_seeds_V": NODESET_SEEDS,
            "threshold": "D3: |dV| > max(10 mV, 1% of the reference circuit's "
                         "max |V|)",
        },
        "n_circuits_tested": len(per),
        "n_flagged_multistable_or_order_dependent": len(flagged),
        "flagged": sorted(flagged),
        "headline_all_circuits": {
            "topologically_perfect": p_all,
            "of_those_op_disagrees": d_all,
            "rate": (d_all / p_all) if p_all else None,
        },
        "headline_excluding_flagged": {
            "topologically_perfect": p_kept,
            "of_those_op_disagrees": d_kept,
            "rate": (d_kept / p_kept) if p_kept else None,
        },
        "per_circuit": per,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"\ntested                        {len(per)}")
    print(f"flagged (reference moved)     {len(flagged)}")
    print(f"headline, all circuits        {d_all}/{p_all} = "
          f"{d_all/p_all:.4f}" if p_all else "n/a")
    if p_kept:
        print(f"headline, excluding flagged   {d_kept}/{p_kept} = "
              f"{d_kept/p_kept:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
