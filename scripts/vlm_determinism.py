#!/usr/bin/env python3
"""Repeat-query determinism of the frontier model, from the raw variant-B repeats.

Section~\\ref{sec:determinism} contrasts our pipeline (byte-identical over five
fresh interpreters) against a hosted model asked the identical question three
times. The pipeline side has an artifact -- results/final/determinism/ -- and
the model side did not, so the manuscript's 0.2344 / 0.4062 / 114 were the only
numbers in the paper with no file behind them. This computes them.

Two agreements, because they answer different questions:

  exact      all three replies list the same terminals for the same components.
             This is the strict reading of "same answer".
  topology   the induced PARTITION of terminals into nets is the same, ignoring
             what the nets are called. A model that renames n3 to n7 throughout
             has returned the same circuit; a byte comparison would call that a
             difference and would be wrong to.

Pairwise agreement is reported alongside the all-three figure because a single
odd reply out of three would otherwise be indistinguishable from three mutually
different ones, and those are very different failure modes.

Usage:
    python scripts/vlm_determinism.py
    python scripts/vlm_determinism.py --run results/vlm/claude_b_test
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rep_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for f in sorted(rep_dir.glob("*.json")):
        d = json.loads(f.read_text())
        out[f.stem] = d.get("components", [])
    return out


def exact_key(comps: list[dict]):
    """Component id -> terminal net list, as written. Order-insensitive on ids."""
    return tuple(sorted((c.get("id"), tuple(c.get("terminals", [])))
                        for c in comps))


def topo_key(comps: list[dict]):
    """Naming-invariant partition of (component, pin) sites into nets.

    Each site is (component id, pin index). Two sites are in the same block iff
    the reply gave them the same net name. The blocks are then sorted, so the
    NAMES drop out and only the grouping survives -- which is what "the same
    circuit" means.
    """
    blocks: dict[str, list[tuple]] = {}
    for c in comps:
        for i, net in enumerate(c.get("terminals", [])):
            blocks.setdefault(str(net), []).append((c.get("id"), i))
    return tuple(sorted(tuple(sorted(v)) for v in blocks.values()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/vlm/claude_b_test")
    ap.add_argument("--out", default="results/final/vlm_determinism")
    a = ap.parse_args()

    run = ROOT / a.run
    reps = sorted(p for p in run.glob("rep*") if p.is_dir())
    if len(reps) < 2:
        raise SystemExit(f"need >=2 repeat dirs under {run}")
    data = {p.name: load(p) for p in reps}

    stems = sorted(set.intersection(*(set(v) for v in data.values())))
    names = [p.name for p in reps]

    def frac_all(keyfn):
        same = [s for s in stems
                if len({keyfn(data[r][s]) for r in names}) == 1]
        return len(same) / len(stems), same

    ex_frac, ex_same = frac_all(exact_key)
    tp_frac, tp_same = frac_all(topo_key)
    changed = [s for s in stems if s not in set(tp_same)]

    pairwise = {}
    for r1, r2 in itertools.combinations(names, 2):
        n = sum(topo_key(data[r1][s]) == topo_key(data[r2][s]) for s in stems)
        pairwise[f"{r1}_vs_{r2}"] = round(n / len(stems), 6)

    summary = {
        "what_this_is": "repeat-query determinism of the hosted model, variant B",
        "run": a.run,
        "model": json.loads((next(reps[0].glob("*.json"))).read_text()).get("_model"),
        "n_repeats": len(reps),
        "n_circuits": len(stems),
        "note": ("no seed or temperature control is exposed by the interface, "
                 "so these are three identical requests"),
        "exact_output_agreement": {
            "fraction_all_repeats_identical": round(ex_frac, 6),
            "n_identical": len(ex_same),
            "definition": "same component id -> terminal net list in every repeat",
        },
        "topology_agreement": {
            "fraction_all_repeats_identical": round(tp_frac, 6),
            "n_identical": len(tp_same),
            "n_changed": len(changed),
            "definition": "naming-invariant partition of terminals into nets",
        },
        "pairwise_topology_agreement": pairwise,
        "pairwise_min": min(pairwise.values()),
        "pairwise_max": max(pairwise.values()),
    }
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"{len(stems)} circuits x {len(reps)} repeats  ({summary['model']})")
    print(f"  exact-output agreement    {ex_frac:.4f}  ({len(ex_same)}/{len(stems)})")
    print(f"  topology agreement        {tp_frac:.4f}  ({len(tp_same)}/{len(stems)})")
    print(f"  circuits changing topology            {len(changed)}/{len(stems)}")
    print(f"  pairwise topology agreement {summary['pairwise_min']:.4f}"
          f"-{summary['pairwise_max']:.4f}")
    print(f"  -> {a.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
