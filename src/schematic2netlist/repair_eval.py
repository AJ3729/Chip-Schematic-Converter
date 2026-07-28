"""Repair-layer evaluation (C5 MSP): solvability lift, minimality,
per-issue breakdown, topology-preservation proof, and gauge-inference
accuracy (BUILD-F repair evaluation).

Reported on its OWN axis, never mixed into the topology benchmark
(plan §2: "made ngspice converge" is not a success metric on its own).

Three measurement families:

1. **Aggregation** (:func:`aggregate_repair`) — consumes a benchmark
   run's ``per_image.csv`` + ``ledgers/``: solvability before/after with
   a paired per-image bootstrap CI on the lift, assumptions/circuit,
   gauge-vs-assumption split, per-issue histogram, minimality-budget
   compliance.
2. **Topology preservation** (:func:`check_topology_preserved`) — the
   integrity rule made experimental: repair must not mutate any
   component's net assignment, and every injected SPICE line must match
   an allowlist of pure additions (reference ties, shunts). Run over
   real pipeline outputs it turns "provably untouched by construction"
   into a measured 0/N.
3. **Gauge accuracy** (:func:`ground_choice_accuracy`) — did the
   pipeline's chosen reference net map to the ground net the human GT
   named? Split by whether a GND symbol was present (gauge case) or the
   reference was assumed (assumption case).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmark import align_components, canonicalize_terminals

# The ONLY netlist lines repair may inject (see repair.repair_circuit):
# a 0-ohm reference tie and a finite shunt to ground. Anything else is
# a topology-preservation violation.
_ALLOWED_EXTRA = (
    re.compile(r"^Rref \S+ 0 0$"),
    re.compile(r"^Rshunt_\S+ \S+ 0 \S+$"),
)


# --------------------------------------------------------------------------
# 1. aggregation from run artifacts
# --------------------------------------------------------------------------

def _paired_bootstrap_ci(
    deltas: np.ndarray, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of the mean of per-image deltas (paired resampling)."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = [
        float(np.mean(deltas[rng.integers(0, n, n)])) for _ in range(n_resamples)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_repair(run_dir: str | Path, max_assumptions: int | None = None) -> dict:
    """Aggregate C5 metrics from a benchmark run directory."""
    run_dir = Path(run_dir)
    with (run_dir / "per_image.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no rows in {run_dir}/per_image.csv")

    before = np.array([int(r["solvable_before"]) for r in rows])
    after = np.array([int(r["solvable_after"]) for r in rows])
    n_assum = np.array([int(r["num_assumptions"]) for r in rows])
    n_gauge = np.array([int(r["num_gauge"]) for r in rows])
    lift = (after - before).astype(float)
    lo, hi = _paired_bootstrap_ci(lift)

    if max_assumptions is None:
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            max_assumptions = int(
                meta["config"].get("repair", {}).get("max_assumptions", 0)
            )

    issue_counter: Counter = Counter()
    category_counter: Counter = Counter()
    ledger_dir = run_dir / "ledgers"
    n_ledgers = 0
    for p in sorted(ledger_dir.glob("*.json")):
        led = json.loads(p.read_text())
        n_ledgers += 1
        for e in led.get("entries", []):
            issue_counter[(e["issue"], e["category"])] += 1
            category_counter[e["category"]] += 1

    return {
        "n_images": len(rows),
        "n_ledgers": n_ledgers,
        "solvable_before_rate": float(before.mean()),
        "solvable_after_rate": float(after.mean()),
        "solvability_lift": float(lift.mean()),
        "lift_ci95_lo": lo,
        "lift_ci95_hi": hi,
        "regressed_images": int(((after - before) < 0).sum()),
        "mean_assumptions": float(n_assum.mean()),
        "median_assumptions": float(np.median(n_assum)),
        "max_assumptions_observed": int(n_assum.max()),
        "mean_gauge": float(n_gauge.mean()),
        "budget": max_assumptions,
        "budget_violations": (
            int((n_assum > max_assumptions).sum()) if max_assumptions else None
        ),
        "per_issue": {
            f"{issue}|{cat}": cnt
            for (issue, cat), cnt in sorted(issue_counter.items())
        },
        "entries_by_category": dict(category_counter),
    }


# --------------------------------------------------------------------------
# 2. topology-preservation proof
# --------------------------------------------------------------------------

def check_topology_preserved(
    components: list[dict], repair_result, before_nets: list[list]
) -> list[str]:
    """Return violations ('' == none) for one image.

    ``before_nets`` is a deep snapshot of each component's node_names
    taken BEFORE repair ran; ``components`` is the same list after.
    """
    violations = []
    after_nets = [list(c.get("node_names", [])) for c in components]
    if before_nets != after_nets:
        violations.append("component net assignments changed by repair")
    if repair_result is not None:
        for line in repair_result.extra_lines:
            if not any(rx.match(line) for rx in _ALLOWED_EXTRA):
                violations.append(f"non-allowlisted injected line: {line!r}")
    return violations


# --------------------------------------------------------------------------
# 3. gauge-inference accuracy: ground choice vs GT
# --------------------------------------------------------------------------

def _net_terminal_slots(comps: list[dict], net: str, id_ceiling: int) -> set:
    """(component id, canonical terminal index) slots on ``net``,
    restricted to matched components (id < id_ceiling)."""
    return {
        (c["id"], idx)
        for c in comps
        if c["id"] < id_ceiling
        for idx, n in enumerate(c["nets"])
        if n == net
    }


def ground_choice_accuracy(
    pred: list[dict], gt: list[dict], iou_threshold: float = 0.3
) -> dict | None:
    """Did pred's reference net ("0") land on GT's ground net ("0")?

    Aligns components (Hungarian, same protocol as the benchmark) and
    canonicalizes terminal order by connectivity signature on both
    sides, then maps pred net "0" onto the GT net with the largest
    overlap in (component, canonical-terminal) slots. A tie between GT
    nets is reported as ambiguous and NOT credited as correct
    (conservative). Returns None when the question is unanswerable
    (pred has no ground terminals on matched components).
    """
    pred_a, gt_a, _ = align_components(pred, gt, iou_threshold)
    pred_c = canonicalize_terminals(pred_a)
    gt_c = canonicalize_terminals(gt_a)
    id_ceiling = max((c["id"] for c in gt_c), default=-1) + 1

    pred_ground = _net_terminal_slots(pred_c, "0", id_ceiling)
    if not pred_ground:
        return None
    gt_nets = sorted({n for c in gt_c for n in c["nets"] if n is not None})
    overlaps = {
        net: len(pred_ground & _net_terminal_slots(gt_c, net, id_ceiling))
        for net in gt_nets
    }
    best = max(overlaps.values(), default=0)
    if best == 0:
        return None
    winners = sorted(n for n, v in overlaps.items() if v == best)
    if len(winners) > 1:
        return {
            "mapped_gt_net": "|".join(winners),
            "correct": False,
            "ambiguous": True,
        }
    return {"mapped_gt_net": winners[0], "correct": winners[0] == "0"}
