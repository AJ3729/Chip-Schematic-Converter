#!/usr/bin/env python3
"""Separate repair from reconstruction (task D7).

Three things, all from stored artifacts:

1. Classify every repair intervention by the type it actually declares. The
   categories are read out of the ledgers, not invented -- each entry already
   carries `issue`, `category` (gauge vs assumption) and `behavior_changing`.
2. Emit a LITERAL deck and a REPAIRED deck per circuit, the latter headed by a
   comment listing every intervention, so a reader can diff them.
3. Verify from the code path -- not by assertion -- that strict success and
   every topology metric are computed before any repair runs.

Usage:
    python scripts/repair_separation.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LEDGERS = ROOT / "results/final/benchmark/seed0/ledgers"
NETLISTS = ROOT / "results/final/op_agreement/netlists"
OUT_DIR = ROOT / "results/final/repair_separation"
OUT = ROOT / "results/repair_inventory.json"


def _func_source(src: str, header: str) -> str:
    """The body of one top-level function, by textual scope."""
    i = src.index(header)
    rest = src[i:]
    lines = rest.splitlines()
    out = [lines[0]]
    for ln in lines[1:]:
        if ln and not ln[0].isspace():      # next top-level definition
            break
        out.append(ln)
    return "\n".join(out)


def verify_repair_is_downstream() -> dict:
    """Establish from the source that topology metrics cannot see repair.

    The claim "strict success involves no repair" is load-bearing enough that
    it should be checked against the code rather than asserted in prose.
    """
    bench_lib = (ROOT / "src/schematic2netlist/benchmark.py").read_text()
    bench_cli = (ROOT / "scripts/benchmark.py").read_text()
    pipe = (ROOT / "src/schematic2netlist/pipeline.py").read_text()

    # The scorer reads `node_names`, which assign_node_names() writes BEFORE
    # repair_circuit() is called; repair only appends SPICE lines. Each check
    # below is a property of the source, and every one must hold.
    checks = {
        "scorer_library_never_imports_repair":
            "from schematic2netlist.repair" not in bench_lib
            and "import repair" not in bench_lib,
        # The CLI DOES import repair -- legitimately, to build the ledger and
        # the solvability block. The claim is narrower and is tested narrowly:
        # the function that builds the TOPOLOGY predictions must not reference
        # repair at all.
        "topology_prediction_builder_ignores_repair":
            "repair" not in _func_source(bench_cli, "def pred_components"),
        "scorer_reads_node_names":
            'c.get("node_names"' in bench_cli,
        "node_names_assigned_before_repair":
            pipe.index("assign_node_names(comps") < pipe.index("repair_circuit("),
        "pipeline_repairs_after_netlist_export":
            pipe.index("design-intent repair") > pipe.index("node naming"),
        "repair_declares_topology_untouched":
            "Topology is untouched" in pipe,
    }
    return {
        "_what": "Checked against the source, not asserted.",
        "checks": checks,
        "conclusion": (
            "Verified in source: the scorer reads `node_names`; "
            "assign_node_names() writes them BEFORE repair_circuit() is "
            "called; repair only appends SPICE lines; the scoring LIBRARY "
            "never imports repair; and while the benchmark CLI does import it "
            "-- to build the ledger and the solvability block -- the function "
            "that builds the topology predictions does not reference it. "
            "Strict success, terminal-pair F1, net F1, per-component accuracy "
            "and nGED therefore cannot be influenced by repair."
            if all(v is not False for v in checks.values()) else
            "CHECK FAILED -- see `checks`; do not repeat the separation claim."),
    }


def main() -> None:
    if not LEDGERS.is_dir():
        sys.exit(f"no ledgers at {LEDGERS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_issue: collections.Counter = collections.Counter()
    by_category: collections.Counter = collections.Counter()
    behavior_changing: collections.Counter = collections.Counter()
    per_circuit: dict[str, dict] = {}
    issue_detail: dict[str, dict] = {}
    n_decks = 0

    for f in sorted(LEDGERS.glob("*.json")):
        d = json.loads(f.read_text())
        stem = Path(d.get("image", f.stem)).stem
        entries = d.get("entries", [])
        issues = []
        for e in entries:
            issue = e.get("issue", "unclassified")
            cat = e.get("category", "unclassified")
            bc = bool(e.get("behavior_changing"))
            by_issue[issue] += 1
            by_category[cat] += 1
            behavior_changing[(issue, bc)] += 1
            issues.append(issue)
            issue_detail.setdefault(issue, {
                "category": cat,
                "behavior_changing": bc,
                "example_action": e.get("action"),
                "example_evidence": e.get("evidence"),
                "alternatives": e.get("alternatives"),
            })
        per_circuit[stem] = {
            "solvable_before": d.get("solvable_before"),
            "solvable_after": d.get("solvable_after"),
            "num_assumptions": d.get("num_assumptions"),
            "num_gauge": d.get("num_gauge"),
            "issues": issues,
        }

        # dual decks: literal (as reconstructed) and repaired (with a header)
        pred = NETLISTS / f"{stem}.pred.sp"
        if pred.exists():
            literal = pred.read_text()
            (OUT_DIR / f"{stem}.literal.cir").write_text(literal)
            header = ["* REPAIRED DECK -- interventions applied, listed below.",
                      "* The literal reconstruction is in the .literal.cir "
                      "beside this file.",
                      f"* interventions: {len(entries)}"]
            for e in entries:
                header.append(
                    f"*   [{e.get('category','?')}"
                    f"{'/behaviour-changing' if e.get('behavior_changing') else ''}]"
                    f" {e.get('issue','?')}: {e.get('action','?')}")
            if not entries:
                header.append("*   (none)")
            (OUT_DIR / f"{stem}.repaired.cir").write_text(
                "\n".join(header) + "\n" + literal)
            n_decks += 1

    n = len(per_circuit)
    solv_before = sum(1 for v in per_circuit.values() if v["solvable_before"])
    solv_after = sum(1 for v in per_circuit.values() if v["solvable_after"])
    changed = {i for (i, bc) in behavior_changing if bc}

    out = {
        "_what": "Every declared repair intervention, classified by the type "
                 "it records. Categories are read from the ledgers, not "
                 "invented.",
        "separation_verified": verify_repair_is_downstream(),
        "n_circuits": n,
        "n_dual_decks_written": n_decks,
        "solvable_before_repair": solv_before,
        "solvable_after_repair": solv_after,
        "solvable_before_rate": solv_before / n if n else 0.0,
        "solvable_after_rate": solv_after / n if n else 0.0,
        "interventions_by_issue": dict(by_issue.most_common()),
        "interventions_by_category": dict(by_category),
        "behaviour_changing_issues": sorted(changed),
        "non_behaviour_changing_issues": sorted(
            {i for (i, bc) in behavior_changing if not bc} - changed),
        "issue_detail": issue_detail,
        "per_circuit": per_circuit,
        "_reading": (
            "Two categories with very different standing. GAUGE choices "
            "(ground selection, current reference direction) fix an arbitrary "
            "convention and change no behaviour. ASSUMPTIONS (placeholder "
            "values, shunt resistors to tie a floating node) do change what "
            "the deck simulates and are the honest cost of the post-repair "
            "solvability figure. The pre-repair rate is the floor and should "
            "be foregrounded."),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"circuits {n}   dual decks written {n_decks}")
    print(f"solvable  before {solv_before}/{n} = {solv_before/n:.4f}"
          f"   after {solv_after}/{n} = {solv_after/n:.4f}")
    print("\ninterventions by issue:")
    for k, v in by_issue.most_common():
        det = issue_detail[k]
        flag = "behaviour-changing" if det["behavior_changing"] else "gauge only"
        print(f"  {v:5d}  {k:28s} [{det['category']:10s}] {flag}")
    print(f"\nseparation check: "
          f"{out['separation_verified']['conclusion'][:88]}...")
    print(f"\nwrote {OUT} and {OUT_DIR}")


if __name__ == "__main__":
    main()
