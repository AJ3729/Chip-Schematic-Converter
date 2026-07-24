"""Design-intent completion: minimal, transparent repair (contribution C5).

Given the ERC diagnosis (:mod:`schematic2netlist.erc`), this turns each
issue into a logged :class:`LedgerEntry` and, where a fix is warranted,
emits the *minimal* set of extra SPICE lines that make the circuit
DC-solvable — never changing the recovered topology.

The integrity rule (what makes this science, not a hack): repair only
*adds explicit constraints*; it must not alter the topology graph. The
benchmark verifies net-F1 is identical with repair on and off. "ngspice
converged" is not a success metric on its own — minimality, transparency,
and gauge-inference accuracy are.

Two categories:

- ``gauge`` — behavior-invariant or determinate-from-drawing; safe to
  infer silently but still logged (current-source reference direction,
  passive terminal order, net naming, ground selection when a GND symbol
  exists).
- ``assumption`` — behavior-changing; must be minimal and flagged with
  alternatives (no ground → pick a reference; floating subnet → shunt to
  ground; unresolved values → placeholders).

Minimality: prefer gauge (free) over SPICE aids over structural edits;
one shunt per galvanically-connected floating subnet, not per net;
count assumptions against ``repair.max_assumptions``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from schematic2netlist.classes import class_role
from schematic2netlist.erc import GROUND_NET, run_erc

LEDGER_SCHEMA_VERSION = 1


@dataclass
class LedgerEntry:
    issue: str
    category: str                       # "gauge" | "assumption"
    behavior_changing: bool
    action: str
    location: dict = field(default_factory=dict)
    alternatives: list = field(default_factory=list)
    confidence: float = 1.0
    evidence: str = ""


@dataclass
class RepairResult:
    entries: list                       # list[LedgerEntry]
    extra_lines: list                   # SPICE lines to inject before .op
    num_assumptions: int
    num_gauge: int


def _full_net_adjacency(components: list[dict]) -> dict[str, set[str]]:
    """Galvanic adjacency: any two nets sharing a component terminal set."""
    adj: dict[str, set[str]] = {}
    for c in components:
        names = [n for n in c.get("node_names", []) if n is not None]
        for a in names:
            adj.setdefault(a, set())
            for b in names:
                if a != b:
                    adj[a].add(b)
    return adj


def _floating_subnets(floating: list[str], adj: dict[str, set[str]]) -> list[list[str]]:
    """Group floating nets into galvanically-connected subnets so one
    shunt per group grounds the whole group (minimality)."""
    floating_set = set(floating)
    seen: set[str] = set()
    groups: list[list[str]] = []
    for start in floating:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for nb in adj.get(n, ()):
                if nb in floating_set and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        groups.append(sorted(comp))
    return groups


def repair_circuit(components: list[dict], node_name_map: dict, cfg: dict) -> RepairResult:
    """Diagnose and emit minimal, logged repairs. Topology is untouched."""
    rcfg = cfg.get("repair", {})
    strategies = rcfg.get("strategies", {})
    shunt_r = float(rcfg.get("shunt_r", 1e9))   # YAML "1e9" may parse as str

    issues = run_erc(components, node_name_map, cfg)
    by_key = {i.issue: i for i in issues}
    entries: list[LedgerEntry] = []
    extra_lines: list[str] = []

    def add(entry: LedgerEntry):
        entries.append(entry)

    # --- GAUGE entries (logged, no behavior change, usually no injection) ---

    if strategies.get("current_direction", True) and "unset_current_direction" in by_key:
        iss = by_key["unset_current_direction"]
        add(LedgerEntry(
            issue=iss.issue, category="gauge", behavior_changing=False,
            action="kept drawn current-source reference direction",
            location=iss.location, alternatives=["reverse reference direction"],
            confidence=1.0, evidence=iss.evidence,
        ))

    has_ground_symbol = any(class_role(c["class"]) == "ground" for c in components)
    if has_ground_symbol and GROUND_NET in {
        n for c in components for n in c.get("node_names", []) if n
    }:
        add(LedgerEntry(
            issue="ground_selection", category="gauge", behavior_changing=False,
            action="used the GND symbol's net as reference (0)",
            location={"nets": [GROUND_NET]}, alternatives=[],
            confidence=1.0, evidence="a GND symbol is present in the drawing",
        ))

    # placeholder values: behavior-changing in the absolute (true values
    # unknown) but a documented limitation, not a solvability fix
    if strategies.get("placeholder_values", True):
        add(LedgerEntry(
            issue="placeholder_values", category="assumption", behavior_changing=True,
            action="assigned per-class placeholder values (no OCR)",
            location={}, alternatives=["OCR-extracted values (future work)"],
            confidence=0.5, evidence="component values are not read from the drawing",
        ))

    # --- ASSUMPTION entries that inject SPICE aids to lift solvability ---

    present_nets = {n for c in components for n in c.get("node_names", []) if n}

    # no ground reference at all -> tie a chosen net to node 0
    if strategies.get("add_ground_reference", True) and "no_ground_reference" in by_key:
        iss = by_key["no_ground_reference"]
        if GROUND_NET not in present_nets and present_nets:
            adj = _full_net_adjacency(components)
            ref = max(present_nets, key=lambda n: len(adj.get(n, ())))
            extra_lines.append(f"Rref {ref} 0 0")   # 0-ohm tie to ground
            add(LedgerEntry(
                issue=iss.issue, category="assumption", behavior_changing=True,
                action=f"selected net {ref} as reference (0-ohm tie to node 0)",
                location={"nets": [ref]},
                alternatives=[f"tie a different net to 0", ".nodeset"],
                confidence=0.5, evidence=iss.evidence,
            ))

    # floating nets / single-terminal stubs -> one shunt per floating subnet
    if strategies.get("shunt_floating_net", True):
        floating: set[str] = set()
        for key in ("no_dc_path_to_ground", "floating_single_terminal_net"):
            if key in by_key:
                floating.update(by_key[key].location.get("nets", []))
        floating.discard(GROUND_NET)
        if floating:
            adj = _full_net_adjacency(components)
            for group in _floating_subnets(sorted(floating), adj):
                rep = group[0]
                extra_lines.append(f"Rshunt_{_san(rep)} {rep} 0 {shunt_r:g}")
                add(LedgerEntry(
                    issue="no_dc_path_to_ground", category="assumption",
                    behavior_changing=True,
                    action=f"added shunt R {shunt_r:g} from {rep} to 0",
                    location={"nets": group},
                    alternatives=["gmin step", f".nodeset {rep}=0"],
                    confidence=0.6,
                    evidence=f"subnet {group} has no DC path to reference",
                ))

    # ideal inductor: logged, but no edit — a DC short is well-posed unless
    # it forms a source loop; keeping it a no-op is the minimal choice
    if strategies.get("inductor_series_r", True) and "ideal_inductor_dc" in by_key:
        iss = by_key["ideal_inductor_dc"]
        add(LedgerEntry(
            issue=iss.issue, category="assumption", behavior_changing=False,
            action="none applied (DC short is well-posed); .nodeset available if singular",
            location=iss.location, alternatives=["series Rdc", ".nodeset"],
            confidence=0.9, evidence=iss.evidence,
        ))

    # unsnapped terminals: a topology gap. Repair MUST NOT add wires, so this
    # is flagged, not fixed — transparency without touching topology.
    if "unsnapped_terminal" in by_key:
        iss = by_key["unsnapped_terminal"]
        add(LedgerEntry(
            issue=iss.issue, category="assumption", behavior_changing=False,
            action="flagged only — not auto-repaired (would change topology)",
            location=iss.location, alternatives=[],
            confidence=1.0, evidence=iss.evidence,
        ))

    num_assumptions = sum(1 for e in entries if e.category == "assumption")
    num_gauge = sum(1 for e in entries if e.category == "gauge")
    return RepairResult(entries, extra_lines, num_assumptions, num_gauge)


def _san(net: str) -> str:
    return str(net).replace(" ", "_").replace("-", "_")


def build_ledger(
    image: str, solvable_before: bool | None, solvable_after: bool | None,
    result: RepairResult,
) -> dict:
    """Assemble the ledger artifact (schema v1)."""
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "image": image,
        "solvable_before": solvable_before,
        "solvable_after": solvable_after,
        "num_assumptions": result.num_assumptions,
        "num_gauge": result.num_gauge,
        "entries": [asdict(e) for e in result.entries],
    }


def export_ledger(ledger: dict, path: str) -> None:
    """Write the ledger as JSON plus a human-readable sidecar."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(ledger, f, indent=2)

    readable = p.with_suffix(".txt")
    with open(readable, "w") as f:
        f.write(f"=== ASSUMPTION LEDGER: {ledger['image']} ===\n")
        f.write(f"solvable before repair: {ledger['solvable_before']}\n")
        f.write(f"solvable after  repair: {ledger['solvable_after']}\n")
        f.write(f"gauge (safe): {ledger['num_gauge']}   "
                f"assumptions (flagged): {ledger['num_assumptions']}\n\n")
        for e in ledger["entries"]:
            tag = "[ASSUMPTION]" if e["category"] == "assumption" else "[gauge]"
            f.write(f"{tag} {e['issue']}\n")
            f.write(f"    action     : {e['action']}\n")
            if e["alternatives"]:
                f.write(f"    alternatives: {', '.join(e['alternatives'])}\n")
            f.write(f"    confidence : {e['confidence']}   "
                    f"behavior-changing: {e['behavior_changing']}\n")
            if e["evidence"]:
                f.write(f"    evidence   : {e['evidence']}\n")
            f.write("\n")
