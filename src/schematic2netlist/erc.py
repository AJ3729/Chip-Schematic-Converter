"""Electrical-rule checks (ERC) on the recovered circuit graph.

This is the *diagnosis* half of the M4 design-intent-completion layer
(contribution C5). It inspects the topology the pipeline recovered and
reports structural/electrical reasons the circuit would fail to
simulate — WITHOUT changing the topology. The *repair* half
(:mod:`schematic2netlist.repair`) turns each issue into a minimal,
logged fix.

An ``Issue`` names a problem, whether fixing it would change circuit
behavior, and where it is. Categorization into ``gauge`` (behavior-
invariant, safe to infer) vs ``assumption`` (behavior-changing, must be
flagged) is assigned by the repair strategy, not here — ERC only
diagnoses.

DC-path reasoning (the cause of most real ngspice failures) treats
components by whether they conduct at DC:

- conduct: resistor, inductor (short at DC), voltage sources, diodes,
  and the drain-source / collector-emitter channel of transistors;
- do NOT conduct at DC: capacitor (open), current source (open),
  transistor gate/base (high impedance).

This is a documented approximation adequate for ERC; it deliberately
mirrors why ngspice reports "no DC path to ground" / singular matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schematic2netlist.classes import class_role

GROUND_NET = "0"

# roles whose terminals provide a DC conduction path between their nets
_DC_TWO_TERMINAL = {"resistor", "inductor", "vdc", "vac", "vdc_oneport", "diode", "zener"}
# transistor channel: terminals (0, 2) conduct; terminal 1 (gate/base) does not
_TRANSISTOR = {"nmos", "pmos", "npn", "pnp"}


@dataclass
class Issue:
    issue: str                      # machine key, e.g. "no_dc_path_to_ground"
    behavior_changing: bool         # would repairing it alter circuit behavior?
    location: dict = field(default_factory=dict)   # {"nets": [...], "components": [...]}
    evidence: str = ""


def _present_nets(components: list[dict]) -> set[str]:
    nets = set()
    for c in components:
        for n in c.get("node_names", []):
            if n is not None:
                nets.add(n)
    return nets


def _dc_adjacency(components: list[dict]) -> dict[str, set[str]]:
    """Undirected net graph over DC-conducting connections only."""
    adj: dict[str, set[str]] = {}

    def link(a, b):
        if a is None or b is None or a == b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for c in components:
        role = class_role(c["class"])
        names = c.get("node_names", [])
        if role in _DC_TWO_TERMINAL and len(names) >= 2:
            link(names[0], names[1])
        elif role in _TRANSISTOR and len(names) >= 3:
            link(names[0], names[2])          # drain-source / collector-emitter
        elif role == "opamp" and len(names) >= 3:
            # output is driven; treat all three as weakly linked for reachability
            link(names[0], names[2])
            link(names[1], names[2])
    return adj


def _reachable_from_ground(adj: dict[str, set[str]]) -> set[str]:
    if GROUND_NET not in adj:
        return set()
    seen = {GROUND_NET}
    stack = [GROUND_NET]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def run_erc(components: list[dict], node_name_map: dict, cfg: dict) -> list[Issue]:
    """Diagnose simulability issues on the recovered graph (no mutation)."""
    issues: list[Issue] = []
    nets = _present_nets(components)

    # 1. No ground reference anywhere.
    has_ground_symbol = any(class_role(c["class"]) == "ground" for c in components)
    if GROUND_NET not in nets:
        issues.append(Issue(
            "no_ground_reference",
            behavior_changing=not has_ground_symbol,
            location={"nets": []},
            evidence=("a GND symbol exists but did not snap to a net"
                      if has_ground_symbol else
                      "no GND symbol and no net named '0'"),
        ))

    # 2. Unsnapped / dangling terminals (topology gaps, not repairs).
    dangling = [
        c["id"] for c in components
        if class_role(c["class"]) not in ("ground", "none")
        and any(n is None for n in c.get("node_names", []))
    ]
    if dangling:
        issues.append(Issue(
            "unsnapped_terminal",
            behavior_changing=False,
            location={"components": dangling},
            evidence=f"{len(dangling)} component(s) have an unsnapped terminal",
        ))

    # 3. Nets with no DC path to ground (only meaningful once a ground exists).
    if GROUND_NET in nets:
        adj = _dc_adjacency(components)
        reachable = _reachable_from_ground(adj)
        floating = sorted(n for n in nets if n != GROUND_NET and n not in reachable)
        if floating:
            issues.append(Issue(
                "no_dc_path_to_ground",
                behavior_changing=True,
                location={"nets": floating},
                evidence=f"nets {floating} have no DC path to reference",
            ))

    # 4. Single-terminal nets (a net touched by exactly one terminal is a
    #    floating stub — mirrors gt.validate_gt's suspicious-net check).
    counts: dict[str, int] = {}
    for c in components:
        for n in c.get("node_names", []):
            if n is not None:
                counts[n] = counts.get(n, 0) + 1
    stubs = sorted(n for n, k in counts.items() if k < 2 and n != GROUND_NET)
    if stubs:
        issues.append(Issue(
            "floating_single_terminal_net",
            behavior_changing=True,
            location={"nets": stubs},
            evidence=f"nets {stubs} are touched by a single terminal",
        ))

    # 5. Ideal inductor: a short at DC, a frequent singular-matrix cause.
    inductors = [c["id"] for c in components if class_role(c["class"]) == "inductor"]
    if inductors:
        issues.append(Issue(
            "ideal_inductor_dc",
            behavior_changing=False,
            location={"components": inductors},
            evidence=f"{len(inductors)} ideal inductor(s) short at the DC operating point",
        ))

    # 6. Current sources have an undetermined reference direction (sign gauge).
    csources = [c["id"] for c in components if class_role(c["class"]) in ("idc", "iac")]
    if csources:
        issues.append(Issue(
            "unset_current_direction",
            behavior_changing=False,
            location={"components": csources},
            evidence=f"{len(csources)} current source(s) have only a sign convention",
        ))

    return issues
