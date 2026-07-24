"""Node naming and SPICE netlist export.

Element choice is driven by each class's *role* (see
schematic2netlist.classes), which supports both the published
Digitize-HCD vocabulary and the legacy Roboflow names via aliases.
This replaces the legacy substring dispatch, fixing two documented v1
quirks: the "supply" branch that shadowed AC supplies (they were
emitted as DC), and the diode/Zener counter collision that could emit
duplicate D1 element names. Referenced device models now get .model
cards so D/M/Q netlists are not unconditionally unparsable.

The ground fallback policy is configurable: "most_connected" (v1)
assigns node "0" to the most-connected node when no ground symbol
snapped; "fail" (v2) raises instead.
"""

from __future__ import annotations

from collections import defaultdict

from schematic2netlist.classes import class_role, class_terminals, is_ground


class GroundNotFoundError(RuntimeError):
    """Raised when no ground node exists and the policy is 'fail'."""


def build_node_name_map(
    components_with_nodes: list[dict],
    ground_fallback: str = "most_connected",
) -> dict[int, str]:
    """Map raw wire-node ids to SPICE node names.

    Guarantees: the ground node is named "0"; all other nodes are
    "n1", "n2", ... ("n0" is never used, avoiding collision with "0").
    """
    ground_raw_id = None
    for c in components_with_nodes:
        if is_ground(c["class"]):
            if c["nodes"][0] is not None:
                ground_raw_id = c["nodes"][0]
                break

    if ground_raw_id is None:
        if ground_fallback == "fail":
            raise GroundNotFoundError(
                "No ground-connected node found (ground symbol did not "
                "touch any wire node)."
            )
        # most_connected fallback: treat the node touching the most
        # component terminals as the reference node.
        node_counts: dict[int, int] = defaultdict(int)
        for c in components_with_nodes:
            for n in c["nodes"]:
                if n is not None:
                    node_counts[n] += 1
        if node_counts:
            ground_raw_id = max(node_counts, key=node_counts.get)
        else:
            ground_raw_id = 0

    all_raw_nodes = sorted(
        {
            n
            for c in components_with_nodes
            for n in c["nodes"]
            if n is not None
        }
    )

    node_name_map: dict[int, str] = {}
    counter = 1
    for raw_id in all_raw_nodes:
        if raw_id == ground_raw_id:
            node_name_map[raw_id] = "0"
        else:
            node_name_map[raw_id] = f"n{counter}"
            counter += 1
    return node_name_map


def assign_node_names(
    components_with_nodes: list[dict], node_name_map: dict[int, str]
) -> None:
    """Attach 'node_names' to each component record, in place."""
    for c in components_with_nodes:
        c["node_names"] = [
            node_name_map[n] if n is not None else None for n in c["nodes"]
        ]


def export_readable_netlist(
    components_with_nodes: list[dict], out_path: str
) -> None:
    with open(out_path, "w") as f:
        f.write("=== NETLIST (NO OCR VALUES) ===\n")
        f.write("nodes derived from wire connected-components\n\n")
        for c in components_with_nodes:
            f.write(
                f"ID {c['id']:02d}  "
                f"{c['class']:<28}  "
                f"raw={c['nodes']}  "
                f"names={c['node_names']}\n"
            )


# role -> (element prefix, model name or None). Diode and Zener share
# the D prefix AND the D counter (fixing the legacy duplicate-name bug).
_TWO_TERMINAL_ROLES = {
    "resistor": ("R", None),
    "capacitor": ("C", None),
    "inductor": ("L", None),
    "vdc": ("V", None),
    "vac": ("V", None),
    "idc": ("I", None),
    "iac": ("I", None),
    "diode": ("D", "Ddefault"),
    "zener": ("D", "Zdefault"),
}

_MODEL_CARDS = {
    "Ddefault": ".model Ddefault D",
    "Zdefault": ".model Zdefault D(bv=5.1)",
    "NMOSdefault": ".model NMOSdefault NMOS",
    "PMOSdefault": ".model PMOSdefault PMOS",
    "QNPNdefault": ".model QNPNdefault NPN",
    "QPNPdefault": ".model QPNPdefault PNP",
}

_ROLE_VALUE_KEYS = {
    "resistor": ("resistor", "1k"),
    "capacitor": ("capacitor", "1u"),
    "inductor": ("inductor", "1m"),
    "vdc": ("dc_supply", "DC 5"),
    "vac": ("ac_supply", "AC 1"),
    "vdc_oneport": ("dc_supply", "DC 5"),
    "idc": ("dc_current", "DC 1m"),
    "iac": ("ac_current", "AC 1m"),
}


def export_spice_netlist(
    components_with_nodes: list[dict],
    out_path: str,
    placeholders: dict | None = None,
    extra_lines: list[str] | None = None,
) -> dict:
    """Write a SPICE netlist with placeholder values (no OCR).

    Dispatch is role-based (see module docstring). Components with
    missing terminals are skipped with an UNSNAPPED comment; degenerate
    same-node two-terminal components with SAME_NODE_SKIPPED. Model
    cards for referenced device models are appended before .op.

    ``extra_lines`` are appended verbatim under a "repair" banner (used
    by the M4 repair layer to inject minimal SPICE aids without touching
    the component lines that encode topology).

    Returns {"wrote_any": bool, "skipped": [reason strings]}.
    """
    ph = placeholders or {}
    counters: dict[str, int] = defaultdict(int)
    skipped: list[str] = []
    lines: list[str] = []
    models_used: set[str] = set()

    def sanitize(node):
        if node is None:
            return None
        s = str(node).strip().replace(" ", "_")
        return s if s else None

    def value_for(role: str) -> str:
        key, default = _ROLE_VALUE_KEYS[role]
        return ph.get(key, default)

    for c in components_with_nodes:
        role = class_role(c["class"])

        # ground symbols are reference markers; crossovers are drawing
        # annotations — neither is a SPICE element
        if role in ("ground", "none"):
            continue

        names = [sanitize(n) for n in c["node_names"]]
        n_needed = class_terminals(c["class"])

        if role == "vdc_oneport":
            # one-port rail source: element between its net and ground
            a = next((n for n in names if n is not None), None)
            if a is None:
                msg = f"* UNSNAPPED {c['class']} raw_nodes={c['nodes']}"
                lines.append(msg)
                skipped.append(msg)
                continue
            if a == "0":
                msg = f"* SAME_NODE_SKIPPED {c['class']} both_on=0"
                lines.append(msg)
                skipped.append(msg)
                continue
            counters["V"] += 1
            lines.append(f"V{counters['V']} {a} 0 {value_for(role)}")
            continue

        usable = [n for n in names if n is not None]
        if len(usable) < n_needed:
            msg = f"* UNSNAPPED {c['class']} raw_nodes={c['nodes']}"
            lines.append(msg)
            skipped.append(msg)
            continue

        if role in _TWO_TERMINAL_ROLES:
            a, b = names[0], names[1]
            if a == b:
                msg = f"* SAME_NODE_SKIPPED {c['class']} both_on={a}"
                lines.append(msg)
                skipped.append(msg)
                continue
            prefix, model = _TWO_TERMINAL_ROLES[role]
            counters[prefix] += 1
            if model:
                models_used.add(model)
                lines.append(f"{prefix}{counters[prefix]} {a} {b} {model}")
            else:
                lines.append(f"{prefix}{counters[prefix]} {a} {b} {value_for(role)}")

        elif role in ("nmos", "pmos"):
            d, g, s = names[0], names[1], names[2]
            model = "NMOSdefault" if role == "nmos" else "PMOSdefault"
            models_used.add(model)
            counters["M"] += 1
            # body tied to source (no body terminal in hand drawings)
            lines.append(f"M{counters['M']} {d} {g} {s} {s} {model}")

        elif role in ("npn", "pnp"):
            cN, b, e = names[0], names[1], names[2]
            model = "QNPNdefault" if role == "npn" else "QPNPdefault"
            models_used.add(model)
            counters["Q"] += 1
            lines.append(f"Q{counters['Q']} {cN} {b} {e} {model}")

        elif role == "opamp":
            # terminals [in+, in-, out] (Digitize-HCD port order);
            # ideal VCVS with high open-loop gain
            inp, inn, out = names[0], names[1], names[2]
            counters["E"] += 1
            lines.append(f"E{counters['E']} {out} 0 {inp} {inn} 100k")

        else:
            counters["X"] += 1
            lines.append(f"* UNKNOWN {c['class']} {' '.join(usable)}")

    wrote_any = any(not ln.startswith("*") for ln in lines)

    with open(out_path, "w") as f:
        f.write("* Auto-generated SPICE netlist (NO TEXT OCR USED)\n\n")
        for ln in lines:
            f.write(ln + "\n")
        if not wrote_any:
            f.write("* WARNING: no valid components written\n")
        for model in sorted(models_used):
            f.write(_MODEL_CARDS[model] + "\n")
        if extra_lines:
            f.write("\n* --- design-intent repair (does not change topology) ---\n")
            for ln in extra_lines:
                f.write(ln + "\n")
        f.write("\n.op\n.end\n")

    return {"wrote_any": wrote_any, "skipped": skipped}
