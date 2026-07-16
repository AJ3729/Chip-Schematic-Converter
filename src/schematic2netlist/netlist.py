"""Node naming and SPICE netlist export.

Migrated verbatim from nodes_mapping_and_netlist.py (v1). The ground
fallback policy is configurable: "most_connected" (v1) assigns node "0"
to the most-connected node when no ground symbol snapped; "fail" (v2)
raises instead.
"""

from __future__ import annotations

from collections import defaultdict


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
        if "ground" in c["class"].lower():
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


def export_spice_netlist(
    components_with_nodes: list[dict],
    out_path: str,
    placeholders: dict | None = None,
) -> dict:
    """Write a SPICE netlist with placeholder values (no OCR).

    Returns {"wrote_any": bool, "skipped": [reason strings]}.
    """
    ph = placeholders or {}
    counters: dict[str, int] = defaultdict(int)
    skipped: list[str] = []
    wrote_any = False

    def sanitize(node):
        if node is None:
            return None
        s = str(node).strip().replace(" ", "_")
        return s if s else None

    with open(out_path, "w") as f:
        f.write("* Auto-generated SPICE netlist (NO TEXT OCR USED)\n\n")

        for c in components_with_nodes:
            cls = c["class"].lower()

            # ground symbols are reference markers only — not SPICE elements
            if "ground" in cls:
                continue

            a = sanitize(c["node_names"][0])
            b = sanitize(c["node_names"][1])

            if a is None or b is None:
                msg = f"* UNSNAPPED {c['class']} raw_nodes={c['nodes']}"
                f.write(msg + "\n")
                skipped.append(msg)
                continue

            if a == b:
                msg = f"* SAME_NODE_SKIPPED {c['class']} both_on={a}"
                f.write(msg + "\n")
                skipped.append(msg)
                continue

            if "resistor" in cls:
                counters["R"] += 1
                f.write(f"R{counters['R']} {a} {b} {ph.get('resistor', '1k')}\n")
                wrote_any = True

            elif "capacitor" in cls:
                counters["C"] += 1
                f.write(f"C{counters['C']} {a} {b} {ph.get('capacitor', '1u')}\n")
                wrote_any = True

            elif "inductor" in cls:
                counters["L"] += 1
                f.write(f"L{counters['L']} {a} {b} {ph.get('inductor', '1m')}\n")
                wrote_any = True

            # NOTE (verbatim v1 behavior): this branch matches any class
            # containing "supply", so "AC Supply" is emitted as a DC source
            # and the ac_supply branch below is unreachable. Preserved for
            # output parity; slated for the post-migration refactor.
            elif "dc supply" in cls or "supply" in cls:
                counters["V"] += 1
                f.write(f"V{counters['V']} {a} {b} {ph.get('dc_supply', 'DC 5')}\n")
                wrote_any = True

            elif "ac supply" in cls:
                counters["V"] += 1
                f.write(f"V{counters['V']} {a} {b} {ph.get('ac_supply', 'AC 1')}\n")
                wrote_any = True

            elif "dc current" in cls or "independent dc current" in cls:
                counters["I"] += 1
                f.write(f"I{counters['I']} {a} {b} {ph.get('dc_current', 'DC 1m')}\n")
                wrote_any = True

            elif "ac current" in cls or "independent ac current" in cls:
                counters["I"] += 1
                f.write(f"I{counters['I']} {a} {b} {ph.get('ac_current', 'AC 1m')}\n")
                wrote_any = True

            elif "diode" in cls and "zener" not in cls:
                counters["D"] += 1
                f.write(f"D{counters['D']} {a} {b} {ph.get('diode_model', 'Ddefault')}\n")
                wrote_any = True

            # NOTE (verbatim v1 behavior): zener uses its own counter but
            # still writes a D-prefixed element, so a diode and a zener in
            # one circuit can collide as duplicate "D1" element names.
            # Preserved for output parity; slated for refactor.
            elif "zener" in cls:
                counters["Z"] += 1
                f.write(f"D{counters['Z']} {a} {b} {ph.get('zener_model', 'Zdefault')}\n")
                wrote_any = True

            elif "mosfet" in cls:
                counters["M"] += 1
                # MOSFET needs drain gate source body — approximated here
                f.write(f"M{counters['M']} {a} {b} 0 0 {ph.get('mosfet_model', 'NMOS')}\n")
                wrote_any = True

            else:
                counters["X"] += 1
                f.write(f"* UNKNOWN {c['class']} {a} {b}\n")

        if not wrote_any:
            f.write("* WARNING: no valid components written\n")

        f.write("\n.op\n.end\n")

    return {"wrote_any": wrote_any, "skipped": skipped}
