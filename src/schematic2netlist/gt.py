"""Ground-truth topology graphs: schema, loader, validator, bootstrap.

GT files live at ``data/gt_netlists/<image stem>.json`` (schema v1)::

    {
      "schema_version": 1,
      "image": "circuit_1199.jpg",
      "source": "pipeline_bootstrap" | "manual",
      "verified": false,          # flipped to true only by a human
      "annotator": null | "name",
      "notes": "",
      "components": [
        {
          "id": 0,
          "class": "resistor",
          "bbox": [cx, cy, w, h],          # image px, center-based (optional)
          "terminals": [
            {"index": 0, "net": "n1"},
            {"index": 1, "net": "0"}       # ground net is always "0"
          ]
        },
        ...
      ]
    }

A terminal's ``net`` may be null in unverified bootstrap files (the
pipeline failed to snap it); verified GT must name a net for every
terminal, unless the component record carries ``"unconnected": true``
(a deliberately dangling element drawn in the source image).

The loader converts GT into the component-list graph format used by
:mod:`schematic2netlist.metrics` ({"id", "class", "nets": [...]}).
"""

from __future__ import annotations

import json
from pathlib import Path

from schematic2netlist.classes import class_role, class_terminals, is_ground

SCHEMA_VERSION = 1

GROUND_NET = "0"


def load_gt(path: str | Path) -> dict:
    with open(path) as f:
        gt = json.load(f)
    return gt


def save_gt(gt: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f, indent=2)


def gt_to_components(gt: dict) -> list[dict]:
    """GT dict -> metrics graph format [{"id", "class", "nets": [...]}]."""
    comps = []
    for c in gt["components"]:
        nets = [None] * len(c["terminals"])
        for t in c["terminals"]:
            nets[t["index"]] = t["net"]
        comps.append({"id": c["id"], "class": c["class"], "nets": nets})
    return comps


def validate_gt(
    gt: dict,
    class_whitelist: set[str] | None = None,
    strict: bool | None = None,
) -> list[str]:
    """Validate a GT dict. Returns a list of issues (empty == valid).

    ``strict`` defaults to the file's own "verified" flag: a verified
    file must be fully connected and internally consistent; an
    unverified bootstrap may contain null nets.
    """
    issues: list[str] = []

    for key in ("schema_version", "image", "components"):
        if key not in gt:
            issues.append(f"missing required key: {key}")
    if issues:
        return issues

    if gt["schema_version"] != SCHEMA_VERSION:
        issues.append(
            f"schema_version {gt['schema_version']} != {SCHEMA_VERSION}"
        )
    if strict is None:
        strict = bool(gt.get("verified", False))

    ids = [c.get("id") for c in gt["components"]]
    if len(ids) != len(set(ids)):
        issues.append("duplicate component ids")

    net_names: set[str] = set()
    for c in gt["components"]:
        cid = c.get("id", "?")
        if "class" not in c or not isinstance(c["class"], str):
            issues.append(f"component {cid}: missing/invalid class")
            continue
        if class_whitelist and c["class"] not in class_whitelist:
            issues.append(f"component {cid}: unknown class {c['class']!r}")
        terminals = c.get("terminals")
        if not isinstance(terminals, list) or not terminals:
            issues.append(f"component {cid}: missing terminals")
            continue
        indices = [t.get("index") for t in terminals]
        if sorted(indices) != list(range(len(terminals))):
            issues.append(
                f"component {cid}: terminal indices {indices} are not 0..{len(terminals) - 1}"
            )
        role = class_role(c["class"])
        if role == "none":
            issues.append(
                f"component {cid}: {c['class']!r} is a drawing annotation, "
                "not an electrical component — remove it from topology GT"
            )
            continue
        if role != "unknown":
            expected = class_terminals(c["class"])
            if len(terminals) != expected:
                issues.append(
                    f"component {cid}: {len(terminals)} terminal(s), "
                    f"expected {expected} for class {c['class']!r}"
                )
        for t in terminals:
            net = t.get("net")
            if net is None:
                if strict and not c.get("unconnected", False):
                    issues.append(
                        f"component {cid}: terminal {t.get('index')} has no net "
                        "(verified GT must be fully connected or marked unconnected)"
                    )
                continue
            if not isinstance(net, str) or not net.strip():
                issues.append(f"component {cid}: invalid net name {net!r}")
                continue
            net_names.add(net)
        if is_ground(c["class"]) and strict:
            g_net = terminals[0].get("net")
            if g_net is not None and g_net != GROUND_NET:
                issues.append(
                    f"component {cid}: ground symbol on net {g_net!r}, must be '{GROUND_NET}'"
                )

    if strict and net_names:
        # every net should touch >= 2 terminals in a verified graph,
        # otherwise it is a floating single-terminal net (usually a typo)
        counts: dict[str, int] = {}
        for c in gt["components"]:
            for t in c["terminals"]:
                if t.get("net"):
                    counts[t["net"]] = counts.get(t["net"], 0) + 1
        for net, n in sorted(counts.items()):
            if n < 2:
                issues.append(f"net {net!r} touches only {n} terminal (suspicious)")

    return issues


def bootstrap_from_pipeline(image_name: str, pipeline_result: dict) -> dict:
    """Create an unverified GT skeleton from a pipeline run for human
    correction — much faster than annotating from zero.

    Terminal count follows the class (1 for ground/one-port sources,
    3 for transistors/op-amps); nets beyond what snapping produced are
    left null for the annotator to fill.
    """
    detections = pipeline_result["detections"]
    components = []
    for c in pipeline_result["components"]:
        det = detections[c["id"]]
        names = c.get("node_names", [None, None])
        n_terms = class_terminals(c["class"])
        terminals = [
            {"index": i, "net": names[i] if i < len(names) else None}
            for i in range(n_terms)
        ]
        components.append(
            {
                "id": c["id"],
                "class": c["class"],
                "bbox": [det["x"], det["y"], det["width"], det["height"]],
                "terminals": terminals,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "image": image_name,
        "source": "pipeline_bootstrap",
        "verified": False,
        "annotator": None,
        "notes": "",
        "components": components,
    }
