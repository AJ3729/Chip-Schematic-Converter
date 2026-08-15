#!/usr/bin/env python3
"""Evaluable-pool coverage under the B2 class map.

A circuit can be fairly scored by a 17-class pipeline only if every component
in it maps into the pipeline's vocabulary. This counts, per circuit and per
drafter, how many do -- and reports the exclusions rather than quietly dropping
them.

AMBIGUOUS counts as excluding, because an ambiguous mapping would silently
change the circuit (see `vss` and `capacitor.polarized` in the map). The two
exclusion causes are tallied separately so the cost of the ambiguity is
visible.

Usage:
    python scripts/cghd_coverage.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CGHD = ROOT / "data/cghd/extracted"
MAP = ROOT / "spec/class_map_cghd.yaml"
OUT = ROOT / "results/cghd_coverage.json"

# drafter_0 is excluded: 917 of its 1,038 images carry no annotation at all
# (reports/cghd_inventory.md section 4).
DRAFTERS = range(1, 25)


def load_map() -> tuple[dict, dict]:
    m = yaml.safe_load(MAP.read_text())
    return m["mapping"], m["coarse_groups"]


def classify(target: str) -> str:
    if target == "NOT_A_COMPONENT":
        return "structural"
    if target == "OUT_OF_VOCABULARY":
        return "oov"
    if target == "AMBIGUOUS":
        return "ambiguous"
    return "in_vocab"


def main() -> None:
    if not CGHD.is_dir():
        sys.exit(f"CGHD not extracted at {CGHD}")
    mapping, _ = load_map()

    per_circuit: dict[str, dict] = {}
    unmapped: collections.Counter = collections.Counter()
    class_freq: collections.Counter = collections.Counter()

    for d in DRAFTERS:
        adir = CGHD / f"drafter_{d}" / "annotations"
        if not adir.is_dir():
            continue
        for xml in sorted(adir.glob("*.xml")):
            names = [o.findtext("name", "").strip()
                     for o in ET.parse(xml).getroot().findall("object")]
            class_freq.update(names)
            kinds: collections.Counter = collections.Counter()
            for n in names:
                if n not in mapping:
                    unmapped[n] += 1
                    kinds["unmapped"] += 1
                    kinds["unmapped_det"] += 1
                    continue
                k = classify(mapping[n]["to"])
                kinds[k] += 1
                # A class may be ambiguous for electrical treatment while its
                # box and coarse class are determinate. Detection can score
                # those; netlist scoring cannot.
                if k == "ambiguous" and mapping[n].get("detection_ok"):
                    kinds["in_vocab_det"] += 1
                else:
                    kinds[f"{k}_det"] += 1
            stem = xml.stem
            g = re.match(r"(C(\d+)_D(\d+))_P(\d+)", stem)
            per_circuit[f"drafter_{d}/{stem}"] = {
                "drafter": d,
                "drawing_group": f"drafter_{d}/{g.group(1)}" if g else None,
                "picture": int(g.group(4)) if g else None,
                "n_objects": len(names),
                **{k: kinds.get(k, 0) for k in
                   ("in_vocab", "structural", "oov", "ambiguous", "unmapped")},
                "evaluable": kinds.get("oov", 0) == 0
                             and kinds.get("ambiguous", 0) == 0
                             and kinds.get("unmapped", 0) == 0,
                "evaluable_detection": kinds.get("oov_det", 0) == 0
                             and kinds.get("ambiguous_det", 0) == 0
                             and kinds.get("unmapped_det", 0) == 0,
            }

    n = len(per_circuit)
    ev = [v for v in per_circuit.values() if v["evaluable"]]
    evd = [v for v in per_circuit.values() if v["evaluable_detection"]]
    groups = {v["drawing_group"] for v in per_circuit.values() if v["drawing_group"]}
    ev_groups = {v["drawing_group"] for v in ev if v["drawing_group"]}
    # a drawing group is fully evaluable only if EVERY capture of it is
    full_groups = {g for g in ev_groups
                   if all(v["evaluable"] for v in per_circuit.values()
                          if v["drawing_group"] == g)}
    det_groups = {g for g in groups
                  if all(v["evaluable_detection"] for v in per_circuit.values()
                         if v["drawing_group"] == g)}

    by_drafter = collections.defaultdict(lambda: [0, 0])
    for v in per_circuit.values():
        by_drafter[v["drafter"]][1] += 1
        by_drafter[v["drafter"]][0] += v["evaluable"]

    excl_oov = sum(1 for v in per_circuit.values() if v["oov"])
    excl_amb = sum(1 for v in per_circuit.values()
                   if not v["oov"] and v["ambiguous"])

    out = {
        "_what": "Evaluable pool under spec/class_map_cghd.yaml. A circuit is "
                 "evaluable only if every annotated object maps into the "
                 "pipeline's 17-class vocabulary. AMBIGUOUS excludes, because "
                 "an ambiguous mapping would silently change the circuit.",
        "cghd_version": 12,
        "drafters_considered": list(DRAFTERS),
        "images_total": n,
        "images_evaluable_netlist": len(ev),
        "evaluable_fraction_netlist": round(len(ev) / n, 4) if n else 0.0,
        "images_evaluable_detection": len(evd),
        "evaluable_fraction_detection": round(len(evd) / n, 4) if n else 0.0,
        "excluded_for_out_of_vocabulary": excl_oov,
        "excluded_for_ambiguous_only": excl_amb,
        "drawing_groups_total": len(groups),
        "drawing_groups_fully_evaluable_netlist": len(full_groups),
        "drawing_groups_fully_evaluable_detection": len(det_groups),
        "fully_evaluable_drawing_groups_detection": sorted(det_groups),
        "per_drafter_evaluable": {str(k): {"evaluable": v[0], "total": v[1]}
                                  for k, v in sorted(by_drafter.items())},
        "cghd_class_frequency": dict(class_freq.most_common()),
        "unmapped_classes_encountered": dict(unmapped),
        "evaluable_images_netlist": sorted(k for k, v in per_circuit.items() if v["evaluable"]),
        "evaluable_images_detection": sorted(k for k, v in per_circuit.items() if v["evaluable_detection"]),
        "fully_evaluable_drawing_groups": sorted(full_groups),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"images                     {n}")
    print(f"evaluable, NETLIST         {len(ev)}  ({len(ev)/n:.1%})")
    print(f"evaluable, DETECTION       {len(evd)}  ({len(evd)/n:.1%})")
    print(f"  excluded, out-of-vocab   {excl_oov}")
    print(f"  excluded, ambiguous only {excl_amb}")
    print(f"drawing groups             {len(groups)}")
    print(f"  fully evaluable, netlist   {len(full_groups)}")
    print(f"  fully evaluable, detection {len(det_groups)}")
    if unmapped:
        print(f"\nUNMAPPED CLASSES (map is incomplete): {dict(unmapped)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
