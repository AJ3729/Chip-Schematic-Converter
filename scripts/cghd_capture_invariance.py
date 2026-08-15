#!/usr/bin/env python3
"""Capture invariance on CGHD (task B7).

CGHD photographs each physical drawing four times under different camera
positions and illuminations. That gives a robustness experiment that needs
**no ground truth at all**: run the frozen pipeline on every capture of a
drawing and ask whether it reconstructs the same circuit.

THE DISTINCTION THIS MEASURES, and why it is not the determinism result
already in the paper:

  determinism         identical input -> identical output.
                      A property of the software. Already measured:
                      byte-identical over five runs, 0/192 topology changes.

  capture invariance  the same drawing, photographed differently -> the same
                      circuit. A property of the whole system under real use,
                      and strictly harder.

A system can be perfectly deterministic and completely capture-variant. Only
the second speaks to whether the tool is usable on a photograph someone
actually took.

Topology is compared naming-invariantly: the partition of
(component_id, terminal_index) pairs into nets. Renaming a net is not a change.
Because component ids are per-image detections, groups are compared through the
*structure* of the partition rather than through ids, so a differing detection
count is itself a disagreement -- which is correct, since it means the two
photographs yielded different circuits.

Usage:
    python scripts/cghd_capture_invariance.py
    python scripts/cghd_capture_invariance.py --limit-groups 20
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from schematic2netlist.classes import canonical_class  # noqa: E402
from schematic2netlist.config import load_config  # noqa: E402
from schematic2netlist.detect import detect_ultralytics  # noqa: E402
from schematic2netlist.pipeline import run_pipeline  # noqa: E402
from stats.bootstrap import bootstrap_rate  # noqa: E402

IMG = ROOT / "data/cghd_1024/images"
ANN = ROOT / "data/cghd_1024/annotations"
CACHE = ROOT / "data/cghd_1024/detections"
OUT = ROOT / "results/cghd_capture_invariance.json"


def topology_signature(res: dict) -> tuple:
    """Naming-invariant structural signature of a reconstruction.

    Two captures agree when their components partition into nets the same way,
    with the same class multiset. Net labels and component ids are arbitrary
    and must not affect the answer.
    """
    comps = res.get("components") or []
    # (class, sorted tuple of net-slot indices) keyed by net identity
    nets: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    classes: collections.Counter = collections.Counter()
    for c in comps:
        cls = canonical_class(c.get("class", ""))
        classes[cls] += 1
        for i, n in enumerate(c.get("node_names") or []):
            if n is not None:
                nets[str(n)].append((cls, i))
    # canonical: a frozenset of sorted member-tuples, itself sorted
    part = sorted(tuple(sorted(v)) for v in nets.values())
    return (tuple(sorted(classes.items())), tuple(part))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-groups", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    # Point the pipeline at the CGHD frames. This changes WHICH IMAGES are
    # read; it changes no threshold, weight or parameter. The freeze forbids
    # tuning, not evaluating on other data.
    cfg["preprocess"]["images_dir"] = "data/cghd_1024/images"
    cfg["detect"]["cache_dir"] = str(CACHE.relative_to(ROOT))
    CACHE.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for f in sorted(ANN.glob("*.json")):
        d = json.loads(f.read_text())
        groups[d["drawing_group"]].append(f.stem)
    multi = {g: sorted(v) for g, v in groups.items() if len(v) >= 2}
    keys = sorted(multi)[: a.limit_groups] if a.limit_groups else sorted(multi)
    print(f"drawing groups with >=2 captures: {len(multi)}  "
          f"(evaluating {len(keys)})")

    # one detection pass over everything, cached
    todo = [s for g in keys for s in multi[g]
            if not (CACHE / f"{s}.json").exists()]
    if todo:
        print(f"detecting {len(todo)} frames...")
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            paths = [str(IMG / f"{s}.jpg") for s in chunk]
            for s, dets in zip(chunk, detect_ultralytics(paths, cfg)):
                (CACHE / f"{s}.json").write_text(json.dumps(dets))
            print(f"  ...{min(i+32, len(todo))}/{len(todo)}", flush=True)

    per_group, errors = {}, 0
    for gi, g in enumerate(keys, 1):
        sigs, ncomp = [], []
        for s in multi[g]:
            try:
                dets = json.loads((CACHE / f"{s}.json").read_text())
                res = run_pipeline(IMG / f"{s}.jpg", cfg, detections=dets)
                sigs.append(topology_signature(res))
                ncomp.append(len(res.get("components") or []))
            except Exception:                               # noqa: BLE001
                sigs.append(None)
                ncomp.append(-1)
                errors += 1
        ok = [s for s in sigs if s is not None]
        pairs = list(itertools.combinations(range(len(sigs)), 2))
        agree_pairs = sum(1 for i, j in pairs
                          if sigs[i] is not None and sigs[j] is not None
                          and sigs[i] == sigs[j])
        per_group[g] = {
            "n_captures": len(sigs),
            "n_ok": len(ok),
            "all_agree": len(ok) == len(sigs) and len(set(ok)) == 1,
            "distinct_topologies": len(set(ok)),
            "pairwise_agree": agree_pairs,
            "pairwise_total": len(pairs),
            "component_counts": ncomp,
        }
        if gi % 20 == 0:
            print(f"  ...{gi}/{len(keys)} groups", flush=True)

    n = len(per_group)
    all_agree = [v["all_agree"] for v in per_group.values()]
    pa = sum(v["pairwise_agree"] for v in per_group.values())
    pt = sum(v["pairwise_total"] for v in per_group.values())
    ci = bootstrap_rate(all_agree, seed=0)

    out = {
        "_what": "Capture invariance: does the frozen pipeline reconstruct the "
                 "same circuit from different photographs of the same physical "
                 "drawing? Needs no ground truth.",
        "_distinct_from_determinism": (
            "Determinism is identical input -> identical output, a property of "
            "the software, already measured at 192/192 byte-identical and "
            "0/192 topology changes over five runs. Capture invariance is the "
            "same drawing photographed differently -> the same circuit, a "
            "property of the whole system under real use, and strictly "
            "harder."),
        "_polarity_caveat": (
            "CGHD does not annotate transistor polarity, so no polarity or "
            "pin-order claim is made or measurable here. Those rest on "
            "Digitize-HCD alone."),
        "cghd_version": 12,
        "n_groups": n,
        "captures_per_group": sorted({v["n_captures"] for v in per_group.values()}),
        "groups_all_captures_agree": int(sum(all_agree)),
        "fraction_all_agree": float(sum(all_agree) / n) if n else 0.0,
        "fraction_all_agree_ci95": [ci.lo, ci.hi],
        "pairwise_topology_agreement": pa / pt if pt else 0.0,
        "pairwise_pairs": pt,
        "pipeline_errors": errors,
        "distinct_topologies_histogram": dict(collections.Counter(
            v["distinct_topologies"] for v in per_group.values())),
        "per_group": per_group,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"\ngroups                      {n}")
    print(f"all captures agree          {sum(all_agree)}/{n} = "
          f"{out['fraction_all_agree']:.4f}  CI95 [{ci.lo:.4f}, {ci.hi:.4f}]")
    print(f"pairwise topology agreement {out['pairwise_topology_agreement']:.4f} "
          f"over {pt} pairs")
    print(f"pipeline errors             {errors}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
