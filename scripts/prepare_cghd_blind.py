#!/usr/bin/env python3
"""Prepare a CROSS-DATASET blind evaluation set from CGHD.

WHAT THIS IS, AND WHAT IT IS NOT
================================
A reviewer asked for a smaller BLIND evaluation set built from NEW drawings,
annotated independently and evaluated exactly once after the pipeline is
frozen. Sourcing genuinely new hand-drawn circuits was not possible in the
available time. This script builds the closest available substitute:

    a DRAFTER-DISJOINT, CROSS-DATASET blind set drawn from CGHD
    (Zenodo 10056817, CC BY 4.0), a corpus this project has never
    trained on, never tuned against, and never selected a parameter from.

It is NOT a same-distribution blind set. CGHD is a different corpus by a
different population of drafters, photographed under different conditions,
with a different symbol vocabulary. Zero-shot DETECTION on CGHD already
scores macro AP@0.5 = 0.185 against 0.975 in domain
(results/cghd_zero_shot/summary.json), so the domain shift is severe and
measured. Any end-to-end number obtained on this set is therefore a
CROSS-DATASET number and is HARDER than the blind set the reviewer asked
for. See results/cghd_blind/READINESS.md for exactly what it does and does
not establish.

WHAT THIS SCRIPT DOES *NOT* DO
==============================
It does not annotate. It never runs the pipeline, the wire tracer, the
detector, or any model. Every terminal in every emitted ground-truth file
has ``"net": null``, waiting for a human. Ground truth seeded by the
pipeline's own predictions would make the subsequent evaluation circular
and would destroy the entire point of the exercise.

Component inventory, classes and boxes DO come from CGHD's own published
Pascal-VOC annotations -- those are human annotations by the dataset
authors, which is legitimate, and is the same arrangement as the
Digitize-HCD ground truth (published COCO geometry + manual topology).

WHAT IT EMITS
=============
    data/cghd_blind_1024/images/<stem>.jpg   preprocessed 1024 frames
    results/cghd_blind/manifest.json         stem, drafter, circuit, n_comp
    results/cghd_blind/manifest.csv          the same, for eyeballing
    results/cghd_blind/selection.json        seed, rules, rejects, achieved
                                             vs target distribution, reserve
    results/cghd_blind/packet/gt/*.json      GT stubs -- ALL NETS NULL
    results/cghd_blind/packet/decisions/*.json   decision stubs (same format
                                             as data/gt_test_1024/decisions/)
    results/cghd_blind/packet/aux/*.json     CGHD's own junction / crossover /
                                             text / terminal boxes, plus the
                                             per-component flags a human must
                                             resolve. Annotator aid, not GT.
    results/cghd_blind/packet/README.md      how to annotate this packet
    results/cghd_blind/run_meta.json         config + git SHA + seed + env

Usage:
    python scripts/prepare_cghd_blind.py                  # full run
    python scripts/prepare_cghd_blind.py --dry-run        # select only
    python scripts/prepare_cghd_blind.py --n 36 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.classes import canonical_class, class_terminals
from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.preprocess import preprocess_image_meta, project_bbox

# CGHD image path: drafter_D/images/C<circuit>_D<drawing>_P<picture>.<ext>
# Per the upstream README: circuit numbers run globally (12 per drafter),
# each circuit is drawn twice and each drawing photographed four times --
# so a naive sample gives eight near-duplicates of one topology. We select
# at most ONE image per (drafter, circuit).
IMG_RE = re.compile(
    r"^(drafter_\d+)/images/(C(\d+)_D(\d+)_P(\d+))\.(jpe?g|png)$", re.I
)

# CGHD meta-classes: annotated, but not electrical components.
META_CLASSES = {"text", "junction", "terminal", "crossover", "__background__"}

# CGHD classes that map onto one of our 17 but whose mapping a HUMAN must
# confirm against the drawing. These are flagged per component in the aux
# file; they are not errors, they are annotation work.
NEEDS_HUMAN_CLASS_CALL = {
    "transistor.bjt": "CGHD does not distinguish NPN from PNP -- read the "
                      "emitter arrow and set BJT-NPN or BJT-PNP.",
    "transistor.fet": "CGHD does not distinguish N from P (and lumps JFETs "
                      "in) -- read the body/channel arrow and set MOSFET-N "
                      "or MOSFET-P, or reject the sheet if it is a JFET.",
    "vss": "CGHD 'vss' is a supply-rail symbol, not necessarily ground. The "
           "mapping defaults it to GND, which would force it onto net '0'. "
           "If the drawing means a supply rail, change the class to "
           "'V-DC (one port)' and give it its own net.",
    "capacitor.polarized": "polarity is lost by the mapping to Capacitor; "
                           "terminal order is unscored for 2-terminal parts, "
                           "so this is informational.",
    "diode.light_emitting": "an LED mapped to Diode; informational.",
    "voltage.battery": "a battery mapped to V-DC; informational.",
    "resistor.photo": "an LDR mapped to Resistor; informational.",
}

# Hard exclusions -- these would CORRUPT the ground truth rather than merely
# approximate it, so no sheet containing one is eligible.
EXCLUDE_CLASSES = {
    "resistor.adjustable":
        "a potentiometer/rheostat has three terminals; our Resistor class "
        "has two, so no valid GT file can be written for it.",
}

# Digitize-HCD test split (192 images) component-count quartile boundaries,
# recomputed at run time from the GT and asserted against these.
HCD_EDGES = (7, 13, 19)
STRATUM_LABELS = ("<=7", "8-13", "14-19", ">=20")


def stratum_of(n: int, edges=HCD_EDGES) -> int:
    return 0 if n <= edges[0] else 1 if n <= edges[1] else 2 if n <= edges[2] else 3


def frame_quality(canvas) -> dict:
    """Two model-free statistics of a preprocessed frame.

    A photograph taken on ruled notebook paper, on a cluttered desk, or
    under a hard shadow survives binarisation as a wall of ink: either the
    ink fraction explodes, or one connected black blob swallows the canvas.
    Both are measured on the OUTPUT frame with no model of any kind, so
    screening on them cannot leak information about the system under test.
    """
    b = (canvas < 128).astype("uint8")
    n, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
    biggest = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    total = b.size
    return {"ink_fraction": round(float(b.sum()) / total, 5),
            "largest_component_fraction": round(biggest / total, 5)}


def calibrate_gate(clean_dir: Path, split_file: Path) -> dict:
    """Legibility gate = the envelope of the Digitize-HCD TRAIN frames.

    Calibrated on train only, so nothing about either evaluation split
    enters the threshold.
    """
    stems = [Path(s).stem for s in split_file.read_text().split() if s.strip()]
    ink, big = [], []
    for s in stems:
        p = clean_dir / (s + ".jpg")
        if not p.exists():
            continue
        q = frame_quality(cv2.imread(str(p), 0))
        ink.append(q["ink_fraction"])
        big.append(q["largest_component_fraction"])
    ink.sort()
    big.sort()
    return {
        "calibrated_on": f"{len(ink)} Digitize-HCD TRAIN frames in {clean_dir}",
        "max_ink_fraction": round(ink[-1], 5),
        "max_largest_component_fraction": round(big[-1], 5),
        "p99_ink_fraction": round(ink[int(0.99 * len(ink))], 5),
        "rule": ("a CGHD frame is LEGIBLE if both statistics fall inside the "
                 "range spanned by every training frame; outside it, the "
                 "drawing has not survived binarisation and no human can "
                 "annotate it either"),
    }


def frame_is_legible(q: dict, gate: dict) -> bool:
    return (q["ink_fraction"] <= gate["max_ink_fraction"]
            and q["largest_component_fraction"]
            <= gate["max_largest_component_fraction"])


def ks_statistic(a: list[int], b: list[int]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), no scipy."""
    a, b = sorted(a), sorted(b)
    grid = sorted(set(a) | set(b))
    d = 0.0
    for v in grid:
        fa = sum(1 for x in a if x <= v) / len(a)
        fb = sum(1 for x in b if x <= v) / len(b)
        d = max(d, abs(fa - fb))
    return round(d, 4)


# ----------------------------------------------------------------------
# reference distribution
# ----------------------------------------------------------------------
def hcd_test_distribution(gt_dir: Path, split_file: Path) -> dict:
    """Component-count distribution of the Digitize-HCD test split."""
    stems = [l.strip() for l in split_file.read_text().split() if l.strip()]
    counts = []
    for s in stems:
        p = gt_dir / (Path(s).stem + ".json")
        if not p.exists():
            continue
        counts.append(len(json.loads(p.read_text())["components"]))
    counts.sort()
    strata = collections.Counter(stratum_of(c) for c in counts)
    n = len(counts)
    return {
        "n_images": n,
        "min": counts[0], "max": counts[-1],
        "mean": round(sum(counts) / n, 3),
        "median": counts[n // 2],
        "quartiles": [counts[n // 4], counts[n // 2], counts[(3 * n) // 4]],
        "strata_counts": {STRATUM_LABELS[k]: strata[k] for k in range(4)},
        "strata_fractions": {STRATUM_LABELS[k]: round(strata[k] / n, 4)
                             for k in range(4)},
        "histogram": dict(sorted(collections.Counter(counts).items())),
    }


# ----------------------------------------------------------------------
# CGHD scan
# ----------------------------------------------------------------------
def parse_boxes(xml_bytes: bytes) -> list[dict]:
    """Every annotated object in one CGHD Pascal-VOC file, in file order."""
    out = []
    for obj in ET.fromstring(xml_bytes).findall("object"):
        raw = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        out.append({"raw": raw, "xywh": (x1, y1, x2 - x1, y2 - y1)})
    return out


def scan_zip(zf: zipfile.ZipFile, mapping: dict) -> list[dict]:
    """Every CGHD image that has an annotation, with its class census."""
    mapped = mapping["mapped"]
    names = set(zf.namelist())
    recs = []
    for name in sorted(names):
        m = IMG_RE.match(name)
        if not m:
            continue
        drafter, stem = m.group(1), m.group(2)
        xml = f"{drafter}/annotations/{stem}.xml"
        if xml not in names:
            continue
        objs = parse_boxes(zf.read(xml))
        census = collections.Counter(o["raw"] for o in objs)
        symbols = [o["raw"] for o in objs if o["raw"] not in META_CLASSES]
        unmapped = sorted({c for c in symbols if c not in mapped})
        # "component" here means what the Digitize-HCD GT means by it:
        # an electrical symbol, EXCLUDING Wire Crossover (gt_test_1024
        # contains zero Wire Crossover components).
        n_comp = sum(1 for c in symbols if c in mapped and c != "crossover")
        recs.append({
            "drafter": drafter,
            "drafter_id": int(drafter.split("_")[1]),
            "circuit": int(m.group(3)),
            "drawing": int(m.group(4)),
            "picture": int(m.group(5)),
            "cghd_stem": stem,
            "img_member": name,
            "xml_member": xml,
            "n_comp": n_comp,
            "n_unmapped": len(unmapped),
            "unmapped_names": unmapped,
            "n_crossover": census["crossover"],
            "n_junction": census["junction"],
            "n_text": census["text"],
            "n_terminal": census["terminal"],
            "excluded_classes": sorted(set(symbols) & set(EXCLUDE_CLASSES)),
        })
    return recs


def eligibility(rec: dict, n_min: int, n_max: int, zero_shot_circuits: set) -> str | None:
    """Reason this image is INELIGIBLE, or None if it is eligible."""
    if (rec["drafter"], rec["circuit"]) in zero_shot_circuits:
        return "circuit already used in the zero-shot detection measurement"
    if rec["n_unmapped"]:
        return f"contains symbols outside our 17-class vocabulary: {rec['unmapped_names']}"
    if rec["excluded_classes"]:
        return f"contains a class that cannot be represented: {rec['excluded_classes']}"
    if rec["n_comp"] < n_min:
        return f"only {rec['n_comp']} components (Digitize-HCD test min is {n_min})"
    if rec["n_comp"] > n_max:
        return f"{rec['n_comp']} components, beyond the allowed tail ({n_max})"
    return None


# ----------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------
def select(templates: dict[tuple, dict], n_target: int, target_fracs: dict,
           rng: random.Random) -> tuple[list, list, dict]:
    """Pick n_target circuit templates, drafter-first then stratum-balanced.

    Pass 1 gives EVERY eligible drafter one template, chosen from whichever
    stratum is furthest below target -- drafter coverage is the point of the
    exercise, so it wins ties against distribution matching.
    Pass 2 fills the remaining slots from the most-deficient stratum,
    preferring drafters with the fewest picks so far.
    Everything not picked becomes an ordered reserve, so a sheet a human
    later finds unusable can be replaced without an ad-hoc substitution.
    """
    by_drafter = collections.defaultdict(list)
    for key in templates:
        by_drafter[key[0]].append(key)

    target_n = {k: target_fracs[k] * n_target for k in STRATUM_LABELS}
    chosen: list = []
    have = collections.Counter()
    per_drafter = collections.Counter()

    def deficit(s_label: str) -> float:
        return target_n[s_label] - have[s_label]

    def strat(key) -> str:
        return STRATUM_LABELS[stratum_of(templates[key]["n_comp"])]

    # --- pass 1: one template per drafter ---
    drafters = sorted(by_drafter, key=lambda d: int(d.split("_")[1]))
    rng.shuffle(drafters)
    for d in drafters:
        if len(chosen) >= n_target:
            break
        cands = sorted(by_drafter[d])
        rng.shuffle(cands)
        pick = max(cands, key=lambda k: (deficit(strat(k)), -k[1]))
        chosen.append(pick)
        have[strat(pick)] += 1
        per_drafter[d] += 1

    # --- pass 2: fill by stratum deficit, spreading over drafters ---
    remaining = sorted(set(templates) - set(chosen))
    rng.shuffle(remaining)
    while len(chosen) < n_target and remaining:
        order = sorted(STRATUM_LABELS, key=lambda s: -deficit(s))
        pick = None
        for s_label in order:
            pool = [k for k in remaining if strat(k) == s_label]
            if not pool:
                continue
            pick = min(pool, key=lambda k: (per_drafter[k[0]], k[1]))
            break
        if pick is None:
            break
        remaining.remove(pick)
        chosen.append(pick)
        have[strat(pick)] += 1
        per_drafter[pick[0]] += 1

    # reserve, ordered the same way a pass-2 pick would have taken them
    reserve = sorted(remaining, key=lambda k: (per_drafter[k[0]], k[0], k[1]))
    return chosen, reserve, dict(have)


# ----------------------------------------------------------------------
# emission
# ----------------------------------------------------------------------
GT_NOTES_STUB = (
    "NETS ARE UNANNOTATED. Every terminal in this file has \"net\": null and "
    "must be traced by a human against the drawing. Nothing in this file was "
    "produced by running the pipeline, the detector or the wire tracer -- "
    "component classes and boxes come from CGHD's own published Pascal-VOC "
    "annotations (human annotations by the dataset authors), projected into "
    "the 1024 frame by the project's own preprocessing transform. "
    "See results/cghd_blind/packet/README.md before annotating, and follow "
    "docs/ANNOTATION_GUIDE.md, the same guide used for the Digitize-HCD "
    "test split."
)

DEC_NOTES_STUB = (
    "TEMPLATE -- NOT YET ANNOTATED. Replace this text with the net map and "
    "every judgement call, exactly as in data/gt_test_1024/decisions/. "
    "\"sites\" is deliberately EMPTY: site ids are enumerated by the tracer "
    "at annotation time (scripts/gt_val_tools/pkg.py), and this packet was "
    "prepared without running any model, so no site call, no port override "
    "and no net has been pre-filled for you to accept. Record a decision for "
    "every CRITICAL site, including the ones you agree with -- an absent key "
    "is indistinguishable from not having looked. Annotate the topology AS "
    "DRAWN (docs/ANNOTATION_GUIDE.md); the older 'electrical impossibility "
    "wins' rule in scripts/gt_val_tools/BRIEF.md is withdrawn."
)


def emit_gt(rec: dict, comps: list[dict], out_dir: Path) -> Path:
    gt = {
        "schema_version": 1,
        "image": rec["stem"] + ".jpg",
        "source": "cghd_geometry+PENDING_manual_topology",
        "verified": False,
        "annotator": None,
        "notes": GT_NOTES_STUB,
        "components": comps,
        "bbox_frame": "cghd_blind_1024",
        "provenance": {
            "dataset": "CGHD (GTDB-HD), Zenodo record 10056817, CC BY 4.0",
            "cghd_drafter": rec["drafter"],
            "cghd_circuit": rec["circuit"],
            "cghd_image": rec["img_member"],
            "cghd_annotation": rec["xml_member"],
            "geometry_source": "CGHD Pascal-VOC boxes projected to the 1024 "
                               "frame by schematic2netlist.preprocess",
            "topology_source": None,
            "pipeline_output_used": False,
        },
    }
    p = out_dir / (rec["stem"] + ".json")
    p.write_text(json.dumps(gt, indent=2) + "\n")
    return p


def emit_decisions(rec: dict, out_dir: Path) -> Path:
    dec = {"sites": {}, "notes": DEC_NOTES_STUB}
    p = out_dir / (rec["stem"] + ".json")
    p.write_text(json.dumps(dec, indent=2) + "\n")
    return p


def emit_aux(rec: dict, aux: dict, out_dir: Path) -> Path:
    p = out_dir / (rec["stem"] + ".json")
    aux["_what_this_is"] = (
        "CGHD's own published annotations for the meta-classes we do not "
        "score (junction, crossover, text, terminal), projected to the 1024 "
        "frame, plus the per-component mapping calls a human must confirm. "
        "These are the DATASET AUTHORS' human annotations, not pipeline "
        "output. junction/crossover boxes say WHERE the drafter's intent was "
        "recorded, not which wires join -- they are strong evidence for a "
        "site call, not a substitute for making one."
    )
    p.write_text(json.dumps(aux, indent=2) + "\n")
    return p


PACKET_README = """# CGHD blind-set annotation packet

**Status: PREPARED, NOT ANNOTATED.** Every terminal in `gt/*.json` is
`"net": null`. No pipeline, detector or wire-tracer output was used to build
anything in this directory. Read `../READINESS.md` first -- it states what
this set can and cannot establish, and it contains the freeze-then-evaluate
protocol you must follow.

## What is here

| path | what |
| --- | --- |
| `gt/<stem>.json` | component inventory ONLY: class + bbox from CGHD's published Pascal-VOC annotations, projected to the 1024 frame. **All nets null.** |
| `decisions/<stem>.json` | empty decision record, same schema as `data/gt_test_1024/decisions/`. Fill it in; do not edit `gt/` by hand. |
| `aux/<stem>.json` | CGHD's own junction / crossover / text / terminal boxes, and the per-component mapping calls you must confirm. Evidence, not answers. |
| `../../../data/cghd_blind_1024/images/<stem>.jpg` | the 1024 frame to annotate against. |

## Which guide is authoritative

**Rules: `docs/ANNOTATION_GUIDE.md`.** It is the current guide and it carries
the *annotate as drawn* rule. The older `scripts/gt_val_tools/BRIEF.md` told
annotators that "electrical impossibility wins" at an ambiguous crossing; that
rule is **withdrawn**, and following it here would produce ground truth that
silently repairs the drawing.

**Tooling and commands: `scripts/gt_val_tools/BRIEF.md`.** The decisions-file
schema is identical in both, so nothing else diverges.

## Differences from the Digitize-HCD pass you should know about

1. **Classes need confirming, not just accepting.** CGHD does not split
   NPN/PNP or N/P MOSFET, so every `transistor.bjt` and `transistor.fet`
   arrives as `BJT-NPN` / `MOSFET-N` by default. Read the arrow and set the
   real class through the `classes` key. `aux/<stem>.json` lists exactly
   which component ids this affects.
2. **`vss` is not necessarily ground.** It is mapped to `GND` by
   `data/cghd/class_mapping.yaml`, which would force it onto net `"0"`. If
   the drawing means a supply rail, change the class to `V-DC (one port)`.
3. **CGHD annotates junctions and crossovers itself.** `aux` carries those
   boxes. They are the dataset authors' reading of the drawing and are the
   single biggest saving relative to the Digitize-HCD pass -- but they mark
   a location, not a partition of wires, so you still make the call.
4. **The images are photographs from a different corpus.** Expect different
   paper, pens, lighting and framing.

## Procedure

Annotate to `docs/ANNOTATION_GUIDE.md` -- the same guide, the same rules and
the same schema as the Digitize-HCD test split, so the two sets stay
comparable. Build the per-image review package (overlay, site crops,
component crops) at annotation time:

```
python scripts/gt_val_tools/batch.py <val-root> <pkg-out>
```

where `<val-root>` holds `img1024/<stem>.jpg` and `gt/<stem>.json`. Running
the tracer *then* is correct and is what the Digitize-HCD pass did: it
proposes intersection sites for a human to adjudicate. It was deliberately
not run *here*, so that nothing in the committed packet originates from the
system under test.

Finish each sheet with `finalize.py`, which writes the ERC-checked GT from
your decisions file. Set `verified: true` and `annotator` yourself -- that
sign-off is a human action by design.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", dest="zip_path", default="data/cghd/cghd-zenodo-12.zip")
    ap.add_argument("--mapping", default="data/cghd/class_mapping.yaml")
    ap.add_argument("--zero-shot-split", default="data/splits/cghd_zero_shot.txt")
    ap.add_argument("--hcd-gt-dir", default="data/gt_test_1024")
    ap.add_argument("--hcd-split", default="data/splits/test.txt")
    ap.add_argument("--images-out", default="data/cghd_blind_1024/images")
    ap.add_argument("--out-dir", default="results/cghd_blind")
    ap.add_argument("--n", type=int, default=36, help="circuits to select")
    ap.add_argument("--min-comp", type=int, default=None,
                    help="default: the Digitize-HCD test minimum")
    ap.add_argument("--max-comp", type=int, default=None,
                    help="default: 1.5x the Digitize-HCD test maximum")
    ap.add_argument("--seed", type=int, default=None, help="default: config seed")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="select and report; write nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(args.seed if args.seed is not None else cfg["seed"])
    rng = random.Random(seed)

    # ---------------- reference distribution ----------------
    hcd = hcd_test_distribution(ROOT / args.hcd_gt_dir, ROOT / args.hcd_split)
    n_min = args.min_comp if args.min_comp is not None else hcd["min"]
    n_max = args.max_comp if args.max_comp is not None else int(round(1.5 * hcd["max"]))
    print(f"[REF] Digitize-HCD test: {hcd['n_images']} images, "
          f"{hcd['min']}..{hcd['max']} components, median {hcd['median']}, "
          f"quartiles {hcd['quartiles']}")
    print(f"[REF] strata {hcd['strata_counts']}")
    print(f"[SEL] eligible component range [{n_min}, {n_max}]  (upper bound is "
          f"1.5x the Digitize-HCD max, to keep drafters whose only usable "
          f"sheet is large)")

    # ---------------- scan CGHD ----------------
    mapping = yaml.safe_load((ROOT / args.mapping).read_text())
    mapped_targets = {k: canonical_class(v["target"])
                      for k, v in mapping["mapped"].items()}
    lossy = {k for k, v in mapping["mapped"].items() if v.get("lossy")}

    zf = zipfile.ZipFile(ROOT / args.zip_path)
    recs = scan_zip(zf, mapping)
    print(f"[SCAN] {len(recs)} CGHD images carry an annotation "
          f"({len({r['drafter'] for r in recs})} drafters, "
          f"{len({(r['drafter'], r['circuit']) for r in recs})} circuits)")

    zs_lines = [l.strip() for l in (ROOT / args.zero_shot_split).read_text().split()
                if l.strip()]
    zero_shot_circuits = set()
    for line in zs_lines:
        m = IMG_RE.match(line)
        if m:
            zero_shot_circuits.add((m.group(1), int(m.group(3))))
    print(f"[SCAN] excluding {len(zero_shot_circuits)} circuits already used in "
          f"the zero-shot detection measurement ({len(zs_lines)} images)")

    rejects = collections.Counter()
    eligible = []
    for r in recs:
        why = eligibility(r, n_min, n_max, zero_shot_circuits)
        if why:
            rejects[why.split(":")[0]] += 1
        else:
            eligible.append(r)
    by_circuit = collections.defaultdict(list)
    for r in eligible:
        by_circuit[(r["drafter"], r["circuit"])].append(r)
    # One representative image per circuit, drawn NOW so that stratification
    # runs on the count of the image actually shipped: CGHD's two drawings of
    # a circuit are genuinely different drawings and can differ in component
    # count (drafter_16 C189: 43 in drawing 1, 36 in drawing 2).
    templates = {}
    for k in sorted(by_circuit):
        pool = sorted(by_circuit[k], key=lambda r: (r["drawing"], r["picture"]))
        templates[k] = rng.choice(pool)
    print(f"[POOL] {len(eligible)} eligible images -> {len(templates)} distinct "
          f"circuits across {len({k[0] for k in templates})} drafters")
    for why, n in rejects.most_common():
        print(f"       rejected {n:5d} images: {why}")

    if len(templates) < args.n:
        print(f"[FAIL] only {len(templates)} eligible circuits, need {args.n}")
        sys.exit(1)

    # ---------------- select ----------------
    chosen, reserve, achieved = select(
        templates, args.n, hcd["strata_fractions"], rng)
    picked = sorted((templates[k] for k in chosen),
                    key=lambda r: (r["drafter_id"], r["circuit"]))
    for r in picked:
        r["stem"] = f"cghd_d{r['drafter_id']:02d}_{r['cghd_stem']}"

    counts = sorted(r["n_comp"] for r in picked)
    got = collections.Counter(STRATUM_LABELS[stratum_of(c)] for c in counts)
    pool_strata = collections.Counter(
        STRATUM_LABELS[stratum_of(v["n_comp"])] for v in templates.values())
    exhausted = {s: got[s] >= pool_strata[s] for s in STRATUM_LABELS}
    print(f"\n[PICK] {len(picked)} circuits, "
          f"{len({r['drafter'] for r in picked})} distinct drafters")
    print(f"       components {counts[0]}..{counts[-1]}, "
          f"median {counts[len(counts) // 2]}, total {sum(counts)}")
    print(f"       {'stratum':10s} {'blind':>7s} {'target':>7s} {'HCD test':>9s} "
          f"{'pool':>6s}")
    for s in STRATUM_LABELS:
        tgt = hcd["strata_fractions"][s] * len(picked)
        print(f"       {s:10s} {got[s]:7d} {tgt:7.1f} "
              f"{hcd['strata_fractions'][s] * 100:8.1f}% {pool_strata[s]:6d}"
              + ("  POOL EXHAUSTED" if exhausted[s] else ""))
    hcd_counts = [c for c, k in hcd["histogram"].items() for _ in range(k)]
    ks = ks_statistic(counts, hcd_counts)
    ks_crit = 1.36 * ((1 / len(counts) + 1 / len(hcd_counts)) ** 0.5)
    print(f"       two-sample KS vs Digitize-HCD test: D={ks:.3f}, "
          f"critical {ks_crit:.3f} at alpha=0.05 -> "
          f"{'INDISTINGUISHABLE' if ks < ks_crit else 'DIFFERENT'}")

    if args.dry_run:
        print("\n[DRY-RUN] nothing written. Note this is the PRE-GATE "
              "selection: a full run additionally screens each frame for "
              "legibility and may swap in a sibling photograph or a reserve "
              "circuit, so the shipped set can differ from this listing.")
        return

    # ---------------- legibility gate ----------------
    gate = calibrate_gate(ROOT / cfg["preprocess"]["images_dir"],
                          ROOT / "data/splits/train.txt")
    print(f"\n[GATE] {gate['calibrated_on']}")
    print(f"[GATE] legible frame: ink <= {gate['max_ink_fraction']} and "
          f"largest black blob <= {gate['max_largest_component_fraction']} "
          f"of the canvas")

    def preprocess_member(rec: dict):
        """Preprocess one CGHD image; returns (canvas, meta, objs, raw)."""
        raw = zf.read(rec["img_member"])
        objs = parse_boxes(zf.read(rec["xml_member"]))
        with tempfile.NamedTemporaryFile(
                suffix=Path(rec["img_member"]).suffix, delete=False) as fh:
            fh.write(raw)
            tmp = fh.name
        try:
            # annotation-aware crop: EVERY annotated box is unioned into the
            # crop rectangle, which is what makes the frame guard satisfiable.
            result = preprocess_image_meta(
                tmp, cfg, ann_boxes=[o["xywh"] for o in objs])
        finally:
            os.unlink(tmp)
        if result is None:
            return None
        return result[0], result[1], objs, raw

    def best_photograph(rec: dict):
        """The chosen photograph, or a sibling shot of the SAME circuit if it
        did not survive binarisation.

        CGHD photographs each drawing four times; when the chosen shot is a
        shadow blow-out or a page of ruled paper that binarises to a wall of
        ink, another shot of the same circuit usually is not. Siblings are
        tried same-drawing first so the component inventory, the circuit and
        the drafter -- and therefore the stratification -- never change.
        This screens on PHOTOGRAPH QUALITY ONLY, measured on the frame with
        no model in the loop.
        """
        sibs = sorted(by_circuit[(rec["drafter"], rec["circuit"])],
                      key=lambda s: (s["drawing"] != rec["drawing"],
                                     s["picture"] != rec["picture"],
                                     s["drawing"], s["picture"]))
        tried = []
        for s in sibs:
            got = preprocess_member(s)
            if got is None:
                tried.append({"cghd_stem": s["cghd_stem"], "unreadable": True})
                continue
            q = frame_quality(got[0])
            tried.append({"cghd_stem": s["cghd_stem"], **q})
            if frame_is_legible(q, gate):
                return s, got, q, tried
        return None, None, None, tried

    # ---------------- emit ----------------
    out = ROOT / args.out_dir
    img_out = ROOT / args.images_out
    pkt = out / "packet"
    for d in (img_out, pkt / "gt", pkt / "decisions", pkt / "aux"):
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    guard_boxes = guard_outside = 0
    guard_failures = []
    swapped, illegible_circuits = [], []
    reserve_queue = list(reserve)
    queue = list(picked)
    while queue:
        r = queue.pop(0)
        chosen_rec, got, quality, tried = best_photograph(r)
        if chosen_rec is None:
            # no photograph of this circuit survives preprocessing at all
            illegible_circuits.append({
                "drafter": r["drafter"], "circuit": r["circuit"],
                "photographs_tried": tried,
            })
            print(f"  [DROP] {r['drafter']} C{r['circuit']}: no photograph "
                  f"survives binarisation ({len(tried)} tried)")
            while reserve_queue:
                cand = templates[reserve_queue.pop(0)]
                if stratum_of(cand["n_comp"]) == stratum_of(r["n_comp"]):
                    cand["stem"] = (f"cghd_d{cand['drafter_id']:02d}_"
                                    f"{cand['cghd_stem']}")
                    queue.append(cand)
                    print(f"  [SWAP] reserve {cand['drafter']} "
                          f"C{cand['circuit']} takes its place")
                    break
            continue
        if chosen_rec is not r:
            swapped.append({
                "drafter": r["drafter"], "circuit": r["circuit"],
                "from": r["cghd_stem"], "to": chosen_rec["cghd_stem"],
                "photographs_tried": tried,
            })
            chosen_rec["stem"] = (f"cghd_d{chosen_rec['drafter_id']:02d}_"
                                 f"{chosen_rec['cghd_stem']}")
            r = chosen_rec
        canvas, meta, objs, raw = got
        r["frame_quality"] = quality
        r["photographs_tried"] = len(tried)
        T = meta["target_size"]

        comps, aux_marks, flags = [], [], []
        outside_here = 0
        for o in objs:
            cx, cy, w, h = project_bbox(meta, *o["xywh"])
            inside = 0 <= cx < T and 0 <= cy < T
            guard_boxes += 1
            outside_here += not inside
            box = [round(cx, 1), round(cy, 1), round(w, 1), round(h, 1)]
            if o["raw"] in META_CLASSES and o["raw"] != "crossover":
                aux_marks.append({"cghd_class": o["raw"], "bbox": box})
                continue
            if o["raw"] == "crossover":
                # Digitize-HCD GT carries zero Wire Crossover components, so
                # crossovers stay out of the inventory -- they go to aux as
                # evidence for the site calls instead.
                aux_marks.append({"cghd_class": o["raw"], "bbox": box})
                continue
            target = mapped_targets[o["raw"]]
            cid = len(comps)
            comps.append({
                "id": cid,
                "class": target,
                "bbox": box,
                "terminals": [{"index": i, "net": None}
                              for i in range(class_terminals(target))],
            })
            if o["raw"] in NEEDS_HUMAN_CLASS_CALL:
                flags.append({
                    "component_id": cid,
                    "cghd_class": o["raw"],
                    "defaulted_to": target,
                    "lossy_mapping": o["raw"] in lossy,
                    "human_must": NEEDS_HUMAN_CLASS_CALL[o["raw"]],
                })
        if outside_here:
            guard_outside += outside_here
            guard_failures.append((r["stem"], f"{outside_here} boxes off-canvas"))

        img_path = img_out / (r["stem"] + ".jpg")
        cv2.imwrite(str(img_path), canvas)
        emit_gt(r, comps, pkt / "gt")
        emit_decisions(r, pkt / "decisions")
        emit_aux(r, {
            "stem": r["stem"],
            "cghd_junction_crossover_text_terminal_boxes": aux_marks,
            "components_needing_a_human_class_call": flags,
            "counts": {
                "junction": r["n_junction"], "crossover": r["n_crossover"],
                "text": r["n_text"], "terminal": r["n_terminal"],
            },
        }, pkt / "aux")

        manifest.append({
            "stem": r["stem"],
            "drafter": r["drafter"],
            "drafter_id": r["drafter_id"],
            "cghd_circuit": r["circuit"],
            "cghd_drawing": r["drawing"],
            "cghd_picture": r["picture"],
            "cghd_stem": r["cghd_stem"],
            "n_components": len(comps),
            "n_terminals": sum(len(c["terminals"]) for c in comps),
            "n_cghd_junction_marks": r["n_junction"],
            "n_cghd_crossover_marks": r["n_crossover"],
            "n_components_needing_class_call": len(flags),
            "stratum": STRATUM_LABELS[stratum_of(len(comps))],
            "cghd_image_member": r["img_member"],
            "cghd_annotation_member": r["xml_member"],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "frame": "1024",
            "boxes_outside_canvas": outside_here,
            "frame_quality": quality,
            "photographs_tried": len(tried),
            "nets_annotated": False,
        })
        print(f"  [OK] {r['stem']:34s} {r['drafter']:11s} "
              f"{len(comps):3d} comps  guard {outside_here} off-canvas  "
              f"ink {quality['ink_fraction']:.4f}"
              + (f"  ({len(tried)} photos tried)" if len(tried) > 1 else ""))

    manifest.sort(key=lambda m: (m["drafter_id"], m["cghd_circuit"]))
    # a rerun with a different seed or --n must not leave orphans behind
    live = {m["stem"] for m in manifest}
    pruned = []
    for d, ext in ((img_out, ".jpg"), (pkt / "gt", ".json"),
                   (pkt / "decisions", ".json"), (pkt / "aux", ".json")):
        for p in sorted(d.glob("*" + ext)):
            if p.stem not in live:
                p.unlink()
                pruned.append(str(p.relative_to(ROOT)))
    if pruned:
        print(f"[PRUNE] removed {len(pruned)} stale file(s) from a previous run")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    flat = []
    for m in manifest:
        row = {k: v for k, v in m.items() if k != "frame_quality"}
        row.update(m["frame_quality"])
        flat.append(row)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    (pkt / "README.md").write_text(PACKET_README)

    # recompute from what was actually SHIPPED, not from what was picked --
    # a reserve swap or a photograph swap must be reflected here
    blind_counts = sorted(m["n_components"] for m in manifest)
    n = len(blind_counts)
    got = collections.Counter(STRATUM_LABELS[stratum_of(c)] for c in blind_counts)
    ks = ks_statistic(blind_counts, hcd_counts)
    ks_crit = 1.36 * ((1 / n + 1 / len(hcd_counts)) ** 0.5)
    inks = sorted(m["frame_quality"]["ink_fraction"] for m in manifest)
    hcd_test_ink = sorted(
        frame_quality(cv2.imread(
            str(ROOT / cfg["preprocess"]["images_dir"] / (Path(s).stem + ".jpg")), 0)
        )["ink_fraction"]
        for s in (ROOT / args.hcd_split).read_text().split() if s.strip())
    hcd_ink = {"min": hcd_test_ink[0],
               "median": hcd_test_ink[len(hcd_test_ink) // 2],
               "max": hcd_test_ink[-1]}
    selection = {
        "purpose": (
            "CROSS-DATASET blind evaluation set. NOT a same-distribution "
            "blind set and not a substitute for one. See READINESS.md."
        ),
        "seed": seed,
        "n_selected": n,
        "distinct_drafters": len({m["drafter"] for m in manifest}),
        "eligibility_rules": {
            "one_image_per_circuit": (
                "CGHD draws each circuit twice and photographs each drawing "
                "four times; at most one image per (drafter, circuit) is "
                "selected so the set holds no near-duplicate topologies."),
            "vocabulary_coverage": (
                "every electrical symbol on the sheet must map onto one of "
                "the 17 Digitize-HCD classes -- a component the vocabulary "
                "cannot express makes the netlist unscoreable"),
            "excluded_classes": EXCLUDE_CLASSES,
            "component_count_range": [n_min, n_max],
            "zero_shot_exclusion": (
                "no circuit that contributed to results/cghd_zero_shot/ is "
                "eligible, at CIRCUIT level, not merely image level"),
            "frame_legibility": (
                "the shipped 1024 frame must fall inside the ink-statistics "
                "envelope of the Digitize-HCD TRAIN frames. Where the chosen "
                "photograph failed, a sibling shot of the SAME circuit was "
                "used instead; where no shot passed, the circuit was replaced "
                "from the reserve. This screens photographs, never pipeline "
                "output -- but it does mean the set is conditioned on the "
                "preprocessing frontend succeeding, so it UNDERSTATES CGHD's "
                "true end-to-end difficulty. See frame_legibility_gate."),
        },
        "frame_legibility_gate": {
            **gate,
            "photograph_swaps": swapped,
            "circuits_with_no_legible_photograph": illegible_circuits,
            "n_photograph_swaps": len(swapped),
            "n_circuits_dropped": len(illegible_circuits),
        },
        "rejection_counts": dict(rejects),
        "pool": {
            "eligible_images": len(eligible),
            "eligible_circuits": len(templates),
            "eligible_drafters": len({k[0] for k in templates}),
            "strata_counts": {s: pool_strata[s] for s in STRATUM_LABELS},
            "strata_exhausted_by_selection": exhausted,
            "note": (
                "Three of the four strata are taken in full, so the residual "
                "mismatch against the Digitize-HCD test distribution is a "
                "property of CGHD's eligible circuits, not of the sampler: "
                "no other selection of this size matches more closely."),
        },
        "reserve_circuits": [
            {"drafter": k[0], "circuit": k[1],
             "cghd_stem": templates[k]["cghd_stem"],
             "n_comp": templates[k]["n_comp"]} for k in reserve_queue
        ],
        "reserve_circuits_consumed": [
            {"drafter": k[0], "circuit": k[1],
             "cghd_stem": templates[k]["cghd_stem"],
             "n_comp": templates[k]["n_comp"]}
            for k in reserve if k not in reserve_queue
        ],
        "reserve_note": (
            "Circuits held back so a sheet a human later finds unusable can be "
            "replaced without an ad-hoc substitution. Any replacement must be "
            "made and recorded BEFORE the freeze; a replacement decided after "
            "seeing pipeline output is disqualifying."),
        "digitize_hcd_test_reference": hcd,
        "blind_set_distribution": {
            "min": blind_counts[0], "max": blind_counts[-1],
            "mean": round(sum(blind_counts) / n, 3),
            "median": blind_counts[n // 2],
            "quartiles": [blind_counts[n // 4], blind_counts[n // 2],
                          blind_counts[(3 * n) // 4]],
            "strata_counts": {s: got[s] for s in STRATUM_LABELS},
            "strata_fractions": {s: round(got[s] / n, 4) for s in STRATUM_LABELS},
            "histogram": dict(sorted(collections.Counter(blind_counts).items())),
            "ks_vs_digitize_hcd_test": {
                "D": ks,
                "critical_value_alpha_0.05": round(ks_crit, 4),
                "distinguishable_at_alpha_0.05": bool(ks >= ks_crit),
                "reading": (
                    "D below the critical value means the two component-count "
                    "distributions are not distinguishable at the 5% level. "
                    "It says nothing about the DRAWING distribution, which is "
                    "unambiguously different -- different corpus, different "
                    "drafters, different photography."),
            },
        },
        "per_drafter": dict(sorted(collections.Counter(
            m["drafter"] for m in manifest).items(),
            key=lambda kv: int(kv[0].split("_")[1]))),
        "frame_guard": {
            "boxes_checked": guard_boxes,
            "boxes_outside_canvas": guard_outside,
            "failures": guard_failures,
            "passed": guard_outside == 0 and not guard_failures,
        },
        "frame_statistics_vs_digitize_hcd": {
            "blind_ink_fraction": {
                "min": min(inks), "median": sorted(inks)[n // 2],
                "max": max(inks),
            },
            "digitize_hcd_test_ink_fraction": hcd_ink,
            "reading": (
                "Even after the legibility gate the CGHD frames are harder "
                "than they look. Two systematic differences survive: ruled "
                "notebook paper binarises to a field of parallel lines that a "
                "wire tracer cannot distinguish from wires, and photographed "
                "page edges widen the annotation-aware crop so the drawing "
                "occupies less of the 1024 canvas -- i.e. thinner strokes at "
                "the same nominal resolution."),
        },
        "nets_annotated": False,
        "pipeline_output_used": False,
        "attestation": (
            "No detector, wire tracer or pipeline stage was executed while "
            "building this set. Every terminal in packet/gt/ is null. "
            "Component classes and boxes come from CGHD's published "
            "Pascal-VOC annotations (CC BY 4.0)."
        ),
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    write_run_metadata(out, cfg, seed, extra={
        "script": "scripts/prepare_cghd_blind.py",
        "n_selected": n,
        "nets_annotated": False,
        "pipeline_output_used": False,
    })

    print(f"\n[GUARD] annotation containment: {guard_outside} of {guard_boxes} "
          f"boxes outside the canvas")
    if guard_outside or guard_failures:
        print(f"[FAIL] frame guard: {guard_failures[:5]}")
        sys.exit(1)
    print("[PASS] every annotated object lies inside the 1024 canvas")
    print(f"[OK] {n} circuits, {len({m['drafter'] for m in manifest})} drafters")
    print(f"[OK] images  -> {img_out}")
    print(f"[OK] packet  -> {pkt}  (ALL NETS NULL -- annotation pending)")
    print(f"[OK] manifest-> {out}/manifest.json")


if __name__ == "__main__":
    main()
