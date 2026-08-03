#!/usr/bin/env python3
"""Disambiguate component class from the UNIT of its value label.

38 of the 67 wrong class labels survive both the three-seed vote and
polarity-preserving TTA, so they are systematic rather than variance: the
detector genuinely believes the wrong class, and no amount of resampling the
same evidence will move it. A different signal is required, and Digitize-HCD
already ships one -- ``text_annotations.json`` gives 11,936 transcribed value
labels, 9.3 per image, with polygons in original-image coordinates.

The unit is the class. Almost perfectly:

    ohm, kohm, Mohm            -> Resistor          26.0% of labels
    H, uH, mH                  -> Inductor          17.7%
    F, uF, mF, nF, pF          -> Capacitor         17.1%
    V, mV, kV                  -> voltage source     15.7%
    A, mA                      -> current source     8.7%
    contains sin or cos        -> the source is AC    248 instances
    Hz, kHz                    -> an AC source is nearby, but this label is a
                                  frequency annotation, not a component value

That lands exactly on the confusions the ensembles cannot fix: Resistor against
Inductor, Resistor read as Capacitor, and -- via sin/cos/Hz -- I-DC read as
I-AC, which was the single largest correctable group in the vote audit.

WHAT THIS IS AND IS NOT. Reading the transcription is oracle information, so
this script prices a ceiling; it is not a pipeline capability on its own. But
the decomposition it establishes is the useful part: everything except reading
the glyphs -- projecting the label, associating it with a component, mapping
unit to class -- is deterministic. And the reading needed is NOT transcription,
only which of ~8 unit families the crop belongs to, an 8-way classification with
~10k training crops available outside the test split. That is a far easier
problem than handwritten OCR, and it is what this result would justify building.

TWO SAFETY RULES, because a mis-associated label is worse than no label:

  family-local only   a relabel is allowed only WITHIN {Resistor, Capacitor,
                      Inductor} or WITHIN {V-DC, V-AC, I-DC, I-AC}. Those are
                      where the confusions live, and confining changes to them
                      makes a bad association harmless -- a resistor value that
                      lands on a MOSFET changes nothing.
  global assignment   each component carries at most one value label and each
                      label belongs to at most one component, so the pairing is
                      an assignment problem solved with Hungarian under a
                      distance cap. Greedy mutual-nearest was tried first and
                      swapped the labels of 16 adjacent L/C pairs -- 62% of all
                      the damage -- because in a tank circuit each side's
                      nearest label is the other's too.

Usage:
    python scripts/inject_text_classes.py --out data/detections_1024_text
    python scripts/inject_text_classes.py --out data/detections_1024_vote_text \\
        --primary data/detections_1024_vote        # compose with the seed vote
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.preprocess import project_bbox

TEXT_JSON = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
             "Component Symbol and Text Label Data/text_annotations.json")

PASSIVE = {"Resistor", "Capacitor", "Inductor"}
SOURCES = {"V-DC", "V-DC (one port)", "V-AC", "I-DC", "I-AC"}


def unit_kind(text: str) -> tuple[str | None, bool]:
    """(unit kind in R/C/L/V/I, label_is_ac) from a value string.

    Returns (None, ...) when the label names no component: a bare number, or a
    frequency annotation, which belongs to a source without identifying it.

    The kind deliberately stops at R/C/L/V/I and does NOT decide DC against AC.
    An AC source is labelled with its amplitude -- "3V" -- and its frequency
    lives in a SEPARATE "2kHz" label, so the absence of sin/cos in the one label
    that happens to associate says nothing about the waveform. Inferring DC from
    that absence produced 16 wrong V-AC -> V-DC relabels on the first attempt.
    ``label_is_ac`` is therefore only ever True on positive evidence.
    """
    s = unicodedata.normalize("NFKC", text).strip()
    low = s.lower()
    ac = ("sin" in low) or ("cos" in low)

    if re.search(r"hz$", low):
        return None, True            # frequency: AC nearby, but not this label
    # order matters: check the two-letter units before the single letters
    if re.search(r"(ω|ohm)$", low):
        return "R", ac
    if re.search(r"[kmμµnp]?f$", low):
        return "C", ac
    if re.search(r"[kmμµnp]?h$", low):
        return "L", ac
    if re.search(r"[kmμµ]?v$", low):
        return "V", ac
    if re.search(r"[kmμµ]?a$", low):
        return "I", ac
    return None, ac


PASSIVE_OF = {"R": "Resistor", "C": "Capacitor", "L": "Inductor"}


def relabel(cur: str, kind: str, label_ac: bool) -> str | None:
    """The class the unit implies, or None to leave the detection alone.

    Changes stay inside a family, and every attribute the text cannot see is
    CARRIED OVER from the detector rather than reset: whether a voltage source
    is one-port, and whether a source is DC or AC. The detector is right about
    those far more often than a single associated label is, and resetting them
    was what made the first version destructive.
    """
    if cur in PASSIVE and kind in PASSIVE_OF:
        want = PASSIVE_OF[kind]
        return want if want != cur else None

    if cur in SOURCES and kind in ("V", "I"):
        is_ac = cur.endswith("-AC") or label_ac
        if kind == "V":
            want = "V-AC" if is_ac else (
                "V-DC (one port)" if "(one port)" in cur else "V-DC")
        else:
            want = "I-AC" if is_ac else "I-DC"
        return want if want != cur else None

    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val",
                    help="exploration/oracle-injection, so it reads val by "
                         "default; --split test only for a reported number")
    ap.add_argument("--primary", default="data/detections_1024")
    ap.add_argument("--transforms", default="data/transforms_1024.json")
    ap.add_argument("--max-dist-frac", type=float, default=1.2,
                    help="cap on text-to-component distance, in units of the "
                         "component box diagonal")
    ap.add_argument("--ac-from-hz", action="store_true",
                    help="also let a nearby Hz label push a source to AC")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    conf = cfg["detect"].get("confidence")
    tf = json.loads(Path(args.transforms).read_text())
    ann = {e["file_name"]: e["instances"]
           for e in json.loads(Path(TEXT_JSON).read_text())["data_list"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(f"data/splits/{args.split}.txt")
             if l.strip()]

    changed = Counter()
    skipped = Counter()
    n_files = n_det = n_chg = n_pair = 0
    for nm in names:
        stem = Path(nm).stem
        pp = Path(args.primary) / f"{stem}.json"
        meta = tf.get(stem)
        insts = ann.get(nm)
        if not (pp.exists() and meta and insts):
            skipped["no annotation or transform"] += 1
            continue
        cache = json.loads(pp.read_text())
        dets = [d for d in cache["detections"]
                if conf is None or d.get("confidence", 1.0) >= conf]

        # project every label into cleaned-image coordinates
        labels = []
        for i in insts:
            x1, y1, x2, y2 = i["bbox"]
            cx, cy, _w, _h = project_bbox(meta, x1, y1, x2 - x1, y2 - y1)
            kind, ac = unit_kind(i["text"])
            labels.append({"x": cx, "y": cy, "kind": kind, "ac": ac,
                           "text": i["text"]})

        # GLOBAL assignment, not greedy. Each component carries at most one
        # value label and each label belongs to at most one component, which is
        # an assignment problem -- and greedy mutual-nearest gets it wrong
        # exactly where it matters. An L and a C sitting side by side in a tank
        # circuit have mutually close labels, and picking each side's nearest
        # independently swapped 16 of them, which was 62% of all the damage the
        # first version did. This is the same correction the ports bug needed.
        pairs: list[tuple[int, int]] = []
        if dets and labels:
            BIG = 1e6
            cost = np.full((len(dets), len(labels)), BIG, dtype=float)
            for di, d in enumerate(dets):
                cap = args.max_dist_frac * (d["width"] ** 2
                                            + d["height"] ** 2) ** 0.5
                for li, lb in enumerate(labels):
                    dist = float(np.hypot(d["x"] - lb["x"], d["y"] - lb["y"]))
                    if dist <= cap:
                        cost[di, li] = dist
            rows, cols = linear_sum_assignment(cost)
            for di, li in zip(rows, cols):
                if cost[di, li] < BIG:
                    pairs.append((int(di), int(li)))
                else:
                    skipped["no label within the distance cap"] += 1

        for di, li in pairs:
            n_pair += 1
            d = dets[di]
            cur = canonical_class(d["class"])
            lab = labels[li]
            kind = lab["kind"]
            if kind is None:
                if args.ac_from_hz and lab["ac"] and cur in SOURCES:
                    kind = "V" if cur.startswith("V") else "I"
                else:
                    skipped["label names no component"] += 1
                    continue
            want = relabel(cur, kind, lab["ac"])
            if want is None:
                skipped["unit agrees or is cross-family"] += 1
                continue
            changed[f"{cur} -> {want}  [{lab['text']}]"] += 1
            d["class"] = want
            n_chg += 1
        n_det += len(dets)
        (out / f"{stem}.json").write_text(json.dumps(cache) + "\n")
        n_files += 1

    print(f"wrote {n_files} caches to {out}")
    print(f"detections {n_det}, text-component pairs {n_pair}, "
          f"relabelled {n_chg} ({n_chg/max(n_det,1):.2%})")
    print(f"\nrefusals:")
    for k, v in skipped.most_common():
        print(f"  {k:28s} {v:5d}")
    print(f"\nlabel changes:")
    for k, v in changed.most_common(25):
        print(f"  {k:46s} {v:4d}")
    print(f"\nPrice this with scripts/audit_relabels.py before benchmarking.")


if __name__ == "__main__":
    main()
