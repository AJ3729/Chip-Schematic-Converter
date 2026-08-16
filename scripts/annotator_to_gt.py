#!/usr/bin/env python3
"""Annotation-tool records -> the two files ``compare_annotations.py`` scores.

``tools/annotator`` writes one file per circuit holding everything the annotator
placed: components with boxes and terminals, intersection sites, interventions,
notes, and the seconds spent. ``scripts/compare_annotations.py`` reads a ground
truth directory in the schema of ``src/schematic2netlist/gt.py`` plus a
``decisions/`` record beside it. This converts the first into the second.

It is a FORMAT conversion and nothing else. No net is inferred, no box is
guessed, no site call is defaulted. Where the tool's record is incomplete this
script says so and refuses to write that circuit, because the alternative --
emitting a plausible file with an invented net in it -- produces an
inter-annotator disagreement that belongs to this script rather than to either
annotator, and nothing downstream could tell the difference.

TWO THINGS THAT ARE NOT ONE-TO-ONE, AND HOW EACH IS HANDLED

  Sites. The tool records each intersection where the annotator saw it, as a
  coordinate. The decisions schema keys site calls by the tracer's site index,
  which is derived from whoever drew the component boxes and so cannot be shared
  between two independent passes. The coordinates are therefore written to the
  ``sites_xy`` key, which compare_annotations resolves against its own tracer
  within a stated tolerance and reports as unresolved when it cannot. See
  ``resolve_sites_xy`` there for why this is the only sound direction.

  Interventions. The tool keeps repairs the annotator WOULD apply separate from
  the topology, and that separation is the point of recording them. They are
  carried into the decisions record under ``interventions``, never folded into
  any net.

Usage:
    python scripts/annotator_to_gt.py                       # blind packet
    python scripts/annotator_to_gt.py --in <dir> --out <dir>
    python scripts/annotator_to_gt.py --strict              # refuse partial work
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.classes import class_terminals  # noqa: E402
from schematic2netlist.gt import SCHEMA_VERSION, validate_gt  # noqa: E402

SITE_KIND_TO_CALL = {"junction": "junction", "crossing": "crossing",
                     "edge_group": "edge-group", "none": "none"}


def convert(rec: dict, stem: str) -> tuple[dict, dict, list[str]]:
    """One tool record -> (gt, decisions, problems). Problems block the write."""
    problems: list[str] = []
    components = []

    for c in rec.get("components", []):
        cid = c.get("id")
        cls = c.get("class")
        terms = c.get("terminals", [])

        if not c.get("bbox"):
            problems.append(
                f"component {cid} ({cls}) has no bounding box; components are "
                "paired between annotations by box overlap, so this one could "
                "not be matched to anything")
        expected = class_terminals(cls) if cls else None
        if expected is not None and len(terms) != expected:
            problems.append(
                f"component {cid} ({cls}) has {len(terms)} terminal(s), "
                f"expected {expected}")
        for t in terms:
            if not t.get("net"):
                problems.append(
                    f"component {cid} ({cls}) terminal {t.get('index')} has no "
                    "net; mark the component unconnected if the lead really "
                    "goes nowhere")

        components.append({
            "id": cid,
            "class": cls,
            "bbox": c["bbox"] if c.get("bbox") else None,
            "terminals": [{"index": t["index"], "net": t.get("net")}
                          for t in sorted(terms, key=lambda t: t["index"])],
            **({"unconnected": True} if c.get("unconnected") else {}),
        })

    gt = {
        "schema_version": SCHEMA_VERSION,
        "image": rec.get("image", f"{stem}.jpg"),
        "source": rec.get("source", "manual"),
        "verified": False,       # a second pass is evidence, not a new authority
        "annotator": rec.get("annotator", "second"),
        "notes": rec.get("notes", ""),
        "components": components,
        "bbox_frame": 1024,
    }

    sites_xy = []
    for s in rec.get("sites", []):
        kind = s.get("kind")
        call = SITE_KIND_TO_CALL.get(kind)
        if call is None:
            problems.append(f"site {s.get('id')} has unknown kind {kind!r}")
            continue
        xy = s.get("xy")
        if not (isinstance(xy, list) and len(xy) == 2):
            problems.append(f"site {s.get('id')} has no usable coordinate")
            continue
        sites_xy.append({"xy": xy, "call": call})

    decisions = {
        "sites_xy": sites_xy,
        "notes": rec.get("notes", ""),
        "interventions": rec.get("interventions", []),
        "annotation_seconds": rec.get("annotation_seconds"),
        "pass": rec.get("pass", 1),
        "_frame": "coordinates are in the 1024 px frame",
    }
    return gt, decisions, problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="data/blind_review/incoming",
                    help="directory of annotation-tool records")
    ap.add_argument("--out", dest="dst", default="data/blind_review/gt_b",
                    help="GT directory to write (with decisions/ beside it)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any circuit had a problem")
    a = ap.parse_args()

    src, dst = ROOT / a.src, ROOT / a.dst
    if not src.is_dir():
        sys.exit(f"no such directory: {a.src}")
    records = sorted(src.glob("*.json"))
    if not records:
        sys.exit(f"{a.src} holds no annotation records")

    (dst / "decisions").mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for p in records:
        stem = p.stem
        rec = json.loads(p.read_text())
        gt, dec, problems = convert(rec, stem)
        problems += [f"schema: {m}" for m in validate_gt(gt, strict=False)]
        if problems:
            skipped.append((stem, problems))
            continue
        (dst / f"{stem}.json").write_text(json.dumps(gt, indent=1) + "\n")
        (dst / "decisions" / f"{stem}.json").write_text(json.dumps(dec, indent=1) + "\n")
        written.append(stem)

    print(f"converted {len(written)}/{len(records)} circuits -> {a.dst}")
    n_sites = sum(len(json.loads((dst / 'decisions' / f'{s}.json').read_text())
                      ["sites_xy"]) for s in written)
    print(f"  {n_sites} intersection calls carried as coordinates")
    if skipped:
        print(f"\n{len(skipped)} circuit(s) NOT written -- fix in the tool and "
              f"re-submit:")
        for stem, problems in skipped:
            print(f"  {stem}:")
            for m in problems[:6]:
                print(f"    - {m}")
            if len(problems) > 6:
                print(f"    ... and {len(problems) - 6} more")
    if written:
        print(f"\nscore with:\n  PYTHONPATH=src python scripts/compare_annotations.py "
              f"--gt-b {a.dst}")
    return 1 if (skipped and a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
