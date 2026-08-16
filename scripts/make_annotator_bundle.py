#!/usr/bin/env python3
"""Assemble the self-contained bundle handed to the independent second annotator.

The blind packet is proven not to leak: every image is byte-identical to an
untouched photograph and hash-disjoint from every render in the repository. The
GUIDE the packet tells that annotator to read first was never checked the same
way, and it leaks badly. It names ten of the fifty-eight packet circuits and
says what is wrong with them -- which had the wrong pin order, where the first
pass overrode the ink, which op-amps were backwards. The packet withholds
stratum labels precisely so a reader cannot tell which drawings are hard, and
then the guide names ten of them.

So this does three things:

  1. Copies the packet -- images to zoom, 1024 frames for coordinates.
  2. REDACTS the guide against the packet: every circuit id that appears in
     both is replaced, and the worked-example table's source column is dropped.
  3. Asserts the result. No shipped text may name a packet circuit. The check
     runs over every text file in the bundle and exits non-zero rather than
     writing a doubtful bundle, exactly as _assert_blind does for the images.

The annotation tool ships with it. It is Python standard library only -- no
install, no virtualenv, no network -- because the person doing this is doing you
a favour and every dependency is a chance for them to stop.

RESIDUAL RISK, STATED RATHER THAN HIDDEN. Two worked examples are crops from
packet circuits. The crops are a few hundred pixels around one junction and
carry no netlist, and removing them would cost the annotator the calibration
that makes the pass worth having. Their source ids are redacted, but an
annotator who recognises a crop while working that circuit gains one site. The
bundle says so, and names which two, so the choice is yours rather than mine.

Usage:
    python scripts/make_annotator_bundle.py
    python scripts/make_annotator_bundle.py --out dist/second-annotator
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKET = ROOT / "results/blind_review/packet"

REDACTION = "[circuit id withheld -- it is in your packet]"

START_HERE = """\
# Start here

Thank you for doing this. It is the one measurement in this project that cannot
be produced by its author, and there is no substitute for it.

## What you are being asked to do, in one paragraph

You have {n} photographs of hand-drawn circuit schematics. For each one, work
out which electrical **net** each component terminal sits on -- that is, which
terminals are joined by wire -- and record it. An annotation of these same
drawings already exists. You have not seen it and must not; the entire value of
your pass is that it is independent. Where the two disagree, the disagreement is
the result.

**You are not asked whether the circuit works, what any component's value is, or
whether it would simulate.** Only what is connected to what.

## Setup: one command, no installation

You need Python 3 (macOS and Linux have it; on Windows install it from
python.org). Nothing else -- no packages, no internet.

From this folder:

```
python3 tools/annotator/server.py --blind
```

Then open **http://127.0.0.1:8765** in a browser. That is the whole setup. The
tool runs entirely on your machine; nothing is uploaded anywhere.

## How to use the tool

Per circuit:

1. Press **b** and drag a box around a component symbol. Pick its class from the
   dropdown on the right.
2. Press **t** and click each of that component's terminals **in port order**
   (see the guide, section 7 -- for a transistor this is collector, base,
   emitter). Before each click, type the net name in the *Current net* box, or
   press **n** for a fresh net. Ground is always net **0** -- press **0**.
3. Press **i** and click every place two wires cross or meet, then press
   **j** (they join), **k** (they cross over, no connection), **e** (edge group)
   or **o** (nothing happens here).
4. Write anything you were unsure about in **Notes**, with coordinates.
5. Press **Enter** to submit and move on.

It autosaves every ten seconds and reopens where you left off, so you can stop
any time. Use **[** and **]** to move between circuits.

Budget 3-5 minutes for a simple drawing, 8-12 for a dense one with transistors.
Short sessions beat long ones.

## Read this before you start

**`ANNOTATION_GUIDE.md`** -- the conventions. It is long, but sections 0-4 and 7
are the ones that decide whether your pass is comparable, and it has eight
worked examples with pictures. Section 0 matters most: **record what is visibly
drawn, not a corrected circuit.** If the ink says two wires meet, they meet,
even if that shorts something or makes the circuit impossible to simulate. Note
what you think was meant; do not move the wire.

**`PACKET_README.md`** -- the ground rules, including what not to look at.

## Two directories of the same drawings

| folder | what it is | use it for |
| --- | --- | --- |
| `images/` | the original photographs, full resolution | **looking** -- zoom in here to read faint pencil |
| `frames_1024/` | the same drawings, normalised to 1024 px | **coordinates** -- any `[x, y]` you write must be in this frame |

The tool already shows you the 1024 frame, so coordinates are handled for you.
Open `images/` separately if you need a closer look at faint ink.

## When you are done

Your work is in `data/blind_review/incoming/` as one JSON file per circuit. Zip
that folder and send it back. That is everything.

If you stop partway, send what you have -- a partial pass is still a
measurement, and the sampling is built so that any prefix is usable.

## Questions

Ask about **conventions** freely -- what counts as a junction, how to order a
MOSFET's pins, what to do with a component drawn half off the page. Do not ask
about a **specific circuit** until your pass is delivered, and do not discuss
one with anyone who has seen the first pass. That is the only rule that, if
broken, cannot be repaired afterwards.
"""


def packet_stems() -> list[str]:
    f = PACKET / "circuits.txt"
    if not f.exists():
        sys.exit("no packet: run scripts/make_blind_packet.py first")
    return [s.strip() for s in f.read_text().split() if s.strip()]


def redact(text: str, stems: set[str]) -> tuple[str, int]:
    """Replace every packet circuit id, and drop the worked-example source column."""
    n = 0

    def sub(m):
        nonlocal n
        if m.group(0) in stems:
            n += 1
            return REDACTION
        return m.group(0)

    text = re.sub(r"circuit_\d+", sub, text)

    # The worked-example table has a column naming the circuit each crop came
    # from. Even redacted that column is now noise; drop the cell contents.
    def strip_source(line: str) -> str:
        if line.startswith("|") and REDACTION in line:
            parts = line.split("|")
            return "|".join(
                ("  --  " if REDACTION in p else p) for p in parts)
        return line

    return "\n".join(strip_source(x) for x in text.splitlines()), n


def assert_no_leak(bundle: Path, stems: set[str]) -> None:
    """No shipped text may name a packet circuit. Exits non-zero on any hit."""
    problems = []
    for p in sorted(bundle.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (
                ".md", ".txt", ".json", ".py", ".js", ".html", ".yaml", ".yml"):
            continue
        # circuits.txt is the work list; naming the packet is its whole job
        if p.name == "circuits.txt":
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        hits = sorted(set(re.findall(r"circuit_\d+", text)) & stems)
        if hits:
            problems.append(f"{p.relative_to(bundle)}: names {hits[:6]}")
    if problems:
        print("\n!!! BUNDLE LEAKS PACKET CIRCUIT IDS -- not usable !!!",
              file=sys.stderr)
        for x in problems:
            print("  " + x, file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist/second-annotator")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    stems = packet_stems()
    stem_set = set(stems)
    bundle = ROOT / a.out
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    # 1. the packet, FLAT at the bundle root.
    #
    # These were symlinks into results/blind_review/packet/ until it turned out
    # that zip stores a symlinked directory as an EMPTY one: the annotator would
    # have opened images/, found nothing, and been stuck before starting. Real
    # directories, at the top level, where a person looks first. The tool
    # resolves both layouts (see server._first_dir) so nothing is duplicated.
    for sub in ("images", "frames_1024"):
        src = PACKET / sub
        if not src.is_dir() or not any(src.iterdir()):
            sys.exit(f"packet/{sub} is empty -- run scripts/make_blind_packet.py")
        shutil.copytree(src, bundle / sub)
    shutil.copyfile(PACKET / "circuits.txt", bundle / "circuits.txt")
    shutil.copyfile(PACKET / "README.md", bundle / "PACKET_README.md")

    # 2. the redacted guide
    guide, n_red = redact((ROOT / "docs/ANNOTATION_GUIDE.md").read_text(), stem_set)
    guide = ("<!-- REDACTED FOR THE SECOND ANNOTATOR. Circuit ids belonging to\n"
             "     your packet have been removed from this guide so it cannot\n"
             "     tell you which drawings the first pass found difficult. -->\n\n"
             + guide)
    (bundle / "ANNOTATION_GUIDE.md").write_text(guide)
    shutil.copytree(ROOT / "docs/annotation_examples",
                    bundle / "annotation_examples")

    # 3. the tool
    shutil.copytree(ROOT / "tools/annotator", bundle / "tools/annotator",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # 4. instructions
    (bundle / "START_HERE.md").write_text(START_HERE.format(n=len(stems)))

    # 5. prove it
    assert_no_leak(bundle, stem_set)

    n_img = len(list((bundle / "images").glob("*")))
    n_frm = len(list((bundle / "frames_1024").glob("*")))
    size = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
    print(f"bundle -> {a.out}")
    print(f"  {len(stems)} circuits, {n_img} photographs + {n_frm} frames")
    print(f"  guide redacted: {n_red} packet circuit id(s) removed")
    print(f"  leak assertion: OK -- no shipped text names a packet circuit")
    print(f"  size: {size / 1e6:.1f} MB")

    if not a.no_zip:
        archive = shutil.make_archive(str(bundle), "zip", root_dir=bundle)
        print(f"  archive: {Path(archive).relative_to(ROOT)} "
              f"({Path(archive).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
