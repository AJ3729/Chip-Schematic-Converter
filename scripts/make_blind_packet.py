#!/usr/bin/env python3
"""Build the blind packet for an INDEPENDENT SECOND ANNOTATION of the test split.

The shipped connectivity ground truth (``data/gt_test_1024/``) was produced by
one annotator and checked by an AI assistant re-deriving the same drawings. That
establishes self-consistency, not correctness: the re-derivation shares the first
pass's reasoning and its blind spots (``data/README.md`` says so explicitly). The
only thing that measures correctness is a second reader who has never seen the
first pass. This script builds what that reader is given.

WHAT "BLIND" MEANS HERE, OPERATIONALLY
--------------------------------------
The packet contains RAW PHOTOGRAPHS AND NOTHING ELSE. No netlist, no net label,
no component box, no junction/crossing call, no pipeline prediction, no prose
notes, and no stratum label. Anything in that list, seen even once, turns the
second annotation into a review of the first one and destroys the measurement it
exists to produce.

The failure mode is silent: this repo's rendering tooling (``gt_val_tools/render``,
``scripts/render_gt_overlay.py``, ``scripts/annotate_topology.py --render``)
defaults to DRAWING BOXES, and a packet built from ``<gt>/renders/`` would look
perfectly plausible in a file listing -- same stems, same count, same extension --
while leaking the entire answer key. So the packet is not merely built from the
right directory; every copied file is PROVEN byte-identical to an untouched
source photograph (sha256), and proven disjoint from every render directory in
the repo. See ``_assert_blind`` -- it exits non-zero rather than shipping a
doubtful packet.

A pixel-statistics check was evaluated as a second line of defence and REJECTED
as unsound: the fraction of strongly-saturated pixels separates renders (min
0.0019 over the 192 test renders) from raw photographs (median 1.3e-06) for most
images, but three test photographs shot with coloured pen or coloured paper reach
0.013-0.016 and sit ABOVE the render minimum. A check with that false-positive
rate would be routinely overridden, which is worse than not having it. Byte
identity is exact, cheap and unfoolable, so it is the assertion that ships.

SAMPLING
--------
50-60 circuits in three strata, seeded and recorded. Drawn in this ORDER, which
matters:

  1. ``uniform``    (~20) drawn from ALL 192 test images. Drawn FIRST and from
                    the whole split, because this is the only stratum that can
                    carry an unbiased inter-annotator agreement estimate. Had it
                    been drawn from what the enriched strata left behind it would
                    be a sample of "circuits that are neither multi-terminal nor
                    hard", which estimates nothing anyone wants.
  2. ``multi_terminal`` (~20) circuits holding a 3+-terminal device (BJT, MOSFET,
                    op-amp), drawn from that pool minus stratum 1. The first pass
                    reported pin order on 3-terminal parts as its dominant error
                    mode, and NO automatic check catches a pin swap: net grouping
                    is unchanged, the ERC still passes, and the benchmark's own
                    scorer canonicalises terminal order away before comparing.
                    Enrichment here is the only way to measure it.
  3. ``hard_core``  (~18) circuits where the pipeline AND both frontier VLMs
                    disagree with the ground truth. Three independent systems
                    failing on one drawing is weak evidence of three independent
                    failures and stronger evidence that the GT itself is wrong,
                    which makes these the highest-yield circuits to re-annotate.

THE HARD-CORE STRATUM AND THE 2026-08-03 ROLE SWAP -- READ ``data/README.md``
----------------------------------------------------------------------------
The two evaluation splits exchanged names on 2026-08-03: the 190 images once
called ``test`` are now ``val``. The VLM baselines under ``results/vlm/*_b/scored/``
were scored BEFORE that swap, so despite the directory names they are scored on
what is now the VALIDATION split and share ZERO images with the current test
split. This script does not assume it either way: it intersects by image stem,
prints the overlap, and if the overlap is zero it says so loudly and falls back
to a pipeline-only definition of the stratum -- a real but WEAKER signal, since
one system failing is ordinary. The fallback is recorded in the manifest and in
``sampling_meta.json`` so no reader can mistake which definition was used.

WHERE THE STRATUM LABELS LIVE -- DELIBERATELY NOT IN THE PACKET
---------------------------------------------------------------
``manifest.csv`` is written one level ABOVE the packet, at
``results/blind_review/manifest.csv``. It is for us, not for the annotator.
Telling a reader that ``circuit_513`` is in the stratum where three systems
disagree with the ground truth is a hint about where to look and what to expect,
and it would bias exactly the circuits the stratum exists to test. The packet
itself gets ``circuits.txt``, which is the same stems in shuffled order with no
stratum column, so the ordering leaks nothing either.

Usage:
    python scripts/make_blind_packet.py
    python scripts/make_blind_packet.py --seed 7 --n-uniform 20 \
        --n-multi-terminal 20 --n-hard-core 18
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from schematic2netlist.classes import canonical_class, class_terminals

ROOT = Path(__file__).resolve().parent.parent

# Directories a packet image is allowed to come from. Anything else -- above all
# any ``renders/`` or overlay directory -- is a hard failure, not a warning.
ALLOWED_SOURCE_DIRS = ("data/raw", "data/cleaned_1024")

# Every directory in the repo known to hold ANNOTATED renders. The packet is
# proven hash-disjoint from all of them.
RENDER_DIR_GLOBS = (
    "data/*/renders",
    "data/*/*/renders",
    "results/gt_overlay",
    "results/paper/qualitative",
    "results/blind_review/../*/renders",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def read_stems(path: Path) -> list[str]:
    """Split manifests hold ``circuit_123.jpg``; everything else keys by stem."""
    return [Path(line.strip()).stem for line in path.read_text().splitlines()
            if line.strip()]


def multi_terminal_pool(gt_dir: Path, stems: list[str]) -> tuple[list[str], dict]:
    """Stems whose GT holds at least one 3+-terminal device.

    Terminal count comes from ``classes.class_terminals`` (the class vocabulary),
    not from ``len(component["terminals"])``, so a GT file that recorded the wrong
    number of terminals for a transistor still lands in the pool -- that file is
    precisely one a second reader should see.
    """
    pool, per_stem = [], {}
    for stem in stems:
        path = gt_dir / f"{stem}.json"
        if not path.is_file():
            continue
        gt = json.loads(path.read_text())
        n = sum(1 for c in gt["components"]
                if class_terminals(canonical_class(c["class"])) >= 3)
        per_stem[stem] = n
        if n:
            pool.append(stem)
    stats = {
        "pool_size": len(pool),
        "devices_total": sum(per_stem.values()),
        "class_histogram": None,
    }
    return pool, {**stats, "per_stem": per_stem}


def failures_from_csv(path: Path, stems_in_split: set[str]) -> tuple[set[str], dict]:
    """Stems with ``strict_success == False``, plus how the CSV maps onto the split.

    The overlap report is the point of the return value: a scored CSV whose stems
    are not in the split being sampled is measuring a different set of images, and
    silently intersecting to the empty set would quietly produce a stratum that is
    not the stratum that was asked for.
    """
    if not path.is_file():
        return set(), {"path": str(path), "present": False, "rows": 0,
                       "stems": 0, "in_split": 0, "failures_in_split": 0}
    rows = list(csv.DictReader(path.open()))
    stems = {Path(r["image"]).stem for r in rows}
    fails = {Path(r["image"]).stem for r in rows
             if str(r.get("strict_success", "")).strip().lower() == "false"}
    return fails & stems_in_split, {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "present": True,
        "rows": len(rows),
        "stems": len(stems),
        "in_split": len(stems & stems_in_split),
        "failures_total": len(fails),
        "failures_in_split": len(fails & stems_in_split),
    }


def build_hard_core(pipeline_csv: Path, vlm_csvs: list[Path],
                    stems_in_split: set[str]) -> tuple[list[str], dict]:
    """The three-way-disagreement pool, with an explicit, recorded fallback.

    Intended definition: pipeline strict-failure AND both VLMs strict-failure on
    the same image. If the VLM CSVs do not overlap the split at all (they do not,
    as of the 2026-08-03 role swap -- see the module docstring), the intersection
    is vacuously empty and the stratum degrades to pipeline-only failures. That is
    a weaker signal and the caller is told so in the loudest terms the terminal
    allows, plus permanently in ``sampling_meta.json``.
    """
    pipe_fail, pipe_report = failures_from_csv(pipeline_csv, stems_in_split)
    vlm_reports, vlm_fail_sets = [], []
    for p in vlm_csvs:
        fails, rep = failures_from_csv(p, stems_in_split)
        vlm_reports.append(rep)
        vlm_fail_sets.append(fails)

    usable = [rep["in_split"] > 0 for rep in vlm_reports]
    if vlm_fail_sets and all(usable):
        pool = set(pipe_fail)
        for s in vlm_fail_sets:
            pool &= s
        definition = "three_way_disagreement"
        fallback = False
    else:
        pool = set(pipe_fail)
        definition = "pipeline_only_failures_FALLBACK"
        fallback = True

    report = {
        "definition": definition,
        "fallback_used": fallback,
        "pool_size": len(pool),
        "pipeline": pipe_report,
        "vlm": vlm_reports,
        "vlm_overlap_with_split": {r["path"]: r["in_split"] for r in vlm_reports},
    }
    return sorted(pool), report


def resolve_source(stem: str, raw_dir: Path, fallback_dir: Path) -> tuple[Path, str]:
    for d, label in ((raw_dir, "raw"), (fallback_dir, "fallback")):
        for ext in (".jpg", ".jpeg", ".png"):
            p = d / f"{stem}{ext}"
            if p.is_file():
                return p, label
    raise SystemExit(f"no source photograph for {stem} in {raw_dir} or {fallback_dir}")


def _render_hashes() -> tuple[dict[str, str], int]:
    """sha256 -> path for every annotated render in the repo.

    Hashing a few thousand PNGs costs a second or two and buys the only statement
    that actually matters to a reviewer: no file in the packet is a render.
    """
    seen: dict[str, str] = {}
    n = 0
    for pattern in RENDER_DIR_GLOBS:
        for d in ROOT.glob(pattern):
            if not d.is_dir():
                continue
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    seen.setdefault(sha256_of(f), str(f.relative_to(ROOT)))
                    n += 1
    return seen, n


def _assert_blind(packet_dir: Path, records: list[dict],
                  frame_records: list[dict] | None = None) -> dict:
    """Prove the packet leaks nothing. Raises SystemExit on ANY doubt.

    Four independent assertions, in increasing order of what they would catch:

      A. byte identity  -- every packet image hashes equal to the source
                           photograph it was copied from. Nothing was drawn on it,
                           nothing was re-encoded; the file IS the photograph.
      B. provenance     -- every source path resolves inside data/raw or
                           data/cleaned_1024. Catches a packet built, plausibly and
                           wrongly, from <gt>/renders/.
      C. render-disjoint-- no packet image hash appears anywhere in any render or
                           overlay directory in the repo. Catches A+B being
                           satisfied by a render that was copied into data/raw.
      D. no side-channel-- the packet holds images plus exactly three known text
                           files, and no .json at all. Catches a netlist, a
                           decision record or a notes file riding along.

    ``frame_records`` are the 1024 frames the annotator's coordinates refer to.
    They go through A, B and C on exactly the same terms as the photographs --
    a preprocessed frame is still a frame of the drawing, and a render dropped
    into cleaned_1024 would leak just as completely as one dropped into raw/.
    """
    problems: list[str] = []
    frame_records = frame_records or []

    # A + B
    for rec in records + frame_records:
        sub = "frames_1024" if rec.get("source_kind") == "frame_1024" else "images"
        src, dst = ROOT / rec["source_path"], packet_dir / sub / rec["file"]
        if not dst.is_file():
            problems.append(f"A: {rec['file']} was not written")
            continue
        dst_hash = sha256_of(dst)
        if dst_hash != rec["sha256"]:
            problems.append(
                f"A: {rec['file']} hash {dst_hash[:12]} != source "
                f"{rec['sha256'][:12]} ({rec['source_path']}) -- the copy was "
                "MODIFIED; a packet built from a render or a re-encode would look "
                "exactly like this")
        rel = str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src)
        if not any(rel.startswith(d + "/") for d in ALLOWED_SOURCE_DIRS):
            problems.append(f"B: {rec['file']} sourced from {rel}, outside "
                            f"{ALLOWED_SOURCE_DIRS}")
        if "render" in rel or "overlay" in rel:
            problems.append(f"B: {rec['file']} source path names a render: {rel}")

    # C
    render_hashes, n_render_files = _render_hashes()
    for rec in records + frame_records:
        if rec["sha256"] in render_hashes:
            problems.append(
                f"C: {rec['file']} is byte-identical to the render "
                f"{render_hashes[rec['sha256']]} -- THE PACKET LEAKS THE ANSWER KEY")

    # D
    allowed_text = {"README.md", "circuits.txt"}
    extra = []
    for f in sorted(packet_dir.rglob("*")):
        if f.is_dir():
            continue
        rel = f.relative_to(packet_dir)
        if rel.parts[0] in ("images", "frames_1024"):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                extra.append(str(rel))
            continue
        if str(rel) not in allowed_text:
            extra.append(str(rel))
    if extra:
        problems.append(f"D: unexpected files in the packet: {extra}")
    leaked_json = [str(f.relative_to(packet_dir))
                   for f in packet_dir.rglob("*.json")]
    if leaked_json:
        problems.append(f"D: JSON in the packet (netlist leak?): {leaked_json}")

    if problems:
        print("\n!!! BLIND-SAFETY ASSERTION FAILED -- packet NOT usable !!!",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        raise SystemExit(2)

    return {
        "byte_identity_checked": len(records) + len(frame_records),
        "photographs_checked": len(records),
        "frames_1024_checked": len(frame_records),
        "source_dirs_allowed": list(ALLOWED_SOURCE_DIRS),
        "render_files_hashed": n_render_files,
        "render_hash_collisions": 0,
        "packet_json_files": 0,
        "assertion": (
            "every packet image is byte-identical (sha256) to an untouched "
            "photograph under data/raw or to its deterministic 1024 preprocessing "
            "under data/cleaned_1024, is hash-disjoint from every render/overlay "
            "directory in the repo, and the packet contains no JSON and no file "
            "other than images/, frames_1024/, README.md and circuits.txt"),
    }


PACKET_README = """\
# Blind re-annotation packet -- connectivity ground truth

You are the independent second annotator. The packet holds {n} hand-drawn circuit
schematics, in two forms of the same drawings:

| directory | what it is | use it for |
| --- | --- | --- |
| `images/` | the photographs, exactly as they came off the camera (~2000 px) | **looking**: it has the most detail, so zoom into it to read faint pencil |
| `frames_1024/` | the same drawings normalised to a 1024 px frame | **coordinates**: every `[x, y]` you write down must be in this frame |

Both are the drawing and nothing else -- no boxes, no nets, no calls. The
1024 frame exists because the annotation schema records positions, and a
position is only meaningful once both passes agree on the frame it is in. Read
faint ink in `images/`, write coordinates from `frames_1024/`.

## What to produce

For every image, one JSON file recording, for each component, which electrical
**net** each of its terminals connects to, and -- at every place where wire ink
meets -- whether the wires are **joined** or one **passes over** the other.

The file format, the net naming rule, the terminal-numbering convention for
3-terminal parts, and the worked examples are all in:

    docs/ANNOTATION_GUIDE.md

Read it before you start. If anything in the guide contradicts this README, the
guide wins.

## Ground rules

1. **Work only from the photographs.** Do not look at any other directory in
   this repository, and in particular do not open anything under `data/gt_*`,
   `results/`, or any file named `*netlist*`, `*decision*` or `*render*`. An
   existing annotation of these same images exists; the entire value of your
   pass is that you have not seen it.
2. **Do not run the pipeline** or any VLM on these images, and do not consult
   any automatic tool's opinion about connectivity. If you want a magnified
   view, zoom the photograph.
3. **Record judgement calls, don't hide them.** Where a crossing has neither a
   solder dot nor a hop, both readings are defensible; say which you chose, at
   which coordinates, and what made you choose it. Disagreements with a reason
   attached are useful data. Silent ones are noise.
4. **Do not skip a circuit because it looks ambiguous.** Ambiguous circuits are
   the ones this exercise is measuring. Annotate it and flag it.
5. **Do not consult the other annotator** (or anyone who has seen the first
   pass) about a specific circuit until your pass is complete and delivered.

## What you are NOT told, on purpose

These circuits were sampled in strata -- some at random, some because they hold
a 3-terminal device, some because several automatic systems disagreed with the
existing annotation. You are not told which is which, because knowing that a
particular drawing is "one of the hard ones" changes how long you look at it and
what you expect to find. Treat every image as equally likely to be routine.

## Delivering

Hand back one directory containing, per image:

    <stem>.json             the netlist: your components, and the net each
                            terminal sits on
    decisions/<stem>.json   your call at each wire-ink intersection, plus notes

### Recording intersections by coordinate

Give each call the position you saw it at, **in the `frames_1024/` frame**:

```json
{{
  "sites_xy": [
    {{"xy": [434, 869], "call": "crossing"}},
    {{"xy": [612, 240], "call": "junction"}}
  ],
  "notes": "S(434,869): plain X, no dot and no hop -- read as a crossing because ..."
}}
```

`call` is one of `junction`, `crossing`, `none`, or an explicit edge grouping.

**Why coordinates and not index numbers.** The existing annotation numbers its
intersections, but that numbering is derived from where *that* annotator drew the
component boxes -- so the numbers are a fact about their pass, not about the
drawing, and you cannot be given them without being given part of their answer. A
position in a shared frame is the one thing both passes can name independently.

A coordinate is matched to an intersection within 12 px. If two intersections are
that close together, or two of your coordinates land on the same one, the call is
reported as unresolved rather than guessed -- so put the coordinate on the ink you
mean, and don't worry about hitting it exactly.

### Scoring

Comparison against the existing annotation is automatic:

    python scripts/compare_annotations.py --gt-b <your output directory>

`circuits.txt` lists the stems in this packet, in no meaningful order.

**Three circuits are already known not to support the site comparison**
(`circuit_858`, `circuit_557`, `circuit_218`): the first pass's own intersection
numbering has drifted relative to the tracer, so its calls there cannot be
trusted to name the ink they once named. Annotate them exactly like the rest --
their nets, pin order and components are compared normally, and only the
per-site agreement excludes them. See `results/blind_review/site_evidence_coverage.json`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/splits/test.txt")
    ap.add_argument("--gt-dir", default="data/gt_test_1024")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--fallback-dir", default="data/cleaned_1024")
    ap.add_argument("--pipeline-csv",
                    default="results/paper_test/seeds/seed0/per_image.csv")
    ap.add_argument("--vlm-csv", nargs="*", default=[
        "results/vlm/claude_b/scored/per_image.csv",
        "results/vlm/openai_b/scored/per_image.csv"])
    ap.add_argument("--out", default="results/blind_review/packet")
    ap.add_argument("--manifest", default="results/blind_review/manifest.csv",
                    help="written OUTSIDE the packet on purpose -- it carries the "
                         "stratum labels, which the annotator must not see")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--n-uniform", type=int, default=20)
    ap.add_argument("--n-multi-terminal", type=int, default=20)
    ap.add_argument("--n-hard-core", type=int, default=18)
    args = ap.parse_args()

    split_path = ROOT / args.split
    gt_dir = ROOT / args.gt_dir
    packet_dir = ROOT / args.out
    stems = read_stems(split_path)
    stem_set = set(stems)
    print(f"split {args.split}: {len(stems)} images")

    # --- pools -------------------------------------------------------------
    mt_pool, mt_stats = multi_terminal_pool(gt_dir, stems)
    hc_pool, hc_report = build_hard_core(
        ROOT / args.pipeline_csv, [ROOT / p for p in args.vlm_csv], stem_set)

    print(f"multi-terminal pool: {len(mt_pool)} images hold a 3+-terminal device "
          f"({mt_stats['devices_total']} devices)")
    for rep in hc_report["vlm"]:
        print(f"VLM {rep['path']}: {rep.get('stems', 0)} scored stems, "
              f"{rep['in_split']} of them in the current split")
    if hc_report["fallback_used"]:
        print("")
        print("!" * 78)
        print("! HARD-CORE STRATUM FELL BACK TO PIPELINE-ONLY FAILURES.")
        print("! The VLM per-image scores share ZERO images with the current test")
        print("! split -- they were scored before the 2026-08-03 role swap, so")
        print("! despite their filenames they score what is now the VALIDATION")
        print("! split (data/README.md, 'the 2026-08-03 role swap'). A three-way")
        print("! disagreement therefore CANNOT be computed for these images, and")
        print("! this stratum is the weaker 'pipeline disagrees with GT' signal.")
        print("!" * 78)
        print("")
    print(f"hard-core pool: {len(hc_pool)} images ({hc_report['definition']})")

    # --- sampling ----------------------------------------------------------
    # The uniform stratum is drawn FIRST and from the WHOLE split; see docstring.
    rng = random.Random(args.seed)
    chosen: dict[str, str] = {}

    uniform = rng.sample(sorted(stems), min(args.n_uniform, len(stems)))
    for s in uniform:
        chosen[s] = "uniform"

    mt_avail = sorted(set(mt_pool) - chosen.keys())
    mt = rng.sample(mt_avail, min(args.n_multi_terminal, len(mt_avail)))
    for s in mt:
        chosen[s] = "multi_terminal"

    hc_avail = sorted(set(hc_pool) - chosen.keys())
    hc = rng.sample(hc_avail, min(args.n_hard_core, len(hc_avail)))
    for s in hc:
        chosen[s] = "hard_core"

    counts = Counter(chosen.values())
    print(f"sampled {len(chosen)}: " +
          ", ".join(f"{k}={counts[k]}" for k in
                    ("uniform", "multi_terminal", "hard_core")))
    if not 50 <= len(chosen) <= 60:
        print(f"WARNING: packet size {len(chosen)} is outside the intended 50-60",
              file=sys.stderr)

    # --- copy --------------------------------------------------------------
    images_dir = packet_dir / "images"
    frames_dir = packet_dir / "frames_1024"
    if packet_dir.exists():
        # Rebuild the image directories only; never touch anything outside them.
        for d in (images_dir, frames_dir):
            for f in d.glob("*"):
                f.unlink()
        # Finder writes .DS_Store into any directory it is asked to display, and
        # assertion D correctly refuses to ship a packet holding a file it cannot
        # account for. Sweep them as part of the rebuild rather than relaxing the
        # assertion: an unexplained file in a blind packet should always be an
        # error, and this makes the one benign case stop recurring.
        for junk in packet_dir.rglob(".DS_Store"):
            junk.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    raw_dir, fb_dir = ROOT / args.raw_dir, ROOT / args.fallback_dir
    records = []
    for stem in sorted(chosen):
        src, which = resolve_source(stem, raw_dir, fb_dir)
        dst = images_dir / f"{stem}{src.suffix.lower()}"
        shutil.copyfile(src, dst)          # copyfile, not copy2: bytes only
        records.append({
            "stem": stem,
            "stratum": chosen[stem],
            "file": dst.name,
            "source_path": str(src.relative_to(ROOT)),
            "source_kind": which,
            "sha256": sha256_of(src),
            "bytes": src.stat().st_size,
            "n_multi_terminal_devices": mt_stats["per_stem"].get(stem, 0),
            "pipeline_strict_failure": stem in set(hc_pool),
        })
    n_fallback = sum(1 for r in records if r["source_kind"] == "fallback")
    print(f"copied {len(records)} images "
          f"({len(records) - n_fallback} raw, {n_fallback} cleaned fallback)")

    # --- the coordinate frame ----------------------------------------------
    # The photographs are ~2000 px on the long side; the annotation schema, the
    # GT bounding boxes and the tracer's site coordinates are all in the 1024
    # frame. Shipping only the photographs asks the annotator to write down
    # coordinates in a frame they were never given, and compare_annotations.py
    # has a whole branch dedicated to catching the result after the fact
    # ("frame_mismatch_suspected"). Ship the frame instead of detecting its
    # absence: images/ stays the record of what was photographed and is the
    # better thing to zoom into, frames_1024/ is what coordinates refer to.
    #
    # This leaks nothing. cleaned_1024 is deterministic geometric and tonal
    # normalisation of the same photograph -- it carries no component box, no
    # net and no site -- and it was already an allowed packet source, so the
    # same four assertions below cover it unchanged.
    frame_records = []
    for stem in sorted(chosen):
        src = ROOT / args.fallback_dir / f"{stem}.jpg"
        if not src.is_file():
            raise SystemExit(
                f"no 1024 frame for {stem} in {args.fallback_dir}. The packet "
                "must not ship photographs without the frame their coordinates "
                "are expressed in; preprocess the split first.")
        dst = frames_dir / src.name
        shutil.copyfile(src, dst)
        frame_records.append({
            "stem": stem, "file": dst.name,
            "source_path": str(src.relative_to(ROOT)),
            "source_kind": "frame_1024",
            "sha256": sha256_of(src), "bytes": src.stat().st_size,
        })
    print(f"copied {len(frame_records)} 1024 frames "
          f"({sum(r['bytes'] for r in frame_records) / 1e6:.1f} MB)")

    # --- packet text (no strata!) ------------------------------------------
    order = sorted(chosen)
    rng.shuffle(order)
    (packet_dir / "circuits.txt").write_text("\n".join(order) + "\n")
    (packet_dir / "README.md").write_text(PACKET_README.format(n=len(records)))

    # --- prove it is blind BEFORE anyone can ship it -----------------------
    blind = _assert_blind(packet_dir, records, frame_records)
    print(f"blind-safety: OK -- {blind['byte_identity_checked']} images "
          f"({blind['photographs_checked']} photographs + "
          f"{blind['frames_1024_checked']} frames) byte-identical to source, "
          f"hash-disjoint from {blind['render_files_hashed']} render files, "
          f"0 JSON in packet")

    # --- manifest (OUTSIDE the packet) -------------------------------------
    manifest = ROOT / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stem", "stratum", "seed", "file", "source_path", "source_kind",
            "sha256", "bytes", "n_multi_terminal_devices",
            "pipeline_strict_failure"]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in records:
            w.writerow({**rec, "seed": args.seed})

    meta = {
        "seed": args.seed,
        "git_sha": git_sha(),
        "split": args.split,
        "split_images": len(stems),
        "gt_dir": args.gt_dir,
        "packet_dir": str(packet_dir.relative_to(ROOT)),
        "manifest": str(manifest.relative_to(ROOT)),
        "n_sampled": len(records),
        "strata": dict(counts),
        "sampling_order": ["uniform (from the whole split)",
                           "multi_terminal (pool minus uniform)",
                           "hard_core (pool minus the above)"],
        "pools": {
            "multi_terminal": {k: v for k, v in mt_stats.items()
                               if k != "per_stem"},
            "hard_core": hc_report,
        },
        "sources": {"raw": len(records) - n_fallback, "cleaned_fallback": n_fallback},
        "blind_safety": blind,
        "manifest_not_in_packet_because": (
            "the stratum label tells the annotator which circuits are expected to "
            "be hard, which would bias exactly the circuits the strata exist to "
            "test; the packet ships circuits.txt (stems only, shuffled)"),
        "rejected_check_saturation_heuristic": (
            "fraction of strongly-saturated pixels does NOT separate renders from "
            "raw photographs: renders min 0.0019 over the 192 test renders, but "
            "circuit_657/circuit_420/circuit_934 raw photographs reach 0.013-0.016. "
            "Byte identity is used instead."),
    }
    (manifest.parent / "sampling_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {manifest.relative_to(ROOT)} and "
          f"{(manifest.parent / 'sampling_meta.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
