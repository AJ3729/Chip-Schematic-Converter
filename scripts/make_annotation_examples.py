#!/usr/bin/env python3
"""Regenerate the worked-example crops referenced by docs/ANNOTATION_GUIDE.md.

Every example is a real crop of a real test-split drawing, taken from
``data/cleaned_1024/`` (the 1024 px frame the ground-truth bounding boxes
and every coordinate in the decision records are expressed in). Nothing is
drawn, redrawn or synthesised: the ink you see is the ink that was
adjudicated, and the recorded call in each caption is read back out of
``data/gt_test_1024/decisions/<stem>.json`` at generation time, so a
caption cannot drift away from the shipped annotation.

The stroke-width numbers in the captions are measured here, not quoted:
``local ink width`` is twice the maximum Euclidean distance-to-background
inside the ink within 7 px of the site (a robust local stroke width), and
``nearby stroke`` is the median of the same quantity on the ink in a ring
16-26 px out. A genuine solder dot roughly triples the stroke *in both
axes*; a pen lead-in, a corner or two overlapping strokes does not.

Usage::

    ./venv/bin/python scripts/make_annotation_examples.py

Writes docs/annotation_examples/*.png (8 files) and prints one line each.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "cleaned_1024"
DECISIONS = ROOT / "data" / "gt_test_1024" / "decisions"
GT = ROOT / "data" / "gt_test_1024"
OUT = ROOT / "docs" / "annotation_examples"

INK_THRESHOLD = 128  # 8-bit grey level below which a pixel counts as ink
ACCENT = (204, 0, 51)  # markers on the feature under discussion
SECOND = (0, 102, 187)  # markers on the contrasting / wrong feature
PAPER = (255, 255, 255)
TEXT = (17, 17, 17)
MUTED = (90, 90, 90)


# --------------------------------------------------------------------------
# the examples
# --------------------------------------------------------------------------
# site: read the recorded call back out of the decision record.
# port: a terminal->lead assignment, captioned from the decision record.
# note: no machine-readable call to quote (a manual_nets assertion etc.).

EXAMPLES = [
    dict(
        name="01_solder_dot.png",
        stem="circuit_1059",
        site="35",
        centre=(437, 771),
        window=46,
        zoom=7,
        ring=17,
        title="1. A genuine solder dot -> junction",
        body=(
            "Resistor R5's left lead ends on column C. The column's 3 px line "
            "swells symmetrically about the contact and then continues at its "
            "normal width above and below. Symmetric in BOTH axes is the whole "
            "test. This is also a T (a wire ENDING on another wire), which can "
            "never be read as a crossing: a wire that stopped on another wire "
            "without joining it would just be a dangling wire."
        ),
        record="junction",
    ),
    dict(
        name="02_pen_lead_in_not_a_dot.png",
        stem="circuit_1156",
        site="11",
        centre=(540, 472),
        window=46,
        zoom=7,
        ring=17,
        title="2. Ink that is NOT a solder dot",
        body=(
            "There is extra ink here too - a pool about 12 px wide but only "
            "5 px tall, sitting to the LEFT of the meeting point. Wide and flat "
            "is two strokes overlapping or a pen lead-in; a dot would be ~9 px "
            "in BOTH axes, as in example 1. So this site carries no positive "
            "evidence of a junction. It is still recorded as a junction, but on "
            "separate grounds (this drafter draws no hops anywhere, and the "
            "crossing reading leaves the ground symbol doing nothing) - and "
            "that reasoning, not the pool, is what the note has to say."
        ),
        record="junction",
    ),
    dict(
        name="03_drawn_hop.png",
        stem="circuit_513",
        site="11",
        centre=(234, 211),
        window=52,
        zoom=6,
        ring=20,
        title="3. A drawn semicircular hop -> crossing",
        body=(
            "The y~205 tap rail arrives from the right and lifts over column 2 "
            "in a semicircle before carrying on. A hop is an explicit statement "
            "of non-connection and it settles the site by itself. This drafter "
            "hops every crossing on the sheet; that habit is what makes the "
            "bare crossings elsewhere on the same sheet informative."
        ),
        record="crossing",
    ),
    dict(
        name="04_bare_crossing.png",
        stem="circuit_1028",
        site="29",
        centre=(788, 548),
        window=52,
        zoom=6,
        ring=20,
        title="4. A bare crossing: no dot, no hop",
        body=(
            "A clean four-way X. Neither stroke thickens and neither deviates. "
            "This is the sheet's ONLY bare crossing - the drafter hopped every "
            "other rung/column meeting - so 'bare' here reads as a deliberate "
            "non-connection and the site is recorded as a crossing. On a sheet "
            "with no hops anywhere the same picture would read as a junction. "
            "State which of the two situations you are in, in your note."
        ),
        record="crossing",
    ),
    dict(
        name="05_box_swallowed_the_contact.png",
        stem="circuit_150",
        centre=(576, 641),
        window=78,
        zoom=5,
        ring=None,
        component=("circuit_150", 6),
        title="5. A component box that swallowed the contact",
        body=(
            "The published GND box (drawn in blue) covers the lower part of the "
            "stem, so once the box is erased the surviving ink has no free END "
            "touching it and the automatic tracer reports no lead at all for "
            "this ground symbol. The eye has no such problem: the stem plainly "
            "runs up to the bottom rail. Record the net you can see. This is an "
            "artefact of where the published box was drawn, never evidence of a "
            "missing wire."
        ),
        note="terminal 0 asserted onto net \"0\" (manual_nets 6.0)",
    ),
    dict(
        name="06_opamp_pin_order.png",
        stem="circuit_1238",
        centre=(238, 287),
        window=88,
        zoom=5,
        ring=None,
        marks=[
            ((193, 300), "t0 = in+", ACCENT),
            ((192, 273), "t1 = in-", ACCENT),
            ((284, 281), "t2 = out", ACCENT),
        ],
        title="6. Op-amp pin order, read off the glyphs",
        body=(
            "Order is (in+, in-, out). 'out' is the lead at the apex of the "
            "triangle - that part is geometry. Which input is which is NOT "
            "geometry: read the drawn + and - marks. On this sheet the '-' sits "
            "on the UPPER input on all five op-amps, so the upper lead is t1 "
            "and the lower is t0 - the opposite of the textbook layout. A "
            "geometric top-to-bottom rule gets every one of them wrong."
        ),
        note="ports 3: t0->p2, t1->p1, t2->p0",
    ),
    dict(
        name="07_terminal_on_label_ink.png",
        stem="circuit_1059",
        centre=(795, 165),
        window=90,
        zoom=5,
        ring=None,
        marks=[
            ((779, 198), "t1: the real lead", ACCENT),
            ((830, 127), "the '3' of '3V'", SECOND),
        ],
        title="7. A terminal landing on label ink, not on the lead",
        body=(
            "The dominant error mode of the whole corpus: the handwritten value "
            "brushes the component box and gets picked up as if it were a wire. "
            "The V-AC source's lower terminal belongs on the lead leaving the "
            "bottom of the symbol, not on the digit stroke to its upper right. "
            "The tell is a net with only one terminal on it. Handwriting is "
            "never a conductor - check every part whose value is written "
            "against its body."
        ),
        note="ports 0: t1->p1 (repointed off the label ink)",
    ),
    dict(
        name="08_as_drawn_short.png",
        stem="circuit_513",
        site="44",
        centre=(423, 722),
        window=58,
        zoom=6,
        ring=22,
        title="8. As drawn, even when the result cannot be simulated",
        body=(
            "The left tip of the y~722 tap rail lands on column 3's lower "
            "vertical, which runs unbroken to the grounded bottom rail. Plain T, "
            "no ambiguity in the ink. Read literally it puts BOTH terminals of "
            "the 15 V source on net \"0\": the sheet is un-simulable as drawn "
            "and the rule check reports a shorted voltage source. That is the "
            "correct annotation. An earlier revision severed the rail to make "
            "the circuit work; it was reverted. Record the ink; put the doubt "
            "in the note."
        ),
        record="junction",
    ),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def ink_mask(grey: np.ndarray) -> np.ndarray:
    return (grey < INK_THRESHOLD).astype(np.uint8)


def stroke_widths(grey: np.ndarray, x: int, y: int) -> tuple[float, float, str]:
    """(nearby stroke width, local ink width, extent of the thick region).

    Widths are 2 x the Euclidean distance-to-background inside the ink, which
    is immune to the run-length contamination you get from measuring across a
    crossing. The extent is the bounding box of the ink that is at least 6 px
    wide - a solder dot's is roughly square, a lead-in's is not.
    """
    ink = ink_mask(grey)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    h, w = ink.shape
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - y) ** 2 + (xx - x) ** 2

    near = r2 <= 7**2
    local = 2 * float(dist[near].max()) if near.any() else 0.0

    ring = (r2 <= 26**2) & (r2 >= 16**2) & (ink > 0)
    nearby = 2 * float(np.median(dist[ring])) if ring.any() else 0.0

    thick = (ink > 0) & (2 * dist >= 6) & (r2 <= 12**2)
    ys, xs = np.nonzero(thick)
    extent = (
        f"{xs.max() - xs.min() + 1}x{ys.max() - ys.min() + 1} px"
        if len(xs)
        else "none"
    )
    return nearby, local, extent


def recorded_call(stem: str, site: str) -> tuple[str, bool]:
    """(the call for this site, whether it was recorded explicitly).

    A site absent from the record is one the annotator left at the tracer's
    default. That happens for sites where only one reading is possible at
    all - a degree-3 T cannot be a crossing, because a wire that stopped on
    another wire without joining it would just be a dangling wire.
    """
    with open(DECISIONS / f"{stem}.json") as f:
        sites = json.load(f).get("sites", {})
    if site not in sites:
        return "junction (degree-3 T; tracer default accepted)", False
    value = sites[site]
    if isinstance(value, str):
        return value, True
    return ("explicit edge groups" if value else "none"), True


def component_box(stem: str, comp_id: int) -> tuple[str, list[float]]:
    with open(GT / f"{stem}.json") as f:
        for c in json.load(f)["components"]:
            if c["id"] == comp_id:
                return c["class"], c["bbox"]
    raise KeyError(f"{stem}: no component #{comp_id}")


def wrap(draw, text: str, fnt, width_px: int) -> list[str]:
    """Greedy wrap to a pixel width."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render(ex: dict) -> Path:
    src = IMAGES / f"{ex['stem']}.jpg"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} is missing. The example crops are cut from the 1024 px "
            "frame; regenerate it with scripts/preprocess_batch.py or restore "
            "the data artifact."
        )
    grey = np.asarray(Image.open(src).convert("L"))

    cx, cy = ex["centre"]
    win, zoom = ex["window"], ex["zoom"]
    x0, y0 = max(0, cx - win), max(0, cy - win)
    x1, y1 = min(grey.shape[1], cx + win), min(grey.shape[0], cy + win)

    crop = Image.fromarray(grey[y0:y1, x0:x1]).convert("RGB")
    cw, ch = crop.size
    crop = crop.resize((cw * zoom, ch * zoom), Image.NEAREST)
    d = ImageDraw.Draw(crop)

    def to_out(px: int, py: int) -> tuple[float, float]:
        return ((px - x0) * zoom + zoom / 2, (py - y0) * zoom + zoom / 2)

    # the drawn component box, when the point is that the box swallowed a lead
    if ex.get("component"):
        _, (bx, by, bw, bh) = component_box(*ex["component"])
        p0 = to_out(int(bx - bw / 2), int(by - bh / 2))
        p1 = to_out(int(bx + bw / 2), int(by + bh / 2))
        d.rectangle([p0, p1], outline=SECOND, width=2)

    # a hollow ring around the site, large enough to leave the ink untouched
    if ex.get("ring"):
        r = ex["ring"] * zoom
        ox, oy = to_out(cx, cy)
        d.ellipse([ox - r, oy - r, ox + r, oy + r], outline=ACCENT, width=3)

    small = font(15)
    for (px, py), label, colour in ex.get("marks", []):
        ox, oy = to_out(px, py)
        r = 7 * zoom / 2
        d.ellipse([ox - r, oy - r, ox + r, oy + r], outline=colour, width=3)
        tw = d.textlength(label, font=small)
        tx = min(max(4.0, ox - tw / 2), crop.size[0] - tw - 4)
        ty = oy + r + 4 if oy < crop.size[1] / 2 else oy - r - 22
        d.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + 18], fill=PAPER)
        d.text((tx, ty), label, font=small, fill=colour)

    # scale bar: 20 px in the 1024 frame
    bar = 20 * zoom
    bx0, by0 = 10, crop.size[1] - 22
    d.line([(bx0, by0), (bx0 + bar, by0)], fill=TEXT, width=3)
    d.line([(bx0, by0 - 5), (bx0, by0 + 5)], fill=TEXT, width=3)
    d.line([(bx0 + bar, by0 - 5), (bx0 + bar, by0 + 5)], fill=TEXT, width=3)
    d.text((bx0 + bar + 8, by0 - 9), "20 px", font=small, fill=TEXT)
    d.rectangle([0, 0, crop.size[0] - 1, crop.size[1] - 1], outline=(200, 200, 200))

    # ---- caption block -------------------------------------------------
    title_f, body_f, meta_f = font(21), font(16), font(14)
    pad, width = 18, crop.size[0]
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    if ex.get("site"):
        call, explicit = recorded_call(ex["stem"], ex["site"])
        nearby, local, extent = stroke_widths(grey, cx, cy)
        meta = (
            f"{ex['stem']}.jpg  site S{ex['site']} at ({cx},{cy})  |  "
            f"recorded: {call}  |  nearby stroke {nearby:.0f} px, "
            f"local ink width {local:.0f} px, thick region {extent}"
        )
        if explicit and call != ex["record"]:
            raise ValueError(
                f"{ex['name']}: caption says {ex['record']!r} but the shipped "
                f"decision record says {call!r}"
            )
    else:
        meta = (
            f"{ex['stem']}.jpg  at ({cx},{cy})  |  recorded: {ex['note']}"
        )

    body_lines = wrap(scratch, ex["body"], body_f, width - 2 * pad)
    meta_lines = wrap(scratch, meta, meta_f, width - 2 * pad)
    cap_h = pad + 28 + 6 + 21 * len(body_lines) + 8 + 18 * len(meta_lines) + pad

    out = Image.new("RGB", (width, crop.size[1] + cap_h), PAPER)
    out.paste(crop, (0, 0))
    dd = ImageDraw.Draw(out)
    y = crop.size[1] + pad
    dd.text((pad, y), ex["title"], font=title_f, fill=TEXT)
    y += 28 + 6
    for line in body_lines:
        dd.text((pad, y), line, font=body_f, fill=TEXT)
        y += 21
    y += 8
    for line in meta_lines:
        dd.text((pad, y), line, font=meta_f, fill=MUTED)
        y += 18

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ex["name"]
    out.save(path)
    return path


def main() -> int:
    if not IMAGES.exists():
        print(f"error: {IMAGES} not found (data/ is gitignored)", file=sys.stderr)
        return 1
    for ex in EXAMPLES:
        path = render(ex)
        print(f"wrote {path.relative_to(ROOT)}  <- {ex['stem']} @ {ex['centre']}")
    print(f"{len(EXAMPLES)} example crops in {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
