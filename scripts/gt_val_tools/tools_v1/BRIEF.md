# Ground-truth netlist verification — worker brief

You are acting as an experienced electronics engineer doing **ground-truth
annotation** of hand-drawn circuit schematics. Your output for each image is a
JSON file recording, for every component, which electrical **net** each of its
terminals connects to. These files are the ground truth against which a
published benchmark is scored, so a wrong net is worse than a slow answer.
Work carefully. Do not guess. Do not rush.

## What is already true (do not redo it)

* The **component inventory is correct**: it comes from the published
  Digitize-HCD COCO annotations. Every drawn symbol is present, with the right
  class and a correct bounding box. Re-classifying is almost never needed — but
  if a symbol is visibly a different class than labelled, say so (see
  `classes` below) rather than silently accepting it.
* A **wire tracer** has already extracted the ink, deleted the component boxes,
  skeletonised what is left, and grouped wire runs into nets. It gets most of
  the topology right. It cannot reliably decide **one** thing: at a place where
  two wire runs touch, are they electrically joined (a junction) or does one
  pass over the other (a crossing)? That decision is yours.

## The one concept that matters

An electrical **net** is a maximal set of terminals joined by wire. The net
touching any GND symbol is named `"0"`. Every other net gets an arbitrary label
(`n1`, `n2`, …) — only the *grouping* is scored, never the names.

At an intersection of wire ink:

| What you see in the crop | Reading |
| --- | --- |
| A **T** — one wire ends on another | junction (joined) |
| A crossing with a **solder dot** (a filled blob noticeably fatter than the strokes) | junction |
| A crossing where one wire makes a **semicircular hop / a sideways jog** around the other | crossing (NOT joined) |
| A plain **X** with no dot and no hop | judgement call — see below |

**Plain X with no dot and no hop — decide in this order:**

1. **Electrical impossibility wins.** If one reading short-circuits a voltage
   source, leaves a branch with no return path, strands a component on a
   one-terminal net, or produces a disconnected island, that reading is wrong.
   Take the other one. This overrides everything below.
2. **If both readings are electrically sane, use the ink prior**: in this
   dataset the drafters draw a hop or a jog when they mean "not connected", so
   a plain crossing is usually a real junction.
3. **Check the drafter's own habit on the same sheet.** If they dotted a
   comparable crossing elsewhere and left this one bare, that is evidence.

Whatever you decide, put the coordinates, the reading you chose, and what the
other reading would have implied into `notes`.

**Pixel-level evidence beats a rendered crop.** A solder dot is a symmetric
blob roughly 3x the stroke width in *both* axes. A pen lead-in, a corner, or
two strokes overlapping is not. When it matters, dump the actual pixels:

```
python3 /home/claude/tools/inkmap.py <stem> <x> <y> [win=22]
```

## Terminal order

| Class | Terminals | Order |
| --- | --- | --- |
| Resistor, Capacitor, Inductor, Diode, Zener Diode, V-DC, V-AC, I-DC, I-AC | 2 | left→right or top→bottom as drawn — **order does not affect any metric, leave the default** |
| GND, V-DC (one port) | 1 | the wire it touches |
| MOSFET-N, MOSFET-P | 3 | **drain, gate, source** |
| BJT-NPN, BJT-PNP | 3 | **collector, base, emitter** |
| Op-Amp | 3 | **in+, in−, out** |

For 3-terminal parts the order **is** scored, so check every one of them in the
`_comps` montage:

* **Gate / base** is the odd lead — the one on its own side of the symbol
  (the one touching the vertical bar of a MOSFET, or the base line of a BJT).
* **Drain vs source / collector vs emitter**: the emitter of a BJT carries the
  arrow; a MOSFET's source is the lead the body arrow points to/from. When the
  drawing does not settle it, use the physical reading — drain/collector toward
  the supply, source/emitter toward ground — and note that you did.
* **Op-Amp**: `out` is the lead at the apex of the triangle; `in+` and `in−`
  are the two on the flat side, read from the drawn `+` / `−` marks. If the
  marks are illegible, note it and use the physical reading (the input in the
  negative-feedback path is `in−`).

## Files you are given, per image `<stem>`

```
/home/claude/val/img1024/<stem>.jpg          the drawing (1024x1024) - the ground truth of record
/home/claude/val/gt/<stem>.json              component inventory ONLY (all nets are null - there is
                                             no prior annotation to anchor on; the nets are your job)
/home/claude/pkg/<stem>/<stem>_summary.txt   text report: components, ports, sites, current nets, warnings
/home/claude/pkg/<stem>/<stem>_overview.png  the drawing with the CURRENT net assignment drawn on it
/home/claude/pkg/<stem>/<stem>_comps_*.png   one zoomed crop per component, ports marked p0,p1,... and ->tN
/home/claude/pkg/<stem>/<stem>_sites_*.png   one zoomed crop per intersection site, CRITICAL ones first
/home/claude/pkg/<stem>/<stem>_report.json   the same information, machine-readable
```

Extra evidence whenever you want it:

```
python3 /home/claude/tools/zoom.py   <stem> <x> <y> [win=60] [zoom=6]  # PNG you can Read
python3 /home/claude/tools/inkmap.py <stem> <x> <y> [win=22]           # ASCII pixel dump
python3 /home/claude/tools/edges.py  <stem> [x y radius]               # wire graph: edge ids,
                                                                       # endpoints, site branches
```

## Your procedure

1. **Read the drawing itself** (`img1024/<stem>.jpg`). Understand what circuit
   it is before looking at any machine output. Note the topology you expect.
2. **Read `<stem>_summary.txt`**.
3. **Read `<stem>_overview.png`** and compare it against your reading of the
   drawing. Every terminal is labelled `<comp>.<terminal>=<net>`; wire ink is
   coloured by net. Mismatches are usually explained by one intersection site.
4. **Read every `<stem>_sites_*.png`**. For each site decide junction or
   crossing. The crop is 5× zoom; if it is still ambiguous, use `zoom.py` at
   8–10×. Sites marked CRITICAL change the netlist; the rest are noise (text
   touching a wire, symbol ink) and can be left alone.
5. **Read every `<stem>_comps_*.png`**. Confirm: the class is right; the ports
   marked `->tN` are the real leads (some `pN` marks are stray symbol ink, not
   leads); 3-terminal parts have the right pin order.
6. **Write the decisions file** `/home/claude/dec/<stem>.json` (schema below).
7. **Apply and check**:
   ```
   python3 /home/claude/tools/finalize.py <stem> /home/claude/dec/<stem>.json /home/claude/val
   ```
   It prints the resulting netlist plus ERC errors/warnings. Fix every ERROR.
   Understand every warning — a warning you cannot explain means you are wrong.
8. **Re-render and look again**:
   ```
   python3 /home/claude/tools/pkg.py <stem> /home/claude/val /home/claude/pkg/<stem> /home/claude/dec/<stem>.json
   ```
   Read the regenerated `<stem>_overview.png`. Iterate until the picture matches
   the drawing.
9. **Write `notes`** into the decisions file and re-run step 7 so the note lands
   in the GT file. The note must contain (a) a one-line description of each net
   ("n3 = collector of Q4, top of R7, right end of the y≈310 rail"), and (b)
   every judgement call with its coordinates and the consequence of the other
   reading. This is what makes the annotation auditable.

## Decisions file schema

Every key is optional; an empty `{}` accepts the tracer's answer as-is.

```json
{
  "sites":    {"13": "junction", "14": "crossing", "32": "none",
               "28": [[61, 62, 58], [53, 64]]},
  "bridges":  {"drop": [4]},
  "ports":    {"7": {"0": 2, "1": 0, "2": 1}},
  "classes":  {"9": "MOSFET-P"},
  "merge":    [["5.0", "7.1"]],
  "unconnected": [12],
  "notes": "Net map: ... Judgement calls: ..."
}
```

* `sites` — site id (from the summary / crops) →
  * `"junction"` — merge every branch at this site into one net.
  * `"crossing"` — pass opposite branches through each other (two nets).
  * `"none"` — join nothing here at all.
  * `[[e,e,...],[e,...]]` — explicit groups of **edge ids**: each inner list is
    one electrical group. Use this when one drawn crossing got split into two
    nearby sites, which `junction`/`crossing` alone cannot express. Get the edge
    ids from `edges.py <stem> <x> <y> 25`, which prints every edge with its
    endpoints and every site with its branch edge ids. A group may name any
    edge id, not only the branches of that site. Worked example: a column wire
    hopping a rail was split into S28 and S32; the fix was
    `"28": [[61,62,58],[53,64]], "32": "none"` — i.e. the two arc halves and
    the column stub form one group, the two rail halves the other, and the
    second site adds nothing.
* `bridges.drop` — the tracer joined a wire tip to nearby ink across a gap; drop
  the bridge if the crop shows they do not actually touch.
* `ports` — component id → terminal index → **port index** `pN` from the summary.
  Use this to fix pin order on transistors/op-amps, or to point a terminal at
  the correct lead when the tracer picked stray ink. `null` leaves a terminal
  unresolved.
* `classes` — only when the drawn symbol is clearly a different class.
* `merge` — force terminals onto one net when the wire is broken in the scan
  (e.g. a pencil line that faded). Use sparingly and always explain it in notes.
* `unconnected` — component ids with a lead the drafter drew going nowhere.
  Those terminals stay `null`; without this flag the validator errors.

## Rules

* GT describes **the drawing**, not what the tracer can find. Never delete a
  component. Never invent a connection to make ERC happy.
* Every terminal must end up with a net, unless its component is listed in
  `unconnected`.
* All GND symbols sit on net `"0"`.
* A net touching only one terminal is almost always a mistake — find the wire.
* If after real effort a case is genuinely undecidable, pick the physically
  sensible reading, say so explicitly in `notes`, and move on.

## Done

For each image you must leave behind:

* `/home/claude/dec/<stem>.json` — your decisions, including `notes`
* `/home/claude/out/gt/<stem>.json` — written by `finalize.py`, ERC-clean

Record a decision for every CRITICAL site, even where you agree with the
default — an explicit `"junction"` documents that you looked; an absent key is
indistinguishable from not having checked.

Report back, per image: the number of components, number of nets, which sites
you overrode and why, anything you were unsure about, and any component whose
class you believe is wrong.
