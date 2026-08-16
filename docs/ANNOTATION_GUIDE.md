# Annotation guide — second annotator

You are re-annotating the **connectivity** of hand-drawn schematics: for each
drawn component, which electrical *net* each terminal sits on. A **net** is a
maximal set of terminals joined by wire. That is the whole job — no SPICE
needed, and you are never asked whether the circuit works.

Annotate **independently**: do not read the existing annotation for an image
before finishing your own. Read this guide, then the eight
[worked examples](#worked-examples).

---

## 0. The rule that overrides every other rule: annotate the topology AS DRAWN

**Ground truth records what is visibly drawn. Not a corrected, more functional,
or easier-to-simulate version of the drawing.**

If the ink says two wires meet, they meet — even if that shorts a voltage
source, strands a component, or makes the circuit impossible to simulate. You
record the short, then note that you can see it is probably not what the drafter
meant, and why. The observation is evidence, not a licence to move the wire.
Anything you would have to *add* to make the circuit work goes in the note,
never into the topology: a placeholder value (values are out of scope
entirely), an assumed ground (never net `"0"`), an inferred source or diode
polarity (terminal order is not polarity — §6), a floating node tied up so
nothing dangles (mark it unconnected instead — §4).

This is not pedantry. The annotation is the yardstick a machine pipeline is
scored against, and a yardstick that quietly repairs drawings marks a correct
reading *wrong* — the published numbers then measure agreement with a circuit
nobody drew.

**Known exceptions — do not copy them.** An earlier brief carried the opposite
rule ("electrical impossibility wins") and a few files were annotated under it.
That rule is withdrawn. `circuit_513` has been corrected to the as-drawn reading
and is currently the only file recorded as un-simulable as drawn
([ex. 8](annotation_examples/08_as_drawn_short.png)). Four sites still override
the ink and are pending review: `circuit_39` S50, `circuit_209` calls (d) and
(e), `circuit_220`'s bottom rail — whose note says in as many words "THIS IS THE
ONE PLACE IN THIS IMAGE WHERE I OVERRODE THE INK" — and `circuit_1059` S45. If
you meet one you are not misreading this guide: annotate as drawn and let the
disagreement stand. A disagreement is a result, not a failure.

---

## 1. What you hand back

**Use the annotation tool.** It writes the right shape for you, autosaves, and
resumes where you left off:

```
python tools/annotator/server.py --blind      # then open http://127.0.0.1:8765
```

Per circuit: press <kbd>b</kbd> and drag a box round each symbol, pick its class,
press <kbd>t</kbd> and click its terminals **in port order** (§7), typing a net
name for each. Press <kbd>i</kbd> and click every wire-ink intersection, then
<kbd>j</kbd>/<kbd>k</kbd>/<kbd>e</kbd>/<kbd>o</kbd> to call it. <kbd>Enter</kbd>
submits. When you are done with all of them:

```
python scripts/annotator_to_gt.py             # converts your work for scoring
```

That script writes nothing it had to guess — if a component has no box, or a
terminal has no net, it names the circuit and refuses it rather than inventing
the missing piece.

### The two files it produces, if you would rather write them by hand

`<stem>.json` is the netlist — your components, each with a class, a bounding
box, and one terminal per port carrying the net it sits on. `decisions/<stem>.json`
is your reading of the ink:

```json
{
  "sites_xy": [
    {"xy": [434, 869], "call": "crossing"},
    {"xy": [612, 240], "call": "junction"}
  ],
  "interventions": [],
  "notes": "Net map: ... Judgement calls: ..."
}
```

**Coordinates are in the 1024 px frame** — `frames_1024/` in your packet. Zoom
`images/` to read faint pencil; take coordinates from the 1024 frame.

**`call`** is one of `"junction"` (all branches become one net), `"crossing"`
(opposite branches pass through each other, two nets), `"none"` (join nothing
here), or an *edge-group* `[[e,…],[e,…]]` — each inner list one electrical
group, used only for a drawn crossing split across two nearby sites. **Record a
call at every critical intersection, including ones you expect to be routine**:
an absent one is indistinguishable from not having looked, and the two passes'
calls are compared item by item, with a Cohen's kappa, by
`scripts/compare_annotations.py`.

> **Why coordinates and not index numbers.** The first pass numbers its
> intersections, but that numbering is derived from where *that* annotator drew
> the component boxes — so it is a fact about their pass, not about the drawing,
> and you cannot be given it without being given part of their answer. A
> coordinate in a shared frame is the one thing both passes can name
> independently. Yours is matched to an intersection within 12 px; if two
> intersections are that close together the call is reported as unresolved
> rather than guessed, so put the coordinate on the ink you mean and don't worry
> about hitting it exactly.

**`interventions`** — repairs you *would* apply, never folded into the topology
(§0). **`notes`** is the most valuable thing you produce: one line per net ("n3 =
collector of Q4, top of R7, right end of the y≈310 rail"), and every judgement
call with coordinates, the reading chosen, and what the other reading would have
implied. Net names are arbitrary — only the *grouping* is compared, never the
names. One name is reserved (§4).

---

## 2. Junctions

**A T is always a junction.** One wire *ends* on another; a wire that stopped
dead without joining would just be a dangling wire, so there is nothing to
decide. ([ex. 1](annotation_examples/01_solder_dot.png))

**A crossing with a solder dot is a junction.** The dot test is a measurement,
not an impression, and no thumbnail settles it — zoom in. A genuine dot is a
**symmetric blob, roughly three times the stroke width in both axes**: on these
sheets a ~3 px stroke goes to **9–16 px**, while contacts that turned out *not*
to be dots went 3 px → 4 px. That gap is wide, and it is the most reliable
discriminator in the task.

**Ink that is not symmetric is not a dot.** A pen lead-in, a corner, a restart
where the drafter lifted the pen, or two strokes overlapping all leave extra ink
— wide and flat, tall and thin, or offset to one side rather than centred on the
meeting point. ([ex. 2](annotation_examples/02_pen_lead_in_not_a_dot.png) —
12 px wide, 5 px tall, so not a dot.)

---

## 3. Wire crossings

**A drawn hop is a crossing.** Where a drafter marks a non-connection they use a
semicircular hop, a sideways jog, or a downward dip. It settles the site by
itself. ([ex. 3](annotation_examples/03_drawn_hop.png))

**Hop conventions are per-drafter and consistent within a sheet** — the most
useful single fact in the task, and it cuts both ways. Where the drafter hops
*everywhere*, a bare crossing is evidence of intent: they had a way of saying
"not connected" and did not use it. Where there are *no* hops anywhere, a bare
crossing tells you nothing and you fall back to §5. So count the hops on the
sheet before deciding any crossing, and put the count in your note — it is the
evidence behind every crossing call you make there.
([ex. 4](annotation_examples/04_bare_crossing.png) — the one bare crossing on a
sheet where every other one is hopped.)

**A terminal on handwritten ink is not a connection.** The first pass's most
common error by a wide margin (1,261 terminals moved) was a terminal landing on
a stroke of the *handwritten value label* — "5KΩ", "100µF", "3V" — brushing the
component. Handwriting is never a conductor; the tell is a net with only one
terminal on it. Check every part whose value is written against its body.
([ex. 7](annotation_examples/07_terminal_on_label_ink.png))

---

## 4. Missing grounds, and things that only look missing

**The net touching any ground symbol is named `"0"`** — all ground symbols in
one image share it, even drawn far apart and never joined by ink, because they
are one reference by definition. That is the only naming rule. **A sheet with no
ground symbol has no net `"0"`; do not invent one.** Twelve of the 192 test
images have none, and §0 applies. **A lead drawn going nowhere stays
unconnected** — terminal `null`, component listed in `unconnected`; do not
attach it to the nearest wire to tidy up.

**But check first whether a box swallowed the contact.** Component bounding
boxes come from the published dataset and some cover a whole lead stub.
Automatic tracing erases the boxes and so sees a component with no lead at all,
while your eye can see the stem running to the rail. Record what you can see,
via `manual_nets`: this is an artefact of where a box was drawn, never evidence
of a missing wire.
([ex. 5](annotation_examples/05_box_swallowed_the_contact.png))

---

## 5. Ambiguous connections

A bare crossing with nothing on the sheet to calibrate against is the genuinely
hard case, and the one most likely to be disputed. Work it in order. **The ink**
first — dot or no dot (§2), hop or no hop (§3), zooming to 8–24×, since most
"ambiguous" sites stop being ambiguous at 10×. Then **the drafter's habit on
this sheet** (§3). **Only then circuit sense**, as a tie-breaker between two
readings the ink genuinely does not distinguish — never as a reason to overrule
ink that *does*. "This reading shorts a source" is a reason to check the pixels
again; if they still say the wires meet, the wires meet (§0).

Record coordinates, the reading chosen, the evidence, **and what the other
reading would have implied**. A documented call a reader can disagree with beats
a confident one they cannot audit; and "undecidable, I chose X" is a legitimate
output.

---

## 6. Symmetric terminals — do not agonise

**For a two-terminal part, which end you call terminal 0 does not matter.** Not
"matters a little": the benchmark cannot see it. Take whatever the tool defaults
to and spend the time on nets.

That is a property of the scoring code (`canonicalize_terminals` in
`src/schematic2netlist/benchmark.py`), which runs on the prediction *and* on the
ground truth before any metric. For every net it collects the set of component
ids touching it; each terminal's signature is the sorted list of the **other
components** sharing that terminal's net (plus a flag for a terminal with no
net); each component's terminals are then sorted by that signature, ties keeping
the recorded order.

Because the same sort is applied to both sides, **swapping two terminals is
invisible to every metric unless the two terminals' nets touch exactly the same
set of other components.** A resistor between a node carrying a capacitor and a
node carrying a source has two distinguishable terminals, and the sort puts them
in the same order however you recorded them. Measured on the shipped 192-file
annotation, 4 of 1,774 two-terminal components and 11 of 376 three-terminal
components have a tied terminal pair — and in all 15 the tied terminals carry
the *same* net, so the swap is a literal no-op. There is currently **no
component in the corpus where swapping two terminals would change any published
number.**

This does not make polarity unimportant to circuits; it means polarity is **not
what this annotation records** — a diode's terminals are its two nets in either
order, and which is the anode is not a field in the file. Nor does it extend to
*which lead* a terminal sits on: a terminal on label ink rather than on the wire
(§3) is a wrong net, and that is scored hard.

---

## 7. Multi-terminal devices — where the errors actually are

Three-terminal parts appear on 58 of the 192 images and are where the first
pass's errors concentrated: every op-amp on `circuit_1238`, `circuit_1240`,
`circuit_142`, `circuit_171` and `circuit_225` had the wrong pin order, as did
4 of 4 PNPs on `circuit_118`, 4 of 4 on `circuit_1273`, 5 of 9 on `circuit_140`,
and 4 of 6 MOSFETs on `circuit_985`.

**Nothing else catches this.** A net-level check sees three pins on three nets
and is happy; so does the electrical rule check; and §6's canonicalising sort
makes a swap invisible to the metric too. So a wrong pin order is a silent,
permanent error in a published artifact — and since the netlist writer emits
`Q<collector> <base> <emitter>`, `M<drain> <gate> <source>` and
`E<out> 0 <in+> <in−>` straight off the terminal index, a swapped pair quietly
becomes a *different circuit*. Your pin order **is** compared against the first
pass, on the raw recorded order, precisely because nothing else sees it.

| Class | index 0 | index 1 | index 2 |
| --- | --- | --- | --- |
| MOSFET-N, MOSFET-P | drain | **gate** | source |
| BJT-NPN, BJT-PNP | collector | **base** | emitter |
| Op-Amp | in+ | in− | out |

Read each pin off the drawn evidence, never off the layout:

- **BJT — the emitter is the lead carrying the arrowhead.** That is the whole
  rule; the base meets the base bar, and the remaining lead is the collector. Do
  *not* use "collector toward the supply, emitter toward ground": transistors
  get drawn upside down, and that heuristic produced the 4-of-4 and 5-of-9 error
  rates above.
- **Op-amp — the inputs come from the drawn `+` and `−` glyphs.** Only the
  output is geometric (the lead at the triangle's apex). On `circuit_1238` all
  five op-amps carry the `−` on the *upper* input, so a top-to-bottom rule gets
  every one backwards ([ex. 6](annotation_examples/06_opamp_pin_order.png)).
  Independent second check: feedback runs from the output back to the
  **inverting** input, so if your reading puts feedback on `+`, re-read the
  glyphs.
- **MOSFET — the gate is the lead landing on the gate bar**, the short bar
  standing off from the channel, not on a channel segment. Drain and source are
  the two touching the channel, and the body arrow points to/from the source.
  Where the drawing does not settle drain vs source, use the physical reading
  and *say in the note that you did*.

A symbol can also be drawn with the wrong convention — `circuit_1175` #23 is
labelled `BJT-PNP` but drawn with the emitter arrow pointing away from the base
bar. Pin roles follow the ink; the class label follows the published annotation
unless clearly wrong.

---

## 8. Component orientation

Not a field you fill in. It matters only as *evidence* for **which lead is
which** on a three-terminal part (§7) — settled by arrowhead, glyphs and gate
bar, not by which way the symbol faces — and for **which net a lead reaches**,
since a symbol rotated 90° has its leads on different sides. Two-terminal parts
have no recorded orientation at all (§6). Where a handwritten caption
contradicts the drawn symbol — a capacitor captioned "50 MH", an inductor
captioned "50 mF" — **the symbol wins**, and the caption goes in the note.

---

## Worked examples

Real crops of real test-split images, regenerated by
`./venv/bin/python scripts/make_annotation_examples.py`. Each caption carries
the coordinates, the call recorded in the shipped annotation, and measured ink
widths.

| | Example | From | Shows |
| --- | --- | --- | --- |
| 1 | [`01_solder_dot`](annotation_examples/01_solder_dot.png) | `circuit_1059` | a genuine solder dot on a T |
| 2 | [`02_pen_lead_in_not_a_dot`](annotation_examples/02_pen_lead_in_not_a_dot.png) | `circuit_1156` | extra ink that is *not* a dot |
| 3 | [`03_drawn_hop`](annotation_examples/03_drawn_hop.png) | `circuit_513` | a drawn semicircular hop |
| 4 | [`04_bare_crossing`](annotation_examples/04_bare_crossing.png) | `circuit_1028` | a bare crossing: no dot, no hop |
| 5 | [`05_box_swallowed_the_contact`](annotation_examples/05_box_swallowed_the_contact.png) | `circuit_150` | a box that swallowed a contact |
| 6 | [`06_opamp_pin_order`](annotation_examples/06_opamp_pin_order.png) | `circuit_1238` | op-amp pin order from the glyphs |
| 7 | [`07_terminal_on_label_ink`](annotation_examples/07_terminal_on_label_ink.png) | `circuit_1059` | a terminal on handwritten label ink |
| 8 | [`08_as_drawn_short`](annotation_examples/08_as_drawn_short.png) | `circuit_513` | as-drawn annotation that shorts a source |

---

## Before you hand an image back

- Every terminal has a net, or its component is in `unconnected`.
- Every ground symbol sits on `"0"`; no ground symbol means no net `"0"`.
- **No net has only one terminal** — almost always label ink (§3). Find the wire.
- Every three-terminal part read individually off arrowhead / glyphs / gate bar (§7).
- Every critical site has an explicit call, including ones you agreed with.
- The note has the net map, and every judgement call with coordinates and the
  consequence of the other reading.
- Nothing in the topology was changed to make the circuit work (§0).

Budget 3–5 minutes for a clean sheet, 8–12 for a dense one with transistors.
Short sessions beat long ones, and a slow answer always beats a guess.
