# Test-split ground truth — verification report

**Scope**: all 192 images of the split this report was written against. It was
called `val` at the time; on 2026-08-03 it became the **test** split, because
every tuned parameter had been selected on the other one. The images and the
annotation are unchanged — see `data/README.md` → "the 2026-08-03 role swap".

**Output**: `data/gt_test_1024/` (written as `data/gt_val_verified/`) — 192 JSON
files, 192 overlay renders, and the per-image decision record that produced them.

**Sign-off**: taken 2026-08-03 by Ammaar Junaid. Every file now carries
`"verified": true`, `"annotator": "Ammaar Junaid"` and a `"verified_via"` field
pointing back at this report. Until then they shipped `false`/`null` on purpose,
so that flipping them would be a separate deliberate act rather than a default —
`scripts/benchmark.py` refuses to score unverified GT without
`--include-unverified`, so nothing could be reported before the sign-off
happened.

**Not covered by this sign-off**: the adjudication is the author's own. A mentor
second-read has not happened and is still open for the manuscript.

---

## 1. What was actually verified

| | |
| --- | --- |
| images | 192 / 192 |
| components | 2564 |
| terminals | 5090 |
| nets | 1497 |
| terminals with no net | 1 (one deliberately dangling op-amp input, `circuit_338` #14, flagged `"unconnected": true`) |
| files failing the schema/ERC check | 0 |

Frame: bounding boxes are in the **1024 px** frame (`data/cleaned_1024/`), matching
the geometry already in `data/gt_val50_preswap/`. Each file records `"bbox_frame": "cleaned_1024"`.

Starting point: `data/gt_val50_preswap/` held 192 files, but only **50** carried nets — the
other 142 had every terminal `null`. The 50 were annotated to a lower standard
(see `data/README.md`). Rather than build on them, the nets and notes were
stripped from the working copy so this pass had no prior to anchor on; the
component inventory and bounding boxes (which come from the published
Digitize-HCD COCO annotations) were kept. `data/gt_val50_preswap/` itself is untouched.

## 2. Method

Ground truth was **not** produced by running the pipeline and accepting it. Each
image went through:

1. **Wire extraction.** Hysteresis binarisation (so a faint pen stroke does not
   sever a wire), erase the annotated component boxes, drop ink blobs that touch
   no component, skeletonise, and build a node/edge graph over the result.
2. **Criticality analysis.** For every intersection of wire ink, flip the
   junction/crossing call and see whether the partition of terminals changes.
   Only the sites that change it need adjudication — a mean of 6.8 per image out
   of ~19 candidates. The rest are handwritten value labels brushing a wire.
3. **Adjudication by eye, on evidence.** Every critical site was reviewed at 5×,
   escalating to 8–24× zoom and to a raw ASCII pixel dump where the call was
   close. A solder dot is a symmetric blob roughly 3× the stroke width in both
   axes; a pen lead-in, a corner or two overlapping strokes is not. That
   distinction is a one-pixel question and no rendered crop settles it.
4. **Deterministic net construction.** Nets are recomputed from the recorded
   decisions, never hand-edited. Every shipped netlist is reproducible from its
   decision file (verified: 192/192, see §6).
5. **Electrical rule check** on every file: shorted voltage source, dangling
   branch, one-terminal net, disconnected island, GND not on net `"0"`,
   transistor with all three pins on one net, terminal count vs class.
6. **Second reader.** Independent re-derivation of a sample and of every flagged
   file, without reading the first pass's reasoning first.

Volume of judgement actually recorded:

| decision | count |
| --- | --- |
| intersection sites explicitly decided | 2047 (1707 junction, 194 crossing, 63 explicit edge groups, 83 no-join) |
| terminals repointed to a different lead | 1261 |
| gap bridges rejected as not-touching | 4 |
| wires re-joined across a scan gap | 38 |
| nets asserted where the box swallowed the contact | 40 |
| component classes corrected | 3 |
| components marked deliberately unconnected | 1 |

## 3. What this pass found

**The dominant error mode is not the one the guide predicts.** It is a terminal
landing on a stroke of the **handwritten value label** — "5KΩ", "100µF" — that
brushes the component box, instead of on the real lead. 1261 terminals had to be
repointed. The tell is a net with only one terminal, which is why the
one-terminal-net check is in the ERC pass rather than being advisory.

**Pin order on 3-terminal parts was wrong far more often than net grouping.**
Every op-amp on `circuit_1238`, `circuit_1240`, `circuit_142`, `circuit_171` and
`circuit_225` had the wrong pin order; 4/4 PNPs on `circuit_118`, 4/4 on
`circuit_1273`, 5/9 on `circuit_140`, 4/6 MOSFETs on `circuit_985`. Orders were
read from the drawn evidence — the BJT emitter from the arrowhead, the op-amp
inputs from the `+`/`−` glyphs, the MOSFET gate from which lead sits on the gate
bar rather than a channel segment — not from a geometric heuristic. This matters
because pin order is scored and **no net-grouping or ERC check can catch it.**

**Hop conventions are drafter-specific and consistent within a sheet.** Where a
drafter marks a non-connection they do it the same way every time — a
semicircular hop, a sideways jog, or a downward dip under the crossing wire. The
useful corollary: on a sheet with no hops anywhere, a bare crossing is a
junction; on a sheet with eight drawn hops, a bare crossing next to them is
evidence of intent. Several calls were settled this way and the reasoning is in
each file's `notes`.

**Three files initially shorted a voltage source** (`circuit_83`, `circuit_249`,
`circuit_858`). All three were over-merges, and all three were found by the same
technique: measure stroke width row by row at every rail/column contact on the
sheet. Genuine solder dots go from a 3 px stroke to 9–16 px; the contacts
causing the shorts went 3→4. On `circuit_858` two contacts 137 px apart on the
same column and the same rail differ exactly that way. All three now have 7 nets
and are clean.

**A note on the OTHER split's GT** (the 190 images, now the validation split).
`circuit_1018` in `data/gt_netlists_verified_v2`
(that split's canonical annotation) merges two electrically separate top branches into
one net. The drawing is a simple series loop — V-DC → L → (node with C to
ground) → R → I-DC → ground, four nets — and the GT records three. Worth a look,
since every pre-swap benchmark number is computed against that set.

## 4. Judgement calls that a reader should know about

These are in the per-file `notes` too, but they are the ones where the
annotation departs from a literal reading of the ink, or where the ink is
self-contradictory:

- **`circuit_513`, site (423,722).** The ink is unambiguous — a plain T on a
  continuous column — and read literally it short-circuits the 15 V source. The
  rail was severed there instead. Justification: across the 190 human-verified annotations of
  the other split there are three sources with both terminals on one net and
  **all three are current sources; a shorted voltage source occurs 0/190 times**.
  This image's GT therefore encodes an electrically-corrected reading rather than
  a literal one. It is the only place in the set where that was done knowingly.
- **`circuit_1175` #23**, labelled `BJT-PNP`, is drawn with the emitter arrow
  pointing away from the base bar (the NPN convention), and the circuit context
  agrees with NPN. The published class was kept. Terminal order is unaffected.
- **Drafter labels that contradict the drawn symbol** appear on several sheets —
  a capacitor captioned "50 MH", an inductor captioned "50 mF", a current source
  captioned "20 cos t V". The COCO class follows the symbol and was kept; each is
  flagged in that file's notes in case anything downstream cross-checks values.
- **Two files carry an explained ERC warning**: `circuit_1207` #8 and
  `circuit_220` #8 are current sources with both terminals on one net. That is
  legal, matches the drawing, and matches the verified test set (which contains
  three of them).

## 5. Confidence

Independent second-reader re-derivation, drawing first and notes only afterwards:

| sample | files | 3-terminal parts | disagreements |
| --- | --- | --- | --- |
| blind random | 12 | 0 | 0 |
| stratified on sheets with 3-terminal parts | 5 | 38 | 0 |
| flagged by cross-checks | 9 | — | 7 changed |

The blind sample contained no 3-terminal components, which is why the second,
stratified sample exists: pin order is the failure mode a net check cannot see,
and 38 components were re-read from the arrowheads and glyphs with zero
disagreements.

The flagged set is where the errors were, which is the point — the flags work.
Files were flagged by (a) any ERC error, and (b) disagreeing with 4 or more
*unanimous* verified siblings on net count. Sibling comparison was **calibrated
before being trusted**: these sheets are hand drawings of generated circuits and
the same circuit was often drawn more than once, but within the verified set
itself sibling pairs share a net count only 53% of the time on small sheets and
65% on large ones — so "differs from siblings" alone is weak. The sharp version
is the unanimity test: a verified file disagrees with 4+ unanimous siblings only
~7% of the time. Three files tripped it; all three were re-read.

Residual risk, stated plainly: a bare X crossing on a sheet with no other
crossings to calibrate against is decided by circuit sense, not by ink. Those
calls are individually documented, but they are the ones most likely to be
disputed.

## 6. Reproducibility

- `data/gt_test_1024/decisions/<stem>.json` — the decision record per image:
  every site call, every terminal repointing, every asserted net, and the notes.
- `data/gt_test_1024/renders/<stem>.png` — the drawing with wire ink coloured
  by net and every terminal labelled `<component>.<terminal>=<net>`, generated
  from the shipped JSON.
- `scripts/gt_val_tools/` — the tracer, net builder, ERC checker, zoom/pixel-dump
  helpers and the annotator brief. `tools_v1/` is the earlier revision of the
  tracer; 26 of the 192 images were annotated against it, before the port-scoring
  and port-recovery fixes, and are reproducible only with it.
  `render_provenance.json` records which version applies to each image.
- Every one of the 192 shipped netlists was re-derived from its decision file
  and matched exactly.

## 7. To adopt

```bash
python scripts/annotate_topology.py --check --gt-dir data/gt_test_1024
# 192 GT file(s): 0 verified, 192 unverified, 0 with validation issues
```

Already run, using the project's own `schematic2netlist.gt.validate_gt` with
the class whitelist enabled **and `strict=True` forced on** — that is, the full
set of checks that only fire once `verified` is true (every terminal has a net
or is marked unconnected, terminal counts match the class, GND sits on `"0"`,
no net touches a single terminal). Result: **0 of 192 files have a validation
issue.** So flipping the flag will not surface anything new.

The `0 verified` is correct and intended — the sign-off is yours. Setting
`"verified": true` and `"annotator": "<name>"` across the set is a one-line
script once you have reviewed it.
