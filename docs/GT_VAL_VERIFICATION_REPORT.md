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

**Not covered by this sign-off**: the adjudication is the author's own. No
independent human has re-read these files. The re-derivation reported in §5 was
performed by an AI assistant, not by a second annotator (see §5). Genuine
independent second annotation by an external annotator is **pending** and is
still open for the manuscript.

---

## 1. What was actually verified

| | |
| --- | --- |
| images | 192 / 192 |
| components | 2564 |
| terminals | 5090 |
| nets | 1496 |
| terminals with no net | 1 (one deliberately dangling op-amp input, `circuit_338` #14, flagged `"unconnected": true`) |
| files failing the schema check | 0 |
| files carrying an explained ERC finding | 3 (`circuit_513` error, `circuit_220` and `circuit_1207` warnings — see §4) |

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
6. **Automated self-consistency re-derivation.** A sample, plus every flagged
   file, re-derived from the drawing without reading the first pass's reasoning
   first — by an AI assistant, not by an independent human reader. What that
   can and cannot establish is stated in §5.

Volume of judgement actually recorded:

| decision | count |
| --- | --- |
| intersection sites explicitly decided | 2047 (1708 junction, 194 crossing, 62 explicit edge groups, 83 no-join) |
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
because **nothing in the evaluation catches it.** No net-grouping or ERC check
can, and — contrary to what this report previously claimed — neither can the
metric cascade: `canonicalize_terminals` sorts a component's terminals by the
partner-component signature of each terminal's net, identically in prediction
and ground truth, so a swapped collector/emitter is invisible. Measured on the
shipped corpus, only 15 components have a tied signature and all 15 carry the
same net on both terminals, so no pin swap here would move any published
number. It still matters, because `netlist.py` emits `Q<c> <b> <e>`,
`M<d> <g> <s>` and `E<out> 0 <in+> <in->` straight off raw terminal index: a
circuit can score 1.000 and simulate wrongly. That gap is a limitation of the
metric cascade (C4), not of the annotation.

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

These are in the per-file `notes` too, but they are the ones where the ink is
hard to read, where it is self-contradictory, or where reading it faithfully
produces a circuit that cannot be simulated:

- **`circuit_513`, site (423,722) — recorded as drawn, and un-simulable as a
  result.** The ink is unambiguous: a plain T where the y≈722 rail's left tip
  lands on column 3's continuous lower vertical, which runs unbroken to the
  bottom rail. Read literally that puts both terminals of the 15 V source on
  net `"0"`, so the sheet has 6 nets, the ERC reports a short-circuited voltage
  source, and **the circuit is un-simulable as drawn**. That is what the GT now
  records. Ground truth states the topology as drawn; the ERC error is the
  correct output and is explained rather than suppressed.
  *Electrical observation, recorded as evidence and not applied as a
  correction*: across the 190 human-verified annotations of the other split
  there are three components with both terminals on one net and **all three are
  current sources; a shorted voltage source occurs 0/190 times**, so the drafter
  probably did not intend the rail tip to land on the column. An earlier
  revision of this file severed the rail on that reasoning; it has been
  reverted. Electrically-motivated corrections do not belong in the topology —
  if the corrected variant is wanted it belongs in a separate ledger keyed on
  this site.
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
  three of them). `circuit_513` #8 is a third such case, alongside the
  short-circuited voltage source above.

## 5. Automated self-consistency re-derivation

**What this is, stated plainly.** The re-derivation below was performed by an
**AI assistant, not by an independent human annotator**. It is not a second
reader, not an expert validation, and not an inter-annotator agreement study.
Calling it any of those would misrepresent it, so this report does not.

It is an **automated self-consistency re-derivation**: each sampled file was
re-derived from the drawing without reading the first pass's reasoning first,
and the two netlists were compared.

- **What it can establish.** That the shipped files are internally consistent
  and free of clerical error — formatting faults, inconsistent terminal counts,
  duplicate or mistyped node names, one-terminal nets, terminals left unassigned,
  a netlist that does not match its own decision record. On the flagged subset it
  did exactly that: 7 of 9 files changed. That is a real check and it caught real
  mistakes.
- **What it cannot establish.** That a *judgement* is correct. It shares the
  first pass's reasoning, conventions and blind spots, so where the first pass
  was systematically wrong — a drafter idiom read the wrong way, a bare crossing
  decided on circuit sense — the re-derivation is disposed to reach the same
  wrong answer and to agree. A zero-disagreement row below is therefore evidence
  of consistency, **not** evidence of correctness.

**Genuine independent second annotation by an external annotator is PENDING.**
No such review has taken place. Nothing in this report should be read as
claiming one.

| sample | files | 3-terminal parts | disagreements |
| --- | --- | --- | --- |
| random, re-derived without the notes | 12 | 0 | 0 |
| stratified on sheets with 3-terminal parts | 5 | 38 | 0 |
| flagged by ERC or sibling cross-checks | 9 | — | 7 changed |
| independent human second annotation | — | — | pending |

The random sample contained no 3-terminal components, which is why the second,
stratified sample exists: pin order is the failure mode a net check cannot see,
and 38 components were re-read from the arrowheads and glyphs with zero
disagreements — consistently, which as above is weaker than correctly.

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
disputed — and they are exactly the class of call an automated re-derivation
cannot adjudicate, because it reasons the same way the first pass did. They
need a human second reader, which is pending.

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
- The tracer needs `scikit-image` (`skeletonize`), which is not in the project's
  runtime dependencies — it was an annotation-time tool, not a pipeline stage.
  `pip install scikit-image` to re-run it.
- 2026-08-04: `trace._nbcount` was rewritten from `cv2.filter2D` to numpy
  shifts. On some OpenCV builds (opencv-python 4.10.0.84 / darwin-arm64)
  filter2D returned halved neighbour counts at 1024×1024, so no pixel reached
  degree 3, every intersection site disappeared, and every decision record
  reconstructed into a single net — silently, with no error. Reconstruction was
  therefore unverifiable outside the machine the annotation was done on. Same
  result wherever filter2D was already correct; 16 files spanning both tracer
  versions were re-derived after the change and matched their shipped netlists
  exactly.

## 7. To adopt

```bash
python scripts/annotate_topology.py --check --gt-dir data/gt_test_1024
# 192 GT file(s): 192 verified, 0 unverified, 0 with validation issues
```

Already run, using the project's own `schematic2netlist.gt.validate_gt` with
the class whitelist enabled and `strict=True` (which every file now gets
anyway, since all 192 carry `"verified": true` after the 2026-08-03 sign-off):
every terminal has a net or is marked unconnected, terminal counts match the
class, GND sits on `"0"`, no net touches a single terminal. Result: **0 of 192
files have a validation issue.**

The electrical rule check is a *separate* pass (`scripts/gt_val_tools/erc.py`)
and it is not clean, by design: `circuit_513` reports a short-circuited voltage
source and `circuit_220`, `circuit_513` and `circuit_1207` report a current
source with both terminals on one net. All four findings are properties of the
drawings, are explained in §4 and in the per-file notes, and must not be
suppressed.

```bash
python scripts/gt_val_tools/erc.py data/gt_test_1024
```
