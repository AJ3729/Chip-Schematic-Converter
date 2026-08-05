# CGHD blind evaluation set — readiness

**Status: PREPARED, NOT ANNOTATED, NOT EVALUATED.**

Read this before touching anything else in this directory.

---

## 0. What this is, in one paragraph

A reviewer asked for a smaller **blind** evaluation set built from **new
drawings**, annotated independently and scored exactly once after the pipeline
is frozen. Sourcing genuinely new hand-drawn circuits was not possible in the
time available. What is prepared here is the closest available substitute: a
**drafter-disjoint, cross-dataset** set of 36 circuits from **CGHD** (Zenodo
record [10056817](https://zenodo.org/records/10056817), CC BY 4.0), a corpus
this project has never trained on, never tuned against, and never selected a
parameter from.

**This is a CROSS-DATASET test. It is HARDER than the blind set the reviewer
asked for, and it is NOT a substitute for one.** A same-distribution blind set
isolates one thing — whether the reported numbers survive contact with data
that had no hand in producing them. This set changes the corpus, the drafters,
the paper, the pens, the photography and the symbol conventions all at once, so
it answers a different and broader question, and it answers it with a
confounded, pessimistic bias. Section 5 states precisely what a reader may and
may not conclude from whatever number comes out.

---

## 1. What exists now

Built by `scripts/prepare_cghd_blind.py` (seed 0, deterministic — reruns
reproduce `manifest.json` byte-for-byte). The frames live under `data/`, which
is gitignored; regenerate them from the CGHD archive with

```
python scripts/prepare_cghd_blind.py          # full run
python scripts/prepare_cghd_blind.py --dry-run  # selection report only
```

| | |
| --- | --- |
| circuits | **36** |
| distinct CGHD drafters | **18** of 25 |
| components | 490 |
| terminals | 1,021 |
| three-terminal parts (pin order is scored) | 99 |
| CGHD junction marks available as evidence | 611 |
| CGHD crossover marks available as evidence | 37 |
| **terminals with a net assigned** | **0** |
| frame | 1024, produced by this project's own preprocessing path |
| frame guard (annotated boxes inside the canvas) | 0 of 1,790 outside — **PASS** |

```
data/cghd_blind_1024/images/<stem>.jpg      the 1024 frames  (gitignored, regenerable)
results/cghd_blind/manifest.json / .csv     stem, drafter, circuit, component count, sha256
results/cghd_blind/selection.json           seed, rules, rejects, achieved distribution, reserve
results/cghd_blind/run_meta.json            config + git SHA + seed + environment
results/cghd_blind/packet/gt/*.json         36 GT stubs — every net null
results/cghd_blind/packet/decisions/*.json  36 empty decision records
results/cghd_blind/packet/aux/*.json        CGHD's junction/crossover/text/terminal boxes,
                                            plus the 130 components needing a human class call
results/cghd_blind/packet/README.md         how to annotate this packet
```

### 1.1 Nothing here came from the pipeline

No detector, wire tracer, or pipeline stage was executed while building this
set. Ground truth seeded by the system's own predictions would make the
subsequent evaluation circular and would destroy the point of the exercise.

What the component inventory *does* come from is CGHD's own published
Pascal-VOC annotations — human annotations by the dataset authors. That is
legitimate and it is exactly the arrangement used for the Digitize-HCD test
split, whose GT files record `source: coco_geometry+manual_topology`. These
record `source: cghd_geometry+PENDING_manual_topology` and carry
`"pipeline_output_used": false`.

The `sites` object in every decisions file is **empty on purpose**. Site ids
are enumerated by the tracer at annotation time; pre-filling them here would
have meant running the system under test and handing the annotator a default
to accept.

### 1.2 How the 36 were chosen

Starting pool: 2,424 CGHD images that carry an annotation, across 25 drafters
and 293 circuits. Eligibility, applied in this order:

| rule | why | images rejected |
| --- | --- | --- |
| no circuit used in `results/cghd_zero_shot/` | that measurement has already been made and reported; blindness should be unqualified. Excluded at **circuit** level, not merely image level | 795 |
| every electrical symbol must map onto one of the 17 Digitize-HCD classes | a component the vocabulary cannot express makes the netlist unscoreable | 1,133 |
| no `resistor.adjustable` | a potentiometer has three terminals and our `Resistor` has two, so no valid GT file can be written | 88 |
| 5 ≤ components ≤ 45 | the Digitize-HCD test range is 5–30; the cap is 1.5× that, which keeps drafters whose only usable sheet is large | 88 |

That leaves **320 eligible images → 40 eligible circuits across 18 drafters**,
from which 36 were selected: one template per drafter first (drafter coverage
is the point of the exercise), then stratum-deficit fill. CGHD draws each
circuit twice and photographs each drawing four times, so **at most one image
per (drafter, circuit)** is used and the set contains no near-duplicate
topologies. The 4 unused circuits became a documented ordered **reserve**; one
of them (`drafter_19` C221) has already been consumed to replace a circuit with
no legible photograph (§1.5), leaving **3 in reserve**
(`selection.json → reserve_circuits` / `reserve_circuits_consumed`).

### 1.3 Component-count distribution vs the Digitize-HCD test split

| | blind (36) | Digitize-HCD test (192) |
| --- | --- | --- |
| min / median / max | 5 / 11 / 43 | 5 / 13 / 30 |
| mean | 13.6 | 13.4 |
| quartiles | 7 / 11 / 16 | 7 / 13 / 19 |
| ≤7 components | 10 (27.8%) | 54 (28.1%) |
| 8–13 | 14 (38.9%) | 46 (24.0%) |
| 14–19 | 5 (13.9%) | 52 (27.1%) |
| ≥20 | 7 (19.4%) | 40 (20.8%) |

Two-sample Kolmogorov–Smirnov **D = 0.177** against a critical value of 0.247
at α = 0.05: the two component-count distributions are **not distinguishable**
at the 5% level.

The 14–19 band is nevertheless visibly light, and that is not a sampler
failure — **three of the four strata were taken in full**. CGHD has only five
eligible circuits in that band, so no selection of this size matches more
closely. This is a property of CGHD's circuits after the vocabulary filter,
and it is worth saying plainly that the filter is what causes it: requiring
every symbol to be one of our 17 classes throws out CGHD's integrated
circuits, logic gates, switches, relays and transformers, which are
disproportionately the mid-to-large sheets.

Note also that matching the *component-count* distribution says nothing about
matching the *drawing* distribution, which is unambiguously different.

### 1.4 Class coverage — six of seventeen classes have no support

Present: Resistor 175, Capacitor 78, GND 58, BJT-NPN 57, Diode 38, MOSFET-N 29,
Op-Amp 13, V-DC 13, Zener Diode 11, Inductor 11, V-AC 7.

Absent, and why:

- **I-DC, I-AC, V-DC (one port)** — no CGHD counterpart exists at all
  (`data/cghd/class_mapping.yaml → missing_targets`). These three cannot be
  measured on CGHD by any selection.
- **BJT-PNP, MOSFET-P** — CGHD does not split NPN/PNP or N/P. They can only
  appear once the annotator makes the class calls in §2.
- **Wire Crossover** — not a GT component by this project's convention
  (`gt_test_1024` contains zero of them); crossovers are shipped as evidence in
  `aux/` instead.

A per-class claim about those six cannot be made from this set.

### 1.5 The frame legibility gate — and why it makes the set easier

CGHD photographs are taken on ruled notebook paper, on cluttered desks, and
under hard shadows. The preprocessing frontend, tuned on Digitize-HCD photos,
degrades badly on them: the first draw produced frames that were 74% black
(complete binarisation blow-out), frames where a shadow formed one connected
blob covering an eighth of the canvas, and frames with a keyboard in shot.

A human cannot annotate those either, so a **mechanical, model-free gate** was
applied: a shipped frame must fall inside the ink-statistics envelope of the
Digitize-HCD **train** frames — ink fraction ≤ 0.0470 and largest connected
black component ≤ 0.0332 of the canvas, both calibrated on 895 training frames
so nothing about either evaluation split enters the threshold. Where the drawn
photograph failed, a **sibling shot of the same circuit** was used (CGHD
photographs each drawing four times; the circuit, drafter and inventory are
unchanged). Where no shot passed, the circuit was replaced from the reserve.

Outcome: **7 photograph swaps, and 1 circuit dropped outright** — `drafter_2`
C19, all eight photographs of which fail binarisation; `drafter_19` C221 took
its place.

**This must be stated in any write-up.** The gate screens photographs, never
pipeline output, so it cannot leak information about the system under test —
but it does mean the set is *conditioned on the preprocessing frontend
succeeding*, and therefore **understates** CGHD's true end-to-end difficulty.
Roughly one CGHD circuit in five needed a different photograph or a
replacement to produce a legible frame at all; that figure is itself a result
about frontend robustness and should be reported next to any score.

Even after the gate the frames are harder than the numbers suggest. Ruled
notebook paper survives binarisation as a field of parallel lines that a wire
tracer cannot distinguish from wires (`cghd_d23_C271_D2_P2`,
`cghd_d19_C225_D1_P3` are the clearest cases), and photographed page edges
widen the annotation-aware crop so the drawing occupies less of the canvas —
thinner strokes at the same nominal resolution.

---

## 2. What a human must still do

### 2.1 The work

1. **Stage the annotation workspace.** The tools in `scripts/gt_val_tools/`
   hardcode a sandbox layout, so they need:

   ```
   /home/claude/tools/          <- scripts/gt_val_tools/*.py
   /home/claude/val/img1024/    <- data/cghd_blind_1024/images/*.jpg
   /home/claude/val/gt/         <- results/cghd_blind/packet/gt/*.json
   /home/claude/val/test.txt    <- one "<stem>.jpg" per line
   /home/claude/dec/            <- results/cghd_blind/packet/decisions/*.json
   /home/claude/out/gt/         <- finalize.py writes finished GT here
   ```

2. **Build the review packets** — `python3 /home/claude/tools/batch.py
   /home/claude/val /home/claude/pkg`. This runs the wire tracer to *propose*
   intersection sites; running it **now** is correct and is what the
   Digitize-HCD pass did. It was deliberately not run during preparation so
   that nothing in the committed packet originates from the system under test.
   **This step is untested on CGHD frames** — it is the first thing to
   smoke-test, and ruled-paper sheets may produce very large site counts.

3. **Annotate**, following `docs/ANNOTATION_GUIDE.md` — same guide, same rules,
   same schema as the Digitize-HCD test split, so the two sets stay
   comparable.

   > **Use `docs/ANNOTATION_GUIDE.md`, not `scripts/gt_val_tools/BRIEF.md`, for
   > the rules.** The older brief tells the annotator that "electrical
   > impossibility wins" at an ambiguous crossing; that rule has been withdrawn
   > in favour of *annotate the topology as drawn*. Following the withdrawn
   > rule would produce ground truth that silently repairs drawings, which
   > marks a correct reading wrong. `BRIEF.md` remains the reference for the
   > tooling commands only; the decisions-file schema is identical in both.

4. **Three things are extra work relative to the Digitize-HCD pass:**
   - **130 components on 31 sheets need a class call.** Every `transistor.bjt`
     arrives as `BJT-NPN` and every `transistor.fet` as `MOSFET-N`, because
     CGHD does not distinguish the subtypes. Read the arrow and set the real
     class. `aux/<stem>.json` lists the exact component ids.
   - **`vss` is not necessarily ground.** The published mapping sends it to
     `GND`, which would force it onto net `"0"`. If the drawing means a supply
     rail, change the class to `V-DC (one port)`.
   - **Ruled paper.** Expect more spurious intersection sites than on
     Digitize-HCD, and expect to reject more of them.

5. **One thing is less work:** CGHD annotates junctions and crossovers itself —
   611 and 37 marks respectively, shipped in `aux/`. These are the dataset
   authors' human reading of the drawing and are strong evidence for a site
   call. They mark a *location*, not a partition of wires, so the call is still
   the annotator's.

6. **Record unusable sheets rather than quietly dropping them.** If a frame
   turns out to be illegible despite passing the gate, or the drawing is
   genuinely ambiguous, write it down with a reason and replace it from the
   reserve **before the freeze**. Replacement decided after seeing pipeline
   output is disqualifying.

### 2.2 How long

**Estimate, not a measurement.** The Digitize-HCD pass adjudicated a mean of
6.8 critical sites per image and inspected every component crop. At a careful
25–45 minutes per sheet — the wide end because these are unfamiliar drawings
with extra class calls and noisier ink — 36 sheets is **roughly 15–27 hours of
annotator time**, call it 3–5 working days including workspace setup, the ERC
pass, the notes, and a re-read of anything the checker flags.

### 2.3 Independence

The reviewer's request implies an annotator who is not the person who built the
pipeline. Nothing in this packet enforces that, and the Digitize-HCD ground
truth carries a standing caveat that its second reading was an automated
self-consistency re-derivation rather than an independent human annotator. If
this set is annotated by the same person, it inherits the same caveat and the
write-up must say so.

---

## 3. Expected difficulty

**Zero-shot detection on CGHD is already measured**
(`results/cghd_zero_shot/summary.json`): macro AP@0.5 = **0.185** over 100
images and 25 drafters, against **0.975** in domain. Per class it ranges from
Op-Amp 0.555 down to Wire Crossover 0.016, Zener 0.032, Inductor 0.047.

The consequence for an end-to-end run is not subtle. Strict success requires
every component found and classified *and* every net correct. At detection
performance in that range, a 5-component sheet is unlikely to survive intact
and a 20-component sheet will not. **A strict-success number near zero is the
expected outcome, and on its own it would be almost uninformative** — it would
measure the detector's domain gap, which is already known, and say nothing
about the topology reasoning that is the contribution.

### 3.1 Therefore: pre-register two numbers, not one

Before the freeze, commit to reporting **both**:

- **End-to-end (mode A)** — the honest cross-dataset number. Expected to be
  poor.
- **Detection-oracle (mode B, `scripts/oracle.py`)** — GT boxes and classes
  injected, everything downstream predicted. This is the number that actually
  tests whether wire extraction, snapping and net reconstruction transfer to
  another corpus, with the known detector gap held out.

The delta between them is the attribution, and it is the only way this
experiment says anything the zero-shot detection run did not already say.

In-domain references to compare against (test split, seed 0):
terminal-pair F1 0.808, net F1 0.877, NGED 0.182, strict success 0.516.

### 3.2 Statistical power

36 images is small. A proportion measured on 36 sheets carries a 95% CI of
about **±16 points at p = 0.5** and **±10 points at p = 0.1**. This set can
detect a collapse. It cannot resolve a 5-point difference, and it must never be
used to rank two configurations.

---

## 4. The protocol: freeze, then evaluate once

Follow this literally. Its whole value is that it is agreed *before* any number
exists.

**Phase 1 — annotate (pipeline untouched).**
Annotate all 36 sheets to the standard in `BRIEF.md`. Do not run the benchmark,
the detector, or the pipeline on any blind image for any reason during this
phase. Do not look at any pipeline output on these images.

**Phase 2 — freeze.**
Commit the finished GT. Then commit a freeze record containing: the git SHA,
the `configs/default.yaml` hash, the detector weight file hash, the exact
command lines to be run, and the metrics to be reported — including the mode A
/ mode B pair from §3.1. Nothing after this point may change any parameter,
threshold, weight file, or metric definition. If something must change, the
blind set is spent; say so and stop.

**Phase 3 — evaluate once.**
Run the pre-registered commands. Write the result to
`results/cghd_blind/eval/` alongside the freeze record.

**Phase 4 — report.**
Report the number obtained. Not the best of several. Not after a fix.

### 4.1 Do not inspect per-image failures before submission

**This is the rule most likely to be broken, and breaking it silently voids the
whole exercise.**

After the single evaluation run, do **not** open the failing images, do not
render overlays of the failures, do not diff predicted against ground-truth
nets sheet by sheet. Report the aggregate.

The reason is mechanical, not ceremonial: looking at which sheets failed and
why is the first step of tuning, whether or not a parameter is subsequently
changed. Once the failure modes are known, every later decision is informed by
this set, and the second evaluation on it is no longer blind. A per-image error
analysis is a legitimate and valuable thing to do — **afterwards**, in a
clearly separated section, on the understanding that the blind set is now
spent and any number computed on it after that point is not a blind number.

If a run crashes or produces obviously corrupt output for infrastructural
reasons (missing file, unreadable image), fix the infrastructure, record what
was fixed in the freeze record, and rerun. Do not use that door for anything
else.

---

## 5. What a reader should and should not conclude

### 5.1 If the number is low

**A low score on this set is a DOMAIN-SHIFT result, not a pipeline failure, and
the write-up must not conflate the two.** Three effects are confounded here and
cannot be separated by this experiment alone:

1. the detector's 0.185 → 0.975 domain gap, already measured;
2. the preprocessing frontend's degradation on ruled paper, desk backgrounds
   and uneven lighting — which §1.5 shows is severe enough that roughly one
   circuit in five needed a different photograph to be usable at all;
3. whatever the topology stage itself does or does not do on unfamiliar
   drawings.

Only (3) is the contribution under test. The mode A / mode B split in §3.1 is
what separates (1) from (3); nothing here separates (2), and no claim should be
made that pretends otherwise.

### 5.2 If the number is high

That would be a strong result — stronger than the reviewer asked for, because
it would clear a harder bar. It should still be reported with the sample size
and its confidence interval attached, and with the note that six of the
seventeen classes have no support in this set (§1.4).

### 5.3 What this set establishes

- The topology stage was measured on drawings from a corpus that had no hand in
  building it: not in training, not in parameter selection, not in early
  stopping, not in any sweep.
- The set is **drafter-disjoint by construction** across **18 distinct
  drafters**. This directly addresses a standing limitation — Digitize-HCD
  ships no drafter metadata, so a drafter-disjoint split is impossible there
  and the existing splits are stratified rather than drafter-held-out. On this
  set, a per-drafter breakdown of the result is meaningful.
- The result is untainted by the two known contaminations of the Digitize-HCD
  test split: the detector was early-stopped on it, and 50 of its images appear
  in sweeps that returned inert.

### 5.4 What it does NOT establish

- **It is not the blind set that was asked for.** It does not answer "do the
  reported numbers hold on new drawings from the same population?" That
  question remains open and can only be closed by new drawings.
- **It cannot separate a topology failure from a photography failure.** See
  §5.1.
- **It is conditioned on legible frames** and therefore understates CGHD's real
  difficulty (§1.5).
- **It says nothing about six of the seventeen classes** (§1.4), three of which
  no CGHD selection could ever cover.
- **It cannot rank configurations.** At n = 36 the confidence intervals are too
  wide, and in any case it may be evaluated only once.
- **It carries the same annotator-independence caveat** as the Digitize-HCD
  ground truth unless a genuinely independent annotator does the work (§2.3).

### 5.5 The honest bottom line

This substitutes a *harder, differently-shaped* test for the one requested. It
is worth running: it is the strongest evidence obtainable in the time
available, it addresses the drafter-disjointness limitation outright, and its
result — whichever way it goes — is informative provided the mode A / mode B
pair is reported together. But a reviewer who specifically wants to know
whether the headline numbers survive on fresh same-distribution drawings will
not have that answer from this, and the write-up should concede the point
rather than let this set be read as closing it. **New drawings remain
necessary.**

---

## 6. Relationship to `results/blind_review/`

There is a **separate and different** effort in the working tree:
`results/blind_review/` is an independent *second annotation* of 58
**Digitize-HCD** test images, aimed at the inter-annotator-agreement gap. It
re-annotates images the project already has ground truth for.

This set does the opposite: **new circuits, no ground truth yet, a different
corpus**. The two are complementary and neither replaces the other —
`blind_review` measures whether the existing ground truth is right;
`cghd_blind` measures whether the pipeline transfers. Do not report them as one
result, and do not let "blind" in both names blur them together.

---

## 7. Provenance and licensing

CGHD / GTDB-HD, Zenodo record 10056817, `cghd-zenodo-12.zip`, **CC BY 4.0** —
attribution required in any publication that reports a number from this set.
Per-image `source_sha256` in `manifest.json` identifies the exact source
photograph. The class mapping onto the 17 Digitize-HCD categories is
`data/cghd/class_mapping.yaml`, and its lossy entries are the ones §2.1 asks
the annotator to confirm.

Known gap, not fixed here because `data/cghd/` is read-only for this task: six
class names present in the shipped annotations —
`capacitor.adjustable`, `inductor.ferrite`, `magnetic`, `mechanical`,
`optical`, `probe` — appear in neither the `mapped` nor the `unmapped` list of
`class_mapping.yaml`. They are handled correctly by both the zero-shot script
and the selection here (treated as unmapped, so any sheet containing one is
ineligible), but the YAML should be updated to list them explicitly.
