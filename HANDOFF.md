# HANDOFF — Chip-Schematic-Converter / schematic2netlist

**For the next Claude Code session.** Read this first, then the plan
(`~/Downloads/IEEE_ACCESS_PLAN_v2_with_repair.md`). Last updated
2026-07-29 (long session: Week-2/3 completion, two major pipeline
corrections, resolution experiment).

## 0. LATEST — 2026-07-30 overnight session

Read 0.1 first; it invalidates numbers, not just conclusions.

### 0.1 nGED WAS NOT REPRODUCIBLE. Every pre-2026-07-30 nGED number is void.

`nx.graph_edit_distance(..., timeout=t)` does not report that it gave up. On
timeout it returns the best bound found SO FAR as a plain float, so the
`if ged is not None` check could not distinguish that from an exact answer and
the documented fallback branch was unreachable. What the benchmark printed as
GED was a function of how much CPU the process happened to get. Same graphs, at
1 s / 3 s / 6 s budgets: **0.4850 / 0.4611 / 0.4371**. At the shipped 5 s
budget, 13 of 18 test images burned the entire budget and all returned a number.

It had already produced a false positive. Turning `stitch_masked_gaps` off
changes the recovered topology on **zero** images — the pipeline output is
identical pixel for pixel and every other per-image column has CI [0,0] — yet
nGED "significantly" improved on 24 of 190 images.

Replaced with Riesen–Bunke bipartite assignment: Hungarian proposes a node
mapping, then the EXACT cost of that concrete mapping is returned, so it is a
genuine upper bound. Deterministic to 12 decimals across repeated calls and
shuffled component order; valid upper bound in 12/12 exactly-solvable pairs.
Absolute values rise (0.1921 → 0.2439 on the same baseline) because a
deterministic bound is looser than one silently given more compute.

**Side effect worth knowing: a 190-image benchmark went from ~35 min to ~75 s,
about 25x.** The GED search was nearly all the runtime. Sweeping a parameter on
the real benchmark is now cheaper than the weld/split proxy was.

### 0.2 Two real bugs in `_bridge_collinear`, one of which does not matter

**Welding parallel rails.** The docstring claimed the two anisotropic closings
leave "the perpendicular direction untouched". Backwards: a HORIZONTAL closing
bridges horizontal gaps between *any* ink, and the horizontal gap between two
side-by-side VERTICAL strokes is exactly such a gap. Two rails 10 px apart fuse
into one solid 17-px band at `bridge_span: 18`; at 9 they do not.

**Kernel parity.** A closing is extensive only for a structuring element
symmetric about its anchor. OpenCV anchors an even-length kernel off-centre, so
dilate and erode stop being adjoint and the closing DELETES ink — a lone pixel
is annihilated. Every even span destroys 0.87–2.40% of all wire ink, every odd
span exactly none. `bridge_span: 18` was destroying 1.44%.

`_line_kernel` now rounds up to odd; `wires.bridge_odd_kernel: false` reproduces
the old behaviour so the ablation keeps an arm.

### 0.3 The decomposition: span is the whole effect, parity is a null

All on the deterministic metric, paired over 190:

| change | tp_f1 | net F1 | verdict |
|---|---|---|---|
| parity only, even 18 → odd 19 | −0.0017 | −0.0013 | **null** |
| span only, odd 19 → odd 9 | **+0.0232** | **+0.0216** | **SIG** |
| combined, even 18 → odd 9 | +0.0215 | +0.0203 | SIG |

**Do not credit the parity fix.** I initially wrote that bs9 "mixes the span
change with the kernel-parity bug"; isolated, parity moves nothing and span
alone reproduces the whole effect. Deleting 1.44% of wire ink is verifiably real
and metrically inert. Keep the fix (free, removes silent damage), do not sell it.

strict success is flat throughout (2 win / 2 lose). The gain is in terminal-pair
and net F1 — fewer welds and splits, not yet whole images flipping.

### 0.4 `bridge_mode: guarded` — the closing minus its welds

Pure orientation gating (`bridge_mode: directional`) removes the welds but also
removes bridging the closing was rightly doing: the legacy closing adds 55% more
ink than the frame contains, the gated version 7.6%, and the split rate rose
0.12. The closing does two jobs — bridging dash gaps AND gluing wobbly strokes
— and gating discards the second with the first.

`guarded` keeps every added pixel and subtracts only welds (44.8% added ink,
removing 18.7% of the fill). The guard is phrased as a condition for KEEPING a
fill: legitimate when ink running along the same direction is within reach on
either side. Rejecting on "perpendicular ink on both sides" was tried first and
is too narrow twice over — it cannot see a weld between strokes neither parallel
nor perpendicular to the pass, which is how adding the diagonal passes re-welded
the very rails they were meant to spare.

### 0.5 Class confusion: the errors are systematic, and text cannot fix them

- **Seed class vote**: strict +0.0105, 2 win / 0 lose, CI touching zero. Free,
  monotone, directionally positive everywhere. Adopt but do not claim.
- **Polarity-preserving TTA** (`scripts/detect_tta.py`): net +7 at 88.9%
  precision, but only **2 of 67** errors are TTA-only — it is redundant with the
  seed vote. Note the domain constraint: ultralytics `augment=True` includes a
  HORIZONTAL FLIP, and MOSFET-N/P and BJT-NPN/PNP are distinguished by a
  mirrored arrow, so a flip *is* the other class.
- **38 of 67 errors survive both**, so they are systematic, not variance.
- **Value labels: NEGATIVE.** `text_annotations.json` gives 11,936 transcribed
  values and the unit IS the class (Ω→R, F→C, H→L, V/A→source, sin/cos→AC),
  verified on every real string form. It still nets **−14 labels at 29.4%
  precision**, because labels are not associable with their own component:
  in circuit_267 the capacitor's nearest label is the inductor's `60H` at 0.76
  box diagonals while its own `1F` sits at 1.70. Hungarian does not help — the
  WRONG pairing has lower total cost (121 vs 303 px). A count constraint
  (multiset of units vs class counts) matches exactly in only 71.6% of images.
  This also blocks emitting real values instead of placeholders: same
  association.

### 0.6 Positive, cheap, and composable

**Constraint-triggered repair** (`scripts/apply_constraint_repairs.py`, all 190):
strict **+0.0105, 2 gained / 0 lost**, tp_f1 +0.0082, percomp +0.0156, from 34
one-terminal-net bridges and 2 self-short body erasures. Purely
circuit-theoretic — no image evidence, no training.

**Why composition matters:** the seed vote corrected labels in 23 images but
gained strict in only 2. Strict is a product over components, so a corrected
label converts only where connectivity is ALREADY perfect. Class work and wire
work multiply rather than add.

### 0.7 The Tier-1 stitcher is currently INERT

`stitch_masked_gaps: false` changes every topology metric by exactly 0.0000
(CI [0,0]) and the pipeline output is identical pixel for pixel. With
`component_mask_pad: 0` the stitchable region is only the text mask, and the
pad-ring holes the stitcher was built to bridge no longer exist. It solved a
self-inflicted problem that was later fixed at the source. The nGED difference
that made it look useful was the metric bug in 0.1.

### 0.8b WHY STRICT SUCCESS IS WHERE IT IS — and what is actually left

**Strict success is a function of circuit size, and not merely because it is a
product over components.**

| components | images | strict | tp F1 | GT nets clean |
|---|---|---|---|---|
| ≤8 | 72 | **0.7639** | 0.9131 | 87.0% |
| 9–12 | 19 | 0.5263 | 0.8887 | 82.8% |
| 13–16 | 37 | 0.0811 | 0.6331 | **49.0%** |
| 17–20 | 23 | 0.1739 | 0.7617 | 71.4% |
| 21+ | 39 | 0.1795 | 0.5677 | 60.1% |

Correlation with component count is −0.51 on BOTH strict and terminal-pair F1.
The second one is the point: tp F1 is a per-pair metric, not a product, and it
still falls 0.91 → 0.57. Per-net cleanliness falls 87% → 49–60%. Large circuits
are genuinely harder, not just penalised harder.

**The gate.** Of 111 failures: 38 are blocked by detection (`unmatched_gt` > 0,
and 26 of those by exactly ONE component), 73 by connectivity alone. Perfect
connectivity would give **0.80**; perfect detection only ~0.44. Connectivity is
the binding constraint. And the connectivity failures are not near-misses — only
8 of 73 are within 0.10 of perfect, while **51 are below 0.80 tp F1** with 3–7
defects each. There is no patch that flips them.

**What was ruled OUT, each with a measurement rather than an argument:**

| hypothesis | test | result |
|---|---|---|
| dense circuits have tighter spacing → raise resolution | inter-stroke gap distribution by size bucket | **identical** (median 15.28 px, p05 1.91 px in every bucket), correlation with tp F1 −0.029. Refuted; 2048 would not help, which is consistent with the earlier null. |
| missing crossing information | 3 independent oracles — GT crossover boxes, GT-guided arm cutting, GT-guided splits at arbitrary sites | all null or negative; see 0.8c |
| component ink leaking past its box | `component_mask_pad` sweep | monotonically harmful, 0 is best |
| morphology manufactures the welds | bridge_span sweep split into precision/recall | precision does NOT recover as bridging is removed (0.7419 at span 3 vs 0.7497 at 7) |

**So the welds are in the raw ink**: wires that genuinely touch where the drawing
crosses them without a hop symbol. No local operation on the extracted graph
fixes that, and three GT-guided oracles confirm it.

### 0.8c THE STRATEGIC CONCLUSION

The morphological pipeline is at its ceiling. Every remaining lever has now been
priced:

- **class labels** +0.0263 strict, significant — needs better *training*, not
  more inference tricks (38 of 67 errors survive both the seed vote and TTA, and
  value labels are refuted).
- **per-site crossing decisions** +0.0333 strict — but only 8 of 1245 sites should
  be split, so a classifier needs ~99.4% specificity against a measured 0.659 AUC.
  Not reachable.
- **everything else in wires** — no local operation helps, with or without GT.

What that leaves is a change of paradigm rather than another parameter: the net
assignment should be **learned directly** rather than derived from morphology and
connected components — a model that labels wire pixels (or skeleton nodes) with
net identity, trained on the 895-image train split, so that "do these two strokes
belong to the same net" is answered by context and circuit structure instead of
by whether their ink happens to touch. Mode C shows the target is reachable in
principle (tp F1 0.9967 once connectivity is correct); nothing in the current
extractor can get there.

### 0.9 WHERE THE PIPELINE LANDED, and what to do next

Two changes shipped. Everything else this session was either a negative or
hygiene.

| stage | net F1 | tp F1 | strict |
|---|---|---|---|
| v1 classical canny | 0.4618 | 0.2170 | 0.0000 |
| v5 + ports snap, pad 0 (previous default) | 0.8075 | 0.7059 | 0.3579 |
| **v6 + bridge_span 7** | 0.8334 | 0.7321 | 0.3632 |
| **v7 + connectivity repair** | 0.8411 | 0.7426 | 0.3789 |
| **v8 + snapping max_expand 80 (new default)** | **0.8436** | **0.7458** | **0.3789** |

Three seeds under the new default: tp F1 0.7458 / 0.7494 / 0.7390, strict
0.3789 / 0.3579 / 0.3421.

Against the *previously shipped* default, paired over 190
(`results/comparisons/full_stack.csv`): tp F1 **+0.0351 SIG**, net F1 **+0.0323
SIG**, per-component **+0.0294 SIG**, strict **+0.0263** (6 win / 1 lose, CI
touches zero). 67 → 72 of 190 images fully correct. Three seeds under the new
default: strict 0.3789 / 0.3579 / 0.3368.

**The attribution moved, which is the real result:**

| stage | before | now |
|---|---|---|
| detection | 0.0627 | 0.0642 |
| wires | **0.3406** | **0.1892** |
| snapping | 0.0028 | 0.0033 |

Wires is still dominant but is now 2.9x detection where it was 5.4x — its share
of the terminal-pair error fell 44%.

**The next moves, in priority order.**

1. **Keep cross-validating the swept parameters — this is still the best
   expected value per hour.** Two are now done and both were off: `bridge_span`
   was 11 points off optimum with a *significant* gain unclaimed, and
   `snapping.max_expand` peaked at 80 rather than 60 (small, not significant,
   adopted). `connectivity_repair.max_gap` saturates at 60 and is confirmed.
   **Still unswept on the real benchmark:** `min_blob_area` (80),
   `min_blob_extent` (30), `textmask.*` (a whole block, and `dilate_kernel: 6`
   is even so it offsets the mask by half a pixel), `nodes.junction_site_box`,
   `preprocess.speck_min_area`/`speck_min_extent`, and
   `snapping.window_depth`/`ground_max_expand`. A 10-point sweep is ~13 minutes
   with `scripts/cv_select_param.py` for the honest number.
2. **Class confusion is the only detection headroom.** The GT-class oracle is
   **+0.0211 strict (4/0), SIG on four metrics** on the improved pipeline. The
   seed vote captures part of it (+0.0105, 2/0, free). TTA is redundant with the
   vote (2 of 67 errors are TTA-only) and value labels are refuted (0.5). What is
   left is 38 systematic errors that need better *training* — more data, or a
   loss that penalises the near-symmetric pairs — not more inference tricks.
3. **Crossings are closed.** Do not reopen. Six approaches, two oracles, and now
   a *significant negative* on per-component accuracy from perfect crossover
   boxes.
4. **`[HUMAN]`: annotate net topology for the val split.** Every parameter in
   this pipeline is tuned on the split it is reported on, because net-level GT
   exists only for test (see 0.10). Cross-validation is a mitigation, not a fix.
5. Regenerate `results/oracle_1024` etc. after any further default change — the
   `*_prefix_metric` copies show what a stale set looks like.

### 0.10a PARAMETER SWEEP COVERAGE — what is now settled, and what cannot be swept

Every value below was checked on the REAL benchmark (~75 s per run) and selected
by 10-fold cross-validation, not by reading a sweep peak.

**Changed (adopted):**

| parameter | was | now | effect |
|---|---|---|---|
| `wires.min_blob_area` | 80 | **10** | together **+0.0207 tp_f1, +0.0368 STRICT** |
| `wires.min_blob_extent` | 30 | **8** | (a 30-point grid; the filter was near-pure harm) |
| `wires.bridge_span` | 18 | **7** | +0.0245 tp_f1, +0.0246 net F1, both SIG |
| `snapping.max_expand` | 60 | **80** | +0.0031 tp_f1, not significant, stable in 10/10 folds |

**Confirmed at the shipped value** — swept and left alone, so do not re-sweep:
`detect.confidence` 0.4 (0.5 wins 9/10 folds on strict but delivers 0.4158
out-of-fold, identical to 0.4 — the edge is selection noise; and lowering it is
monotonically WORSE, 0.3737 at 0.1, so recovering missed components that way does
not work), `nodes.connectivity` 8, `connectivity_repair.max_gap` 60 (saturates —
60/90/120 are bit-identical), `connectivity_repair.body_frac` 0.5,
`connectivity_repair.dir_tol_deg` 40, `snapping.window_depth` 8,
`snapping.ground_max_expand` 80, `snapping.expand_step` 4,
`nodes.junction_site_box` 30, and the whole `textmask` block (`dilate_kernel`,
`adaptive_c`, `min_area` — all flat within 0.0002).

**Cannot be swept as things stand** (see 0.10b for the guards now in place):
every `preprocess.*` key, because frames are pre-generated and the benchmark
reads them rather than re-running preprocessing. Sweeping them requires
regenerating frames AND the detection cache per value; the speck experiment that
ignored this produced a bogus sharp optimum. The speck filter itself barely
changes the frame at all (mean level difference 7.09 at 40/12 versus 7.15 with
removal off entirely), so the expected effect is small.

### 0.10b THE MEASUREMENT AUDIT — six defects, three of them corrupting results

Asked to hunt measurement bugs, these are what turned up. The first two were
actively wrong-ing published numbers; the rest are traps that silently produce
false negatives.

| # | defect | status |
|---|---|---|
| 1 | **nGED not reproducible** — `nx.graph_edit_distance(timeout=)` returns its best-so-far bound as a plain float, so 72% of images reported a load-dependent number as exact (0.4850 / 0.4611 / 0.4371 at 1/3/6 s on the same graphs). It had already produced a significant result from a change that provably does nothing. | FIXED — deterministic Riesen–Bunke bound, and a 190-image benchmark went 35 min → 75 s |
| 2 | **per-component accuracy recall-only** — `gt_pairs <= pred_pairs`, so a circuit with every net welded into one scored **1.000**, identical to a perfect answer. Blind to welding, which is the dominant failure mode. | FIXED — exact variant is now primary, recall variant kept and both emitted |
| 3 | **`detect.confidence` unsweepable** — the cache's own floor is 0.4118 because the 0.4 threshold was baked in when it was written, so sweeping below 0.4 is a no-op and reports a flat curve that reads as "no effect". | GUARDED — `sweep_param.py` warns; low-confidence cache regenerated to make it explorable |
| 4 | **every `preprocess.*` sweep is a no-op** — frames are pre-generated, so the benchmark never re-runs preprocessing. All five speck arms scored identically for this reason. | GUARDED — `sweep_param.py` refuses `preprocess.*` outright |
| 5 | **frames/cache generation mismatch** — regenerating frames while keeping the committed detection cache costs **0.027 tp_f1 and 0.032 strict** and nothing complains; `audit_data_freshness.py` reports "ok" for both. This produced a bogus "the speck filter has a sharp optimum" result. | GUARDED — `scripts/check_cache_alignment.py`; ink coverage is useless (6.059 vs 6.069) but box-centre-to-ink-centroid CENTRING separates cleanly (2.795 / 2.799 aligned vs 3.633 mismatched) |
| 6 | **`detect_batch.py --images-dir` defaulted to the 512-era `data/cleaned`** and ignored `cfg["preprocess"]["images_dir"]`, so a cache generated without the flag was computed in a different coordinate frame — 0 of 2802 boxes within 1 px of the committed ones. | FIXED — default now follows the config. Caught by #5 twenty minutes after writing it. |

**Checked and CLEAN**, so nobody re-audits them: the paired bootstrap in
`compare_runs.py` (resamples per-image deltas, seeded, sorted order); pipeline
determinism (two identical runs, **0 differing cells** over 190 images × 18
columns — only true because of #1); `align_components` (Hungarian on 1−IoU with a
post-assignment threshold re-check that correctly rejects padded cells); and
vacuous strict success (**0** GT files have zero terminal pairs).

**A false alarm I raised and must not be repeated:** the committed frames do not
reproduce byte-for-byte from the current config, which looked like a
reproducibility failure of the whole pipeline. It is not. Re-running BOTH stages
from raw inputs gives tp_f1 0.7675 / strict 0.4105 against the reported 0.7669 /
0.4158 — the pipeline reproduces, and only *mixing generations* breaks.

**The transferable lesson**, which cost the most: `measure_blob_filter_damage.py`
reported the blob filter "harmless" TWICE — 0 of 20 and then 0 of 34 deleted blobs
bridged two surviving components — while the real benchmark says removing it is
worth **+0.0368 strict success**. The proxy only tested one failure mode. A stub
carrying a component PIN touches one wire island, so a "bridges ≥ 2 components"
test cannot see it. Sweep on the objective; a negative result is only as good as
the mechanism it tested.

### 0.10 The methodological hole a reviewer WILL find

`data/gt_1024`, `gt_netlists`, `gt_netlists_verified` and `_v3` overlap the 190
test images completely and the 192 val images **not at all**. So `bridge_span`,
`component_mask_pad`, `min_blob_area`, the snapping radii — every tuned value —
was chosen on the split it is reported on.

`scripts/cv_select_param.py` is the mitigation: choose per fold on the other
folds, score out-of-fold. For `bridge_span` the naive peak is 0.7321 and the
honest cross-validated figure is **0.7281**, with the value stable (7 wins 9 of
10 folds). Quote the second number. The real fix is item 4 above.

### 0.8 ENVIRONMENT: the disk is nearly full

`/System/Volumes/Data` is at **99%, ~2.8 GiB free**. Eight concurrent benchmarks
crashed with `OSError: [Errno 28] No space left on device` writing a temp SPICE
netlist. Keep concurrency at 3. `[HUMAN]` decision: these four archives total
~5.9 GB and their contents are already extracted —
`data/cghd/cghd-zenodo-12.zip` (3.2G), `data/digitize_hcd/Digitize-HCD_Dataset.zip`
(2.0G), `RTX_3090_training.zip` (428M), `runs.zip` (253M). Deleting them would
more than double free space; not done, since re-downloading is your call.

## 1. What this project is

IEEE Access submission: **"first public benchmark on Digitize-HCD"** for
hand-drawn circuit → SPICE netlist conversion, plus a lightweight local
pipeline and a novel transparent-repair layer. The governing plan is
`~/Downloads/IEEE_ACCESS_PLAN_v2_with_repair.md` (30-day, MSP-vs-IDEAL
tagging, contributions C1–C5). The owner is Ammaar Junaid (`[HUMAN]`
items are his). **Standing user conventions — do not break these:**

- One focused commit per step; co-author trailer; **push only when
  asked**. (`PROGRESS.md` was deleted by the user on 2026-07-27 — do
  not recreate it; this file is the running record.)
- **No hand-typed numbers anywhere**: every number in the paper
  regenerates from committed code + `results/` CSVs via
  `scripts/make_paper_tables.py`.
- Statistics: **paired per-image bootstrap** (2,000 resamples) for
  config comparisons on the same images, NOT independent-CI overlap.
- The user pushes back hard on quick fixes that dress up broken stages
  — measure root causes (oracle!) before proposing fixes, and report
  negative results plainly.

## 2. Current benchmark state

All on the 190-image test split vs **human-verified GT**. The canonical
GT is now `data/gt_netlists_verified_v3` (published-COCO box geometry,
human-verified topology), set once in `configs/default.yaml` under
`benchmark.gt_dir` and read by every script.

### ⚠ READ FIRST — state at session end

**Adopted default (committed):** `component_mask_pad: 0`,
`snapping.strategy: ports`, **`preprocess.target_size: 1024`** with
every dimensional parameter scaled to match, `preprocess.images_dir:
data/cleaned_1024`, `detect.cache_dir: data/detections_1024`,
`benchmark.gt_dir: data/gt_1024` (2x the v3 boxes; v3 remains the
canonical annotation and is what gt_1024 is generated from).

**⚠ THE 512 NUMBERS BELOW ARE SUPERSEDED — DO NOT QUOTE THEM WITH ANY
1024 RESULT.** Every committed ablation, the oracle, and the repair
evaluation were measured at 512. They are internally consistent with
each other and with nothing else. Regenerating them at 1024 is the
first task — see §3.5.

**Superseded headline (512, 3 detector seeds, 190 verified circuits):**

| metric | mean ± SD |
|---|---|
| terminal-pair F1 | 0.666 ± 0.006 |
| net F1 | 0.778 ± 0.004 |
| per-component acc | 0.492 ± 0.012 |
| strict end-to-end success | **0.305 ± 0.014** |
| DC-solvable after repair | 0.730 ± 0.008 |
| SPICE validity | 0.995 ± 0.000 |

Strict success was **0.058** at the start of this session. The 5.3x came
from two CORRECTIONS, not new capability — see §3. A single-seed 1024
run puts strict at 0.353 and per-component at 0.569; the 3-seed 1024
headline does not exist yet.

**⚠ THE 1024 REGENERATION QUEUE IS RUNNING — check it before anything.**

`scratchpad/run_regen_1024.sh`, launched 2026-07-29 08:43 under
`caffeinate -i`, log at `scratchpad/regen1024.log`. Eleven stages:
seeds 0/1/2, the six-row C2 ablation ladder, oracle, repair (`--verify`),
failure analysis, runtime, flagging calibration. Outputs go to NEW
directories (`results/benchmark_1024/`, `results/ablations_1024/`,
`results/oracle_1024/`, …) so the 512 results survive for comparison and
nothing is overwritten in place.

Watch it: `tail -f scratchpad/regen1024.log | grep -E "^--- |^### "`.
Each stage prints `--- <label> OK` or `--- <label> FAILED`, and the
script does NOT `set -e` — one bad stage is recorded and the queue
continues, then the tail prints `### STAGES FAILED: ...`. Check that
line before trusting the run.

Prerequisites already done, do not redo: ablation and seed configs
rebased onto 1024 into `scratchpad/seedcfg1024/` (by
`scratchpad/rebase_configs.py`, which replays each config's diff from
the OLD default onto the new one and doubles px-valued keys —
`component_mask_pad` 8 -> 16); `data/detections_seed1_1024` and
`data/detections_seed2_1024` generated, since seeds 1/2 previously had
512-frame boxes only.

When it reports `### ALL STAGES OK`, run **`scratchpad/finalize_1024.sh`**
— detector seed statistics at 1024, the C2 ablation ladder, all paper
tables via `make_paper_tables.py --variant 1024`, the three paired
contrasts that carry the argument (pad8-vs-pad0, boundary-vs-ports,
nostitch-vs-default), and the no-hand-typed-numbers audit.

`make_paper_tables.py` now takes `--variant {1024,512}` and defaults to
1024, so the frame sizes cannot be mixed in one table. Verified by
regenerating at 512: every table reproduces byte-identically except
`benchmark_3seed.tex`, which was itself stale — it still held
pre-correction numbers (strict 0.060) while `results/benchmark/seed*`
has held 0.305 since the two corrections. **Committed paper tables lag
committed results; regenerate before reading any of them.**

**⚠ THE LAPTOP SLEEPS AND SILENTLY 20x's LONG RUNS.** A benchmark chain
accumulated 27 min of CPU across 6.5 h of wall clock overnight. Always
wrap long chains: `caffeinate -i <script>`. Nothing in the logs
indicates sleep is the cause — it just looks pathologically slow.

### ⚠ SNAPPING IS DONE — the oracle's +0.133 was one bug (2026-07-29 late)

`ports.match_ports` assigned ports to boundary **runs**. One net crosses a
component's boundary several times, so runs repeat nodes, and several
ports could win several runs *of the same net* — a MOSFET-N reading
`[n1, n4, n4, n3, n4]` put Drain, Gate and Source all on `n4` with all
three distinct nodes present, at a 0.13-diagonal fit, so the trust check
never fired. Assignment is now over distinct **nodes**
(`src/schematic2netlist/ports.py`). Oracle mode C per-component
0.8675 → **0.9972**; attributed snapping headroom 0.1325 → **0.0028**.

**Per-component attribution is now: detection 0.0627, wires 0.3406,
snapping 0.0028.** Connectivity is the whole remaining story — do not
spend more time on snapping.

**End-to-end confirmed (paired bootstrap, 2000 resamples,
`results/comparisons/ports_distinct.csv`):** strict success **0.3526 →
0.3526, delta exactly 0.0000, CI [0,0], 0 wins / 0 losses / 190 ties** —
every image ties, as predicted. terminal-pair F1 −0.0001 (ns),
per-component −0.0040 (ns, CI spans 0), **nGED −0.0066 significant** (40
wins / 3 losses), DC-solvable post-repair +0.0053. So the fix is free on the
headline metric, significantly better on nGED, and its real value is the
corrected attribution plus multi-terminal netlist correctness.

Modes A/B fall −0.0040/−0.0088 and that is NOT real harm: the fall is
entirely inside images at terminal-pair F1 < 0.5 where the pipeline welds
the page into one node, every GT pair is then *vacuously* contained, and
per-component accuracy reads 1.0 at F1 = 0.25. No image at F1 ≥ 0.9
changed; images at F1 == 1.0 are 80 before and after. See
`results/snapping_diag/ports_distinct_impact.json`. This is also
independent motivation for the outstanding over-merge-sensitive metric.

**Caveat that changes how numbers should be read.**
`canonicalize_terminals` sorts by `(signature, ORIGINAL INDEX)`, so in
principle terminals tying on signature could leak their original order into
terminal-pair indices. **Measured: this never fires.** Only 13 of 2155 GT
components (0.6%) have a signature tie and all 13 are *same-net* ties,
where permuting is a no-op. Permuting every component's terminals at random
and scoring against the original returns terminal-pair F1 exactly 1.0 on
3820/3820 trials across all 191 images. The metric IS permutation-invariant
here; do not spend time "fixing" it. (Consequence: the 24.2%
permutation_only population really is invisible to every benchmark number.)
GT terminals carry only `{index, net}`; `bootstrap_gt_merged.py` filled
that order from the pipeline's own `node_names`, verified for nets not pin
identity. **C3's pin-identity claim has no ground truth here.** Wire
geometry also cannot decide polarity for 27.4% of components at all
(`results/pose_identifiability/`): Diode/Zener/V-DC/I-DC are 100% tied.
Either scope the C3 claim or build a hand-labelled port-identity set.

### ⚠ OVERNIGHT 2026-07-30: what worked, what died

**WORKED — class vote across detector seeds (the only positive result).**
Three seeded detectors already exist. Voting on the CLASS while keeping the
primary seed's boxes relabels 32 of 2815 detections, and against verified GT
**27 are corrections, 4 break, 1 stays wrong — net +23**, fixing 40% of the 67
known class errors. Free: no training, no GPU.
`scripts/ensemble_detection_classes.py` → `data/detections_1024_vote`.
All 4 failures are arrow-direction pairs (MOSFET-N↔P, BJT-PNP→NPN).

**DIED — vector tracer.** Significantly worse on every topology metric:
terminal-pair F1 −0.0300, per-component −0.0680, **strict −0.0421 (0 win / 8
lose)**. Only nGED and DC-solvability improve. So notch-and-relink is the best
mechanism available — it beats plain CC by +0.074 and vector by +0.030. An
earlier note claiming vector was "at parity on net F1" was wrong (0.7933 vs
0.8083). Phase 1b (vector + GT crossovers) cancelled as pointless.

**DIED — per-group class classifiers.** Trained on real train-split crops from
the pipeline's own frames. Shape cues learn beautifully (V-DC/V-AC 0.9861,
R/L 0.9779, Diode/Zener 0.9620, I-DC/I-AC 0.9362); arrow cues do not
(BJT 0.7154, MOSFET 0.6246). But **every group loses to the detector**
(−0.008 to −0.253) — a 72k CNN on 64px crops can't match YOLOv8s at 640px
with context. Not integrated. Beating it needs GPU-scale capacity.

**OPEN — resolution.** Welded regions gain significantly more ink components in
the ORIGINAL photograph than correct regions do (+2.16 vs +1.22, Mann-Whitney
p=0.00004, d=0.533, CI [+0.47,+1.40]), so part of the fusion is created by the
0.52-scale downsampling. A validated 2048 test-split stack now exists
(`scratchpad/plan_cfg/res2048.yaml`, `data/cleaned_2048`, `data/gt_2048`,
`data/detections_2048`); GT was rebuilt by GEOMETRIC matching to COCO because
double-projection was off 252–629px on 5 images, and validated against the
frames — 2048 has FEWER blank GT boxes than 1024 (0.35% vs 0.47%). Stroke
half-width 1.91 → 2.87. **Early warning: node counts and all topology metrics
were IDENTICAL to 1024 on the first 2 images.** Full run in flight.

**⚠ CORRECTION PENDING — the gtxover mechanism claim.** The Phase-0 oracle
(perfect crossover boxes → strict 0.3263, 5 previously-perfect images lost,
precision fell 5/5 = new welds from a wrong relink) ran BEFORE the
`already_split` guard existed (ab55f2ca is not an ancestor of either oracle
run). That guard exists to prevent exactly that damage. The re-run against
current HEAD is in flight; **do not quote the mechanism conclusion until it
lands.**

### ⚠ CONNECTIVITY IS INFORMATION-LIMITED — the crossing programme is closed (2026-07-30)

**Do not start another crossing classifier.** Every avenue was measured on the
same 4822 real sites, labelled from verified GT by the causal cut test (no
human annotation needed — `scripts/build_real_crossing_features.py`):

| approach | AUC | max usable precision |
|---|---|---|
| render CNN, 750 epochs | 0.4849 | — |
| render CNN v3 | 0.5094 | — |
| CGHD CNN (real photographs) | 0.596 | — |
| 8 geometric features, grouped CV | **0.6589** | **0.70** |
| ink darkness from ORIGINAL photos, grouped CV | 0.5379 | — |
| geometry + darkness, grouped CV | 0.6551 | 0.70 |

And two oracles cap it independently: perfect GT crossover boxes give strict
**0.3263** vs 0.3526 baseline (WORSE), perfect per-box decisions give **0.0**
terminal-pair headroom.

**Why, physically.** Max-flow on a welded node's skeleton (anchor stubs made
uncuttable) reports UNBOUNDED flow for 39 of 56 pairwise welds: both nets'
terminals reach the SAME arm. The two nets are one continuous conductor with
no branch point between them, so **no cut exists**. Binarization plus
collinear bridging fused them, and the greyscale cue that could have
distinguished them (ink doubles where strokes cross) is real *within* a
drawing — degree≥4 AUC 0.6548, split darker in 70.7% of images, pen-pressure
control at chance — but does **not** generalize between drawings.

The pipeline's own frames cannot even carry that cue: 93.5% of pixels are
exactly 255, ink crushed to median grey ~8. Measuring darkness there returns
0.4998 — a null on destroyed evidence. Originals hold ink median 86, sd 29.3.

**`nodes.vector.dot_ratio` is a decision rule running on noise.** AUC
**0.5017** over 4822 real sites (means 1.321 vs 1.372). Note the scope: the
junction-dot rule lives ONLY in `vector_nodes.py`, and the adopted default is
`nodes.method: crossover`, so it is not firing in the shipped pipeline today.
(The `dot` in `nodes.py` is a dot PRODUCT for arm collinearity, unrelated.)
It matters if and when vector is adopted — measure its removal then.

**`stitchable_mask` cannot bridge a component gap at all.** It sets the padded
box then erases the un-padded body; with `component_mask_pad: 0` those are the
same rectangle, so only text regions remain stitchable. That is why the
stitcher measured as a no-op — while 41% of split nets have their gap inside a
FOREIGN component's box. The obvious fix (open a corridor where a box shows
more stubs than terminals) was measured and REJECTED: only 8.4% of components
show extras and most show +1 where a through-wire needs +2, so the trigger
would fire ~223 times to repair ~19 splits.

### ⚠ WHAT CAN STILL MOVE, WITH NUMBERS (2026-07-30)

**The gate** (`results/blockers/strict_blockers.json`): 38 images (20%) have
`unmatched_gt > 0` so strict is IMPOSSIBLE for them; 152 are detection-clean,
of which 67 are strict and **85 connectivity-bound**. So connectivity work
alone caps strict at **152/190 = 0.80**.

**76% of the detection block is CLASS CONFUSION, not missed detection** — box
present at IoU ≥ 0.3, label wrong, in near-symmetric pairs (MOSFET-N↔P 16,
Inductor→Resistor 7, BJT-NPN↔PNP 3, I-AC→I-DC 3). 26 of 38 images are blocked
by exactly ONE component. **Detection mAP 0.9725 hides this entirely** — that
belongs in the paper next to mAP. This is where real annotations pay:
`scripts/train_class_disambiguator.py` (per-group models, rotations only —
a reflection would mirror the arrow that defines N vs P) and
`scripts/inject_gt_classes.py` prices it as an oracle first.

**Two prior-free detectors**, no threshold, nothing to train:

| constraint | GT rate | detections | precision |
|---|---|---|---|
| component shorts its own pins | 0.60% | 160 | **0.9500** |
| net has only ONE terminal | **0.00%** (0/1509) | 47 | **1.00** |

Zero currently-strict images contain either, so both are lethal and safe. But
coverage is thin — only 16 one-terminal nets and 25 self-shorts fall in the
reachable 0.5–0.9 band; 112 of 160 self-shorts sit in the hopeless <0.3 bucket.

**Connectivity decomposition** (`results/connectivity_diag/`): clean 58.6%,
welded 10.7%, split 3.8%, **welded+split 18.8%** (largest — the notch
signature), lost_terminal 0.2%, unmatched 7.9%. `unmatched` is a weld
symptom, not its own mechanism: every such net's terminals sit on nodes
carrying 2–6 GT nets. Node load: 58.4% carry 1 net, **33.7% carry exactly 2**,
7.9% carry ≥3. Pairwise welds are 93.2% ONE ink blob; mega-nodes are 53.8%
logical unions (relink artifacts).

**Reachable images need ~2 fixes but heterogeneous ones.** Of 45 images with
≤2 defects, only 3 are split-only and 3 weld-only; 26 carry an `unmatched`
net. So no single mechanism flips many images — fixes must compound.

### ⚠ WHERE STRICT SUCCESS ACTUALLY LIVES (2026-07-29 late)

Strict success is not spread across the test set — it is one bucket.
Stratified by terminal-pair **precision** (the over-merge axis),
`results/stratified_1024/precision_buckets.json`:

| precision | n | mean F1 | strict |
|---|---|---|---|
| ≥ 0.9 | 79 | 0.980 | **67** |
| 0.7–0.9 | 26 | 0.745 | 0 |
| 0.5–0.7 | 36 | 0.594 | 0 |
| 0.3–0.5 | 27 | 0.390 | 0 |
| < 0.3 | 22 | 0.262 | 0 |

All 67 strict successes come from the top bucket, which converts at 85%;
everything below converts at 0%. On the failing images recall stays ~0.51
while precision collapses — **the conductors are found and merely fused.**
So the work is separating nets on ~62 specific images in 0.5–0.9, not
improving an average. If those converted at the top bucket's rate strict
would reach ≈0.63; that is a target for locating effort, not a promise.

### ⚠ STRICT-SUCCESS PUSH (2026-07-29 evening) — read this block first

Mentor directive: strict success is the target, repair layer frozen,
benchmark framing. Findings, in order of importance:

**1. THE NOTCH IS CHAOTIC IN BOX PLACEMENT — the root cause of four
failures.** Shifting a crossover box 2 px takes terminal-pair F1 from
1.0000 to 0.5233 (circuit_1166) and 0.6582 (circuit_968). The offset
notch stops covering the intersection, so the nets stay welded. This
explains: the learned classifier (-0.110, synthesises boxes at arbitrary
centroids), the GT-crossover oracle (-0.026 strict; same count, centres
within 1-2 px of predicted, still lost the same 5 images), and every
sweep preferring fewer notches.
- `nodes.relink: snap` centres the notch on the enclosed skeleton branch
  point instead of the box centre. Jitter spread 0.4767 -> **0.0000**.
- It converges to the WORSE jittered value, i.e. those images scored
  1.0000 only because the notch MISSED. Part of the crossover method's
  per-image benefit is luck. Full 190 benchmark running; default stays
  `band` until it reports.
- `relink: angle` (ring-direction arm pairing) is byte-identical to
  `band` — that is what localised the cause to the notch, not the
  re-link. Kept, unused.

**2. Notching still earns its place in aggregate:** crossover beats plain
CC by **+0.074** terminal-pair F1 at 1024 (0.6845 vs 0.6102, 60 images).
So the lesson is "stop ADDING notches", not "stop notching".

**3. 86% of welds have a single intersection cut point** — and **41% of
those cut points are degree-3 (T) sites**, which every mechanism we
built refuses to split (`min_degree=4`; vector's junction default).
Causal test in `scripts/locate_welds.py` (93 welds, 25 images). An
earlier weaker version of that test reported 61% by asking only whether a
deg-4 site sits INSIDE a welded node; welded nodes are large, so that
number was meaningless and is superseded.

**4. Text masking is a measured NULL.** Perfect GT text masks score
terminal-pair -0.007 (sig), strict -0.005 (ns). 10.5% of text boxes are
fully unmasked and 48% of images are affected, but the misses are
electrically benign. The 18-class text detector's best case IS this null
— demoted to optional. `scripts/measure_textmask.py`,
`results/textmask_eval/`.

**5. Stitching is a complete no-op** on the shipped default (0/0/190 ties
on every metric but nGED). pad=0 removed its work.

**6. `nodes.method: vector` landed but is NOT adopted.** Splits without
editing ink (the only mechanism that can safely act on T-sites), but its
reconstruction loses connections CC keeps: 0.6385 vs 0.6795 with
splitting fully disabled, fragmentation 1.33 vs 1.25. Two confident
fixes (merging overlapping site disks; widening arm-incidence detection)
changed the numbers by EXACTLY ZERO, so the cause is not what I assumed.
Endpoint linking helps (+0.015). Unresolved, off by default.

**7. Crossing classifier v2 data is built and training.**
`data/crossings_synth`: 112,956 self-labeled patches (32.8k/59.9k train,
7.4k/12.8k val) rendered over real train/val layouts with exact
electrical labels by construction; test split never read. NOTE
`train_junction.py` silently OOM-died on float32 host storage — fixed to
uint8 + chunked validation. Its consumer should be a placement-invariant
splitter, NOT the notch path (see finding 1).

**⚠⚠ nGED WAS NOT DETERMINISTIC — it moved with machine load.**
`graph_edit_distance` passed `timeout=30.0` to networkx, which budgets
by WALL CLOCK, so the metric depended on how busy the machine was.
Running the benchmark six-way parallel instead of alone changed nGED on
19 of 190 images, always for the worse, by up to 0.167 — and the paired
bootstrap called that difference significant. Isolated directly:
byte-identical inputs score 0.1875 at a 10-second budget and 0.0208 at
30 seconds.

**STILL OPEN — a first fix was tried and REVERTED.** Replacing the
wall-clock budget with a fixed round count (`max_rounds`) removed the
time-dependence but was NOT bounded: `optimize_graph_edit_distance`'s
first yield can run unbounded, and on `circuit_1268` (21 nodes / 27
edges, near-isomorphic pred and gt — worst case for GED symmetry) it
hangs indefinitely, which would freeze every benchmark. Reverted to the
timeout-based version every committed number uses (bounded ~60 s/image).

**The correct fix, prototyped but not yet adopted:** a deterministic
polynomial GED upper bound from the KNOWN component alignment —
Hungarian-match pred nets to gt nets on symmetric-difference cost, add
insert/delete for unmatched nets. Gives 0 on isomorphic graphs, no
search, cannot hang. It redefines the metric (values shift; `circuit_1`
→ 0.214) so it must be adopted for the WHOLE suite at once and validated
on unmatched components first. See task "Add over-merge-sensitive metric"
sibling. Do this before quoting any nGED in the paper.

**Consequences.** The 512-vs-1024 nGED delta (−0.0104, "significant") is
UNRELIABLE — the two runs ran under different loads. The other four
significant metrics in that comparison — terminal-pair F1, net F1,
per-component, DC-solvable pre-repair — are computed exactly and stand.
Treat every committed nGED as provisional until the deterministic bound
lands. nGED is 1 of 7 metrics and does NOT affect the learned-classifier
verdict, which rests on the exact metrics.

**⚠⚠ THE COMMITTED DETECTION mAP DID NOT REPRODUCE — 0.972 vs 0.051.**
`results/detection/summary.json` reports mAP@0.5 = 0.9725. Re-running
its exact command today returns **0.051**. The detector is fine; the
LABELS were stale.

- `data/cleaned` and `data/transforms.json` were regenerated 2026-07-27
  (the preprocessing fix). `data/yolo_cleaned/labels/` dates from
  2026-07-23 and still projects the published COCO boxes through the
  OLD transforms — a systematic ~0.04-normalized y-offset.
- Proof it is the labels, not the weights: on `circuit_1`, predictions
  match the trusted v3 GT boxes **15/15** at mean IoU 0.890, and the
  stale YOLO labels **1/17** at 0.303. After rebuilding the labels,
  **17/17** at 0.892.
- Rebuild: `scripts/make_yolo_dataset.py --frame cleaned --cleaned-dir
  <frames> --transforms <transforms.json> --out-dir <out>`. Done for
  both frames: `data/yolo_cleaned_rebuilt` (512) and `data/yolo_1024`.

**This did NOT touch the pipeline benchmark.** Those runs score against
`gt_netlists_verified_v3`/`gt_1024` and consume cached detections, which
agree with v3 — the stale labels only ever fed `eval_detector.py`. So
every topology number stands; only the C4 detection table was wrong.

**Resolved — the manuscript's ~0.97 claim survives.** Re-evaluated
against rebuilt labels at both frame sizes:

| run | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| committed (labels fresh when written) | 0.9725 | 0.7264 |
| 512 rebuilt (`results/detection_512fixed`) | 0.9747 | 0.7109 |
| **1024, the default** (`results/detection_1024`) | **0.9739** | **0.7078** |

Detection is essentially resolution-insensitive, which is expected —
inference runs at `imgsz 640` from either frame size. `eval_detector.py`
no longer defaults to the stale dataset (running it bare was what
produced 0.051).

The real lesson is that regenerating `data/cleaned` silently invalidates
every derived artifact, and nothing checked. **Whenever preprocessing
changes, rebuild the YOLO dataset and re-run `eval_detector.py`** — and
run `scripts/audit_data_freshness.py`, which walks
frames -> transforms -> detections/labels/GT and exits nonzero on
anything older than what it derives from. It currently flags exactly one
artifact, the stale `data/yolo_cleaned`, which is retained only for
provenance and must not be evaluated against.

**⚠ SMOKE TESTS WRITE INTO COMMITTED RESULTS.** Every analysis script
defaults `--out-dir` to its canonical directory, so running one bare to
check its arguments overwrites a committed artifact. Doing exactly that
replaced `results/repair/summary.json` with a version missing the whole
`--verify` block (`topology_violations`, ground accuracy) and pointing
at a different `source_run`; `git checkout` restored it, and the loss
would have been silent otherwise since the file still parsed. **Always
pass a scratch `--out-dir` when smoke-testing**, and `git status
results/` before committing.

**⚠ THE RUNTIME STAGE NEEDS AN IDLE MACHINE.** `benchmark_runtime.py`
is stage 10 of the regeneration queue. Timings taken while anything
else runs are not measurements (REPRODUCE.md §6). Do not start sweeps,
demos or a second benchmark while the queue is in its final stages — or
re-run that stage alone afterwards.

**The demo is verified against the 1024 default** (`demo/app.py`, run it
with `./venv/bin/python demo/app.py`). It needed no porting because it
preprocesses and detects live from `CFG` rather than reading the
detection cache, so it followed the config automatically. Checked, not
assumed: `/api/process` returns all ten stages, every stage render is
1024x1024, and the stage-10 waveform payload is populated (`has_gt`,
four probes, both recovered and verified sims). `/config` and the
startup banner now report `target_size`.

### ⚠ THE BIGGEST FINDING: component_mask_pad was destroying the wires

`wires.component_mask_pad` had never been swept. It padded every
detected component box by 8 px before erasing it from the ink,
destroying the wire evidence immediately adjacent to each component —
exactly the terminals the next stage must find. Setting it to **0**,
full 190 images, canonical v3 GT:

| metric | pad=8 (old default) | **pad=0** | 95% CI |
|---|---|---|---|
| terminal-pair F1 | 0.494 | **0.625** | [0.585, 0.669] |
| net F1 | 0.674 | **0.752** | [0.721, 0.783] |
| per-component acc | 0.202 | **0.386** | [0.331, 0.444] |
| strict success | 0.060 | **0.221** | [0.168, 0.284] |

**Strict success is 3.7x.** Consequences:

- **Tier-1 stitching is now a NO-OP** — it existed to repair the holes
  the padding created. Removing the damage beats repairing it, and it
  retires the stage costing 50% of runtime.
- **Ports snapping (C3) was being suppressed by it.** At pad=8 it bought
  +0.013 terminal-pair F1; at pad=0 the 60-image sweep says +0.068 and
  per-component 0.344 -> 0.517. The contribution was real all along.
- DC-solvability is flat (0.717 -> 0.705), which is CORRECT: repair
  already made circuits simulate; better topology changes *which*
  circuit, not whether it runs.

`results/ablations/pad0/`. The pad=0 + ports run and a 15-axis sweep of
every remaining pipeline-time parameter were in flight at session end —
**check those before changing the default**, and change it once, with
everything regenerated against it.

Numbers below this line predate the finding and are pad=8.

**Headline, canonical v3 GT, 3 detector seeds** (`results/benchmark/seed{0,1,2}`):

| metric | mean ± std |
|---|---|
| terminal-pair F1 | 0.494 ± 0.010 |
| net F1 | **0.674 ± 0.005** |
| per-component acc | 0.202 ± 0.015 |
| nGED | 0.224 ± 0.004 |
| strict success | 0.060 ± 0.008 |
| DC-solvable after repair | 0.719 ± 0.032 |

Seed variance is small on topology metrics, so behaviour is dominated by
the algorithms rather than detector initialisation.

**Oracle on v3** (`results/oracle/`, 160/190 renders valid): detection
+0.034, **wires +0.437**, snapping +0.008. Wires own the error ~13× over
snapping.

**C3 ports ablation** (`results/comparisons/boundary_vs_ports.csv`,
paired, 190 images): terminal-pair F1 **+0.0126** [+0.0045,+0.0216] SIG,
per-component **+0.0168** SIG, net F1 +0.0046 ns. Ports help pin-level
metrics and not net-level ones — which is correct, since knowing which
pin a wire reaches does not change which wires exist. Do not oversell it
as a connectivity win.

**Runtime** (`results/runtime/`, 60 images, M1 CPU, cached detections):
83 ms/image. Stitching 50%, snapping 26%, everything else <10% each.

**Stratified** (`results/stratified/`): small circuits (≤8 comps) 0.717
net F1 / 0.125 strict; large (>16) 0.699 net F1 / **0.000 strict**.
Strict success is a product over components and collapses with size —
this is the answer to "why is strict only 6%".

The v2→v3 switch is measurement, not behaviour: same config, same
predictions, +0.038 net F1 (paired CI [+0.019, +0.060], 16 wins / 1
loss / 173 ties), while SPICE validity and solvability are bit-
identical because they do not depend on GT alignment. Full comparison
in `results/comparisons/gtboxes_v2_vs_v3.csv`.

The table below is the historical progression, still scored against
**v2** GT — it is internally consistent (every row same GT) and shows
the effect of each pipeline change, but its absolute values are ~0.04
low. Regenerate against v3 when the suite finishes
(`scripts/make_ablation_table.py`).

| Config | net F1 | term-pair F1 | per-comp | strict |
|---|---|---|---|---|
| v1 classical + directional snap | 0.4334 | 0.1627 | 0.0239 | 0.000 |
| v2 fixed preproc | 0.4310 | 0.2025 | 0.0362 | 0.000 |
| v3 + ink wires + boundary snap | 0.5853 | 0.3985 | 0.1424 | 0.026 |
| v4 + mask-hole stitching | 0.6065 | 0.4319 | 0.1737 | 0.053 |
| **v5 + crossover-aware CC = DEFAULT** | **0.6366** | **0.4626** | **0.1841** | **0.053** |

**Detection is solved**: yolov8s 0.972 mAP@0.5 (3 seeds ±0.0012).

**Repair (C5) is measured, not asserted** (`results/repair/`):
solvability 0.353→0.716, paired lift +0.363 CI [0.295, 0.432], **0
regressions**, **0/190 topology violations**, 4.0 assumptions + 1.7
gauge entries/circuit. Ground-choice gauge accuracy 0.821 on decidable
cases when a GND symbol exists (n=183); **0.000 when no GND symbol is
drawn** (n=7) — the most-connected-net fallback is a placeholder, not a
method. Report that honestly.

## 3. What changed this session (read this before planning)

1. **The oracle's conclusion REVERSED.** The old claim "snapping owns
   4× the error of wires" is dead. Two bugs caused it: the synthetic
   mode-C map used 0 for background where `build_wire_nodes` uses **-1**
   (so snapping read the whole page as one giant node), and mode B
   built detections from GT alone, which has no `Wire Crossover`
   entries, silently deleting the crossover boxes the baseline had.
   Both fixed; `oracle_render.py` now routes wires orthogonally
   (Lee maze, avoiding foreign bodies) and **verifies every render**,
   excluding and counting the ones that fail. **Wire tracing is now the
   correct next build target**, not snapping.
2. **C3 ports landed (MSP).** `configs/port_templates.json` distills
   Digitize-HCD's port-name annotations (Anode/Cathode,
   Drain/Gate/Source, …). Binning by **pose** (8 sectors) rather than
   axis roughly halves localization error for every directional class
   (Diode 0.221→0.088, MOSFET-P 0.427→0.198). `snapping.strategy:
   ports` matches boundary crossings to the best pose, so terminal *k*
   is the *k*-th named port — this fixes the arbitrary-terminal-order
   correctness bug. Falls back to boundary when no pose fits.
   Note two things the data does *not* support: single-port classes
   (GND, one-port source) have no orientation signal and sit at
   ~0.33/0.43 error, and both AC sources have **unnamed** ports
   upstream — their template fixes an ordering convention only, so the
   paper must not claim polarity for them.
3. **Paper exists** (`paper/`, Phase F started). IEEE Access scaffold,
   Intro / Related Work / Dataset / Method / Setup **drafted**,
   Results/Discussion/Conclusion skeletons. All numbers come from
   `scripts/make_paper_tables.py` (25 macros + 5 tables). `\todoa{}` =
   needs Ammaar; `\draftnote{}` = drafting note. Both must be empty
   before submission. **The scholarly voice is the user's — these are
   drafts for his revision, not finished prose.**
4. **A measurement bug was found in the benchmark itself.** GT boxes
   were bootstrapped from pipeline output and human-verified for
   *topology only*; ~20% are square, and a square box around an
   elongated symbol cannot exceed ~IoU 0.25 against a correct
   detection. 8.4% of GT components could not reach the 0.3 alignment
   threshold at all, and six images matched nothing and scored a
   spurious net F1 of 0.000. `scripts/fix_gt_boxes.py` rebuilds the
   boxes from the **published COCO annotations** (median centre shift
   0.0 px — the boxes were centred right and shaped wrong):
   matchability 91.6% → 98.4%, topology byte-identical.
   **v3 is now canonical** (owner's decision, 2026-07-28), set once in
   `configs/default.yaml` as `benchmark.gt_dir`; `_v2` is retained
   unmodified for provenance. Worth knowing when reading older
   numbers: `circuit_136` went from a recorded 0.000 to a *perfect*
   1.000 — it had always been reconstructed correctly and the
   benchmark could not see it.
4.5. **THE TWO CORRECTIONS THAT MOVED EVERYTHING.** Both were
   parameters nobody had swept, and both were *destroying evidence*
   rather than merely mistuned — which is why tuning could never have
   found them.

   - `wires.component_mask_pad` was 8: every detected component box was
     padded before being erased from the ink, deleting the conductor
     immediately adjacent to the component — exactly the terminal
     snapping must find. Setting it to 0: strict success 0.058 -> 0.221
     alone. Effect is monotonic in the pad
     (`results/sweeps/masking.csv`).
   - `snapping.strategy: ports` (C3) was measured at only +0.013
     terminal-pair F1 against padded masks and delivers +0.068 against
     clean ones. **The C3 contribution was real all along and was being
     suppressed by a defect two stages upstream.** Together: strict
     success 0.058 -> 0.321, paired bootstrap 50 wins / 0 losses,
     every topology metric significant
     (`results/comparisons/default_vs_pad0_ports.csv`).

   Consequence worth acting on: **tier-1 stitching is now a NO-OP** —
   it existed to repair the holes the padding created. Confirm from the
   ablation ladder's `abl_nostitch` row, then delete the stage; it was
   50% of runtime.

4.6. **A 15-axis sweep says the remaining parameter space is
   exhausted** (`results/sweeps/full_parameter_sweep.csv`): best
   available change +0.002, ~20 knobs inert at every value. Do not
   spend more time tuning.

3.5. **RESOLUTION EXPERIMENT — COMPLETE, 1024 ADOPTED.** Hypothesis:
   preprocessing downscales ~2000 px photos to 512, taking wire strokes
   from 7.4 px to **1.9 px** — near the survivability floor of
   binarization and morphology — and that this causes the shattering.
   Built `data/cleaned_1024`, `data/transforms_1024.json`,
   `data/gt_1024`, `data/detections_1024`; config at
   `scratchpad/seedcfg/res1024.yaml`; detection deliberately left at
   `imgsz 640` so boxes come back in frame coordinates and only wire
   tracing gains detail.

   Stroke width went 1.91 -> 2.74 px (43%, not the 2x predicted).

   **FINAL, 190 paired images** (`results/comparisons/res512_vs_1024.csv`):

   | metric | 512 | 1024 | delta | 95% CI | sig |
   |---|---|---|---|---|---|
   | terminal-pair F1 | 0.6712 | 0.7076 | +0.0365 | [+0.010,+0.065] | YES |
   | net F1 | 0.7797 | 0.8083 | +0.0286 | [+0.011,+0.048] | YES |
   | per-component | 0.5058 | 0.5692 | +0.0634 | [+0.017,+0.110] | YES |
   | nGED (lower better) | 0.1989 | 0.1886 | −0.0104 | [−0.019,−0.003] | YES |
   | strict success | 0.3211 | 0.3526 | +0.0316 | [−0.021,+0.084] | no |
   | DC-solvable pre-repair | 0.4737 | 0.5211 | +0.0474 | [+0.005,+0.090] | YES |
   | DC-solvable post-repair | 0.7211 | 0.7421 | +0.0211 | [−0.016,+0.058] | no |

   **METHODOLOGICAL WARNING — a partial read of this run was actively
   misleading, twice.** At 47/190 it looked null-to-negative
   (terminal-pair −0.016, per-component −0.019) and was reported as a
   null result. At 77/190 the sign had flipped to +0.024. The final
   answer is a significant win on 5 of 7 metrics. `benchmark.py` scores
   images in split order, which is not a random sample, so a prefix
   estimates nothing. **Do not report a delta from an unfinished run —
   wait, or shuffle the split.**

   Cost is not the obstacle: per-image pipeline time is 22 ms -> 73 ms
   (3.3x, but 51 ms absolute). Full-benchmark wall clock is dominated by
   nGED and ngspice, not the wire stage, so a 1024 ablation row costs
   roughly what a 512 one did.

   **The confound I flagged here was not real — resolved against me.**
   I had recorded that scaling `wires.morph_kernel` (2->4) and
   `wires.min_blob_area` (20->80) might be cancelling the gain, since
   both are noise-removal rather than dimensional. Measured at 1024 on
   60 images (`scratchpad/res1024_noisefloor.csv`):
   - `morph_kernel` is **inert under the default**. `extract_wires()`
     returns early for `method: ink` (wires.py:286) and only the canny
     baseline reaches the morphology block (wires.py:293). Reverting it
     is bit-identical on all four metrics — that identity is what
     exposed the dead knob.
   - `min_blob_area` reverted 80 -> 20 moves terminal-pair F1 by
     **−0.0024**, i.e. nothing. `clean_blobs` keeps a blob on area OR
     extent (wires.py:242), so this threshold only deletes specks that
     are also short. Real wire segments pass on extent regardless.

   Both config comments now say so. No published number depended on
   either knob (they appear only in `run_meta.json` config dumps).

   **Adopted in `configs/default.yaml`** using the exact parameter
   values the winning run measured, so the committed comparison
   reproduces from the config (verified: the only key that differs is
   the new `preprocess.images_dir`). To revert, set `target_size: 512`
   and undo the scaled block — but note the 512 results directories are
   all still on disk, so reverting costs nothing to check.

   **New guard.** Frame size lived in the config while the image
   directory was a CLI flag, so a 1024 config pointed at 512 frames
   scored the wrong pixels with no error anywhere — and detection boxes
   are in frame coordinates, so alignment silently corrupts too.
   `benchmark.py::assert_frames_match_config` now refuses to start on a
   mismatch; `tests/test_frame_guard.py` pins it, including a check that
   the shipped default is self-consistent. `sweep_wires.py` had the same
   class of bug (hardcoded `data/detections/`) and is fixed.

   Note what did NOT move: **strict success is +0.032 but not
   significant** (CI [−0.021,+0.084]), and post-repair DC-solvability is
   flat. Resolution buys connectivity, not end-to-end correctness — the
   remaining strict failures are not resolution-limited, so the argument
   in §3.4 about where the residual error lives still stands.

5. **Tier-4 path tracing was probed and rejected** — see priority 2 and
   `results/path_tracing_probe/`.
6. **Reproducibility scaffolding**: `REPRODUCE.md` maps every reported
   number to the command that makes it, and
   `scripts/audit_paper_numbers.py` fails the build if a result-shaped
   literal appears in prose instead of a generated macro. It caught a
   real drift on its first run (a stated class support of 470 against a
   committed 448).
7. `scripts/benchmark.py --gt-dir` now defaults to
   `data/gt_netlists_verified_v2` (the old default silently scored 0
   rows).
8. Tests 97 → **124**.

## 4. Running / unfinished at session end

**The 1024 regeneration queue is running — see the READ FIRST block in
§2 for how to watch it and what to run afterwards.** Everything below is
the state of the 512 work it supersedes.

The 512/v3 suite is COMPLETE and committed — 3-seed benchmark, ports
ablation, oracle, repair, stratified, runtime. It is internally
consistent and is the fallback if 1024 is ever reverted. It must not be
quoted alongside a 1024 number.

Still genuinely open:

- **The learned-connectivity verdict is UNDECIDED**, not negative — the
  partial run that suggested otherwise was deleted (see §6.2 for why and
  for the exact commands). This decides benchmark-only vs
  novel-contribution framing, so it is the highest-value item left.
- `results/ablations/wire_method.csv` was scored against v2 GT and is
  superseded twice over (v3, then 1024). `scratchpad/finalize_1024.sh`
  regenerates it as `results/ablations_1024/wire_method.csv`.
- Committed `paper/tables/*` lag committed `results/` — regenerate
  before reading them.

A partial `results/` directory means a stage was interrupted: check for
`summary.json` before trusting it, and re-run that one command. The
queue does not `set -e`, so read its final `### STAGES FAILED` line
rather than assuming success from the absence of noise.

**Determinism confirmed:** `results/benchmark/seed0` reproduced
`results/v5_stitch_crossover` to four decimals on every metric (same
config, same detections, independent run). Any future difference
between two runs of the same config is a bug, not noise. The same check
applies to the new queue: `results/benchmark_1024/seed0` runs the same
config as `results/ablations/res1024` and must match it — if it does
not, the parallel harness is at fault, not the pipeline.

Seed configs live in the session scratchpad (`scratchpad/seedcfg1024/`,
rebased onto 1024 by `scratchpad/rebase_configs.py`); regenerate
trivially by copying `configs/default.yaml` with `seed`,
`detect.weights` and `detect.cache_dir` changed.

## 5. Architecture and key files

```
configs/default.yaml       — SINGLE SOURCE OF TRUTH for thresholds
configs/port_templates.json— derived port templates (C3), committed
src/schematic2netlist/
  pipeline.py    preprocess.py   wires.py    nodes.py
  snapping.py    — boundary (default) | ports (C3) | directional | uniform
  ports.py       — port templates -> pin identity/polarity
  netlist.py     — role-based SPICE; pin order only meaningful under `ports`
  erc.py/repair.py     — C5 diagnosis + ledgered minimal repair
  repair_eval.py       — C5 measurement (lift, topology proof, gauge acc)
  oracle_render.py     — mode-C GT wire routing + render verification
  benchmark.py metrics.py gt.py classes.py detect.py simulate.py
scripts/
  benchmark.py           — THE evaluator (verified GT only by default)
  oracle.py              — GT-injection attribution (mode C valid again)
  benchmark_repair.py    — C5 evaluation (--verify recomputes per image)
  benchmark_runtime.py   — per-stage timings
  build_port_templates.py— rebuilds configs/port_templates.json
  make_ablation_table.py — consolidated C2 ablation CSV
  make_paper_tables.py   — ALL paper numbers/tables from results/
  compare_runs.py sweep_wires.py detect_batch.py eval_detector.py
  make_yolo_dataset.py record_transforms.py reproject_gt.py
  annotate_topology.py make_splits.py train.py ablate.py
paper/                   — IEEE Access manuscript (see paper/README.md)
```

Data (gitignored except splits + README): `data/raw` (sha256-verified),
`data/cleaned`, `data/transforms.json`,
**`data/gt_netlists_verified_v2/` = canonical GT**,
`data/gt_netlists_verified/` = user's read-only originals (NEVER
modify), `data/detections/` (+ `_seed1`/`_seed2` caches),
`data/digitize_hcd/extracted/.../Component Port Location Data/` (C3
supervision). Weights: `experiments/train_all/runs/yolov8s_640_seed{0,1,2}/`.

## 6. Next steps, in priority order

1. **Finish/verify the queued runs** (§4) and regenerate paper tables.
2. **Learned junction/crossover net assembly — BUILT, benchmark pending.**
   This is the live thread. Chain of evidence:
   - Deterministic tuning is exhausted: sweeping every stitching guard
     buys +0.02 terminal-pair F1 against +0.44 of headroom
     (`results/sweeps/stitch_guards.csv`).
   - Isotropic path tracing **fails** and was rejected before being
     built (`results/path_tracing_probe/`): tight settings merge almost
     nothing, loose ones short more nets than they fix. A scalar cost
     cannot separate "one rail with a gap" from "two nets passing
     close".
   - Phase 0 found the real target (`results/intersections/`): ~20
     stroke intersections per image, the detector labels **11%**, and
     **72.6%** of wire nodes carrying terminals fuse ≥2 GT nets.
   - CGHD annotates `junction` / `crossover` explicitly. 71,931 patches
     built by streaming the 3.2 GB archive (disk is at 98%, so do NOT
     extract it). Classifier trains to **0.969 balanced accuracy** on
     drafter-disjoint validation.
   - **MPS trains a degenerate model** (balanced acc pinned at 0.5000,
     silently). Use CPU or CUDA. Device auto-select skips MPS.
   - Integrated as `nodes.method: learned` (cc | crossover | learned),
     returning an audit record of sites found/classified/judged.
   **Remaining: threshold sweep + full benchmark + paired comparison.**
   Whether the extra splits are CORRECT is unmeasured — do not claim a
   win until `results/comparisons/crossover_vs_learned.csv` exists.

   **The verdict is UNDECIDED, not negative.** A 34/190 partial run of
   `results/ablations/nodes_learned/` existed and looked like a
   degradation; it was deleted rather than left to be misread, because
   the 1024 experiment proved a prefix of this benchmark can carry the
   wrong sign (§3.5) — 34 images is a quarter of the sample that lied.
   Nothing about the learned method has been measured to completion.
   Redo it at 1024 (`nodes.junction_site_box` is already scaled to 30)
   and only then compare:
   ```
   ./venv/bin/python scripts/threshold_sensitivity.py    # pick the threshold
   ./venv/bin/python scripts/benchmark.py --split test \
       --config <learned-1024.yaml> --out-dir results/ablations_1024/nodes_learned
   ./venv/bin/python scripts/compare_runs.py \
       results/benchmark_1024/seed0 results/ablations_1024/nodes_learned \
       --out results/comparisons/crossover_vs_learned.csv
   ```
   This is the experiment that decides benchmark-only vs novel-
   contribution framing, so it is the highest-value item left.
3. Remaining ablations: detector n/s/m, input resolution, legacy knobs
   (`min_blob_area`, `connectivity`, `ground_fallback`).
4. **Ground-selection fallback is a known-bad heuristic** (0/6). Either
   improve it (nearest-to-GND-symbol-like structure) or scope it out
   explicitly in the paper.
5. Paper: fill Results narration, Discussion, Limitations, Abstract;
   verify every `TODO-VERIFY` in `paper/refs.bib`; build the failure-
   cases and ledger-example figures. Then Phase G (red-team,
   reproducibility drill, numbers audit).
6. [IDEAL] learned port-heatmap model (the oracle-vs-axis gap in
   `results/ports/template_accuracy.json` quantifies its value);
   CGHD zero-shot; C5 expert-acceptance study.

## 7. Open issues / warts

- 44/190 mode-C renders fail verification (dense drawings where a pin
  is walled in). They are excluded and counted, not silently averaged.
- Detector was trained on OLD preprocessing frames; transfers fine
  (0.9747 mAP@0.5). User said retraining is for "some other time".
- `STATUS_REPORT.md` (untracked, 2026-07-23) is stale — superseded by
  this file. `docs/examples/` and `results/benchmark/` are untracked.
- **`per_component_connected_accuracy` is RECALL-ONLY and cannot see
  over-merging.** It counts a component correct when its GT terminal
  pairs are a SUBSET of the predicted pairs (`metrics.py:146`), so
  welding two nets together leaves it at 1.0 while net F1 collapses.
  Twelve images at 1024 score 1.00 per-component with strict success 0,
  one as low as net F1 0.222. Quote it alongside an F1 metric or a
  reviewer will rightly call it flattering — and never use `(1 - acc) x
  n` as a count of miswired components; that is a LOWER bound.
- **The failure mode flipped at 1024, from shattering to over-merging.**
  Terminal-pair precision minus recall: 512 `+0.0075` (shattered),
  1024 `-0.0295` (over-merged). The gain is real — precision rose too
  (+0.018), which pure merging cannot do — but recall rose three times
  as fast (+0.055), so per-component accuracy's +0.063 is its most
  flattering possible reading and the F1 deltas (+0.037 terminal-pair,
  +0.029 net) are the honest ones. This RAISES the value of the learned
  junction/crossover work (§6.2): over-merging is the more damaging
  error, since welding two nets corrupts every component on both, and
  splitting wrongly-fused nets is exactly what that classifier targets.
- Strict success is low in absolute terms, and the failures are NOT near
  misses: of 123 failing images at 1024, **zero** are one component away
  and 87.8% have four or more miswired (median at least 10 of 14
  components). Small fixes will not move strict success.
- Env: local venv (Python 3.11), `./venv/bin/python -m pytest -q`
  (147 passing + 1 xfail at the time of writing; the xfail is the
  deliberately-retained stale `data/yolo_cleaned`). Run it rather than
  trusting the count here.
- `data/yolo_cleaned` is kept only for provenance and is STALE by
  design; `scripts/audit_data_freshness.py` will always flag it and
  therefore always exits nonzero. Read its output, do not gate on its
  status alone.

## 8. [HUMAN] gates outstanding

ORCID + authorship decision; mentor second-read of ambiguous GT;
[IDEAL] C5 expert-acceptance study (~30 ledgers); APC ($2,160)
awareness; iThenticate access; biographies; GitHub push + Zenodo DOI;
RunPod sessions for any retraining. GT verification itself is DONE.
