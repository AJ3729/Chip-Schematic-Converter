# HANDOFF — Chip-Schematic-Converter / schematic2netlist

**For the next Claude Code session.** Read this first, then the plan
(`~/Downloads/IEEE_ACCESS_PLAN_v2_with_repair.md`). Last updated
2026-07-29 (long session: Week-2/3 completion, two major pipeline
corrections, resolution experiment).

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
