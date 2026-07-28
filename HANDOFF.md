# HANDOFF — Chip-Schematic-Converter / schematic2netlist

**For the next Claude Code session.** Read this first, then the plan
(`~/Downloads/IEEE_ACCESS_PLAN_v2_with_repair.md`). Last updated
2026-07-28 (session: Week-2 completion + Week-3 entry).

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

All on the 190-image test split vs **human-verified GT** (all 191 files
verified, strict validator passes). Consolidated in
`results/ablations/wire_method.csv` (regenerate with
`scripts/make_ablation_table.py`).

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
gauge entries/circuit. Ground-choice gauge accuracy 0.820 on decidable
cases when a GND symbol exists (n=174); **0.000 when no GND symbol is
drawn** (n=7) — the most-connected-net fallback is a placeholder, not a
method. Report that honestly.

**Oracle attribution is valid again** (`results/oracle/`, 146/190
renders verified): detection +0.0205, **wires +0.4697**, snapping
+0.0107.

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
   matchability 91.6% → 98.4%. Output is in
   `data/gt_netlists_verified_v3/` with topology byte-identical;
   **the canonical GT was not modified** and a comparison benchmark is
   queued. Adopting v3 is the owner's call, but the evidence is
   one-sided: it makes the GT match the published annotations instead
   of an echo of our own pipeline.
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

A background chain was left running (check `results/` before rerunning):

- `results/benchmark/seed{0,1,2}/` — 3-seed full benchmark. **Verify it
  completed**; `paper/tables/benchmark_3seed.tex` says "pending" until
  ≥2 seeds have `summary.json`, then regenerate the paper tables.
- `results/ablations/snapping_ports/` + `results/comparisons/
  boundary_vs_ports.csv` — the C3 ablation (boundary vs ports).
- `results/gt_v3_boxes/` + `results/comparisons/gtboxes_v2_vs_v3.csv` —
  how much of the current numbers were the GT-box artifact (§3.4).
- `results/threshold_sensitivity/` — alignment-threshold robustness.
- `results/runtime/` — per-stage timings, deliberately last so the
  machine is quiet (timings under CPU contention are worthless).

Each stage runs only after the previous finishes, so a partial
`results/` directory means the chain was interrupted — check for a
`summary.json` before trusting any of them, and just re-run that
script's one command.

**Determinism confirmed:** `results/benchmark/seed0` reproduced
`results/v5_stitch_crossover` to four decimals on every metric (same
config, same detections, independent run). Any future difference
between two runs of the same config is a bug, not noise.

Seed configs live in the session scratchpad; regenerate trivially by
copying `configs/default.yaml` with `seed`, `detect.weights` and
`detect.cache_dir` changed.

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
2. **Wire connectivity — but NOT the tier-4 design as planned.** The
   oracle says wires own +0.47 of the error, so this is still the
   highest-value target. However, `scripts/explore_path_tracing.py`
   already probed the planned approach (Dijkstra over an ink-vs-gap
   cost field, anchored at boundary-crossing sites) and it **fails**:
   see `results/path_tracing_probe/`. Tight cost settings merge almost
   nothing (2–4 new connections over 30 images at precision 1.0);
   loose ones short more nets than they fix (precision 0.19–0.56). An
   isotropic distance cost cannot separate "one rail with a gap" from
   "two nets passing close". Tier-1 stitching worked precisely because
   it demanded collinearity at both endpoints plus a hole explaining
   the gap. **Any next attempt needs directional evidence** —
   state-augmented (pixel, direction) search with turn penalties, or a
   learned junction/crossing discriminator — not a scalar cost field.
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
- Strict success is low in absolute terms; per-component accuracy is
  the number that moves it (it is a product over ~14 components).
- Env: local venv (Python 3.11), `./venv/bin/python -m pytest -q` →
  119 passing.

## 8. [HUMAN] gates outstanding

ORCID + authorship decision; mentor second-read of ambiguous GT;
[IDEAL] C5 expert-acceptance study (~30 ledgers); APC ($2,160)
awareness; iThenticate access; biographies; GitHub push + Zenodo DOI;
RunPod sessions for any retraining. GT verification itself is DONE.
