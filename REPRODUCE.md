# Reproducing every number

This file is the contract behind the project's rule that **no number in
the manuscript is hand-typed**. Each reported quantity is produced by one
of the commands below, written to a file under `results/`, and read from
there by `scripts/make_paper_tables.py`, which emits the LaTeX macros and
tables the paper `\input`s. If a number cannot be traced to a command
here, it does not belong in the paper.

Run everything from the repository root with the project venv
(`./venv/bin/python`, Python 3.11). `ngspice` must be on PATH.

## 0. Prerequisites

| Requirement | Notes |
| --- | --- |
| Digitize-HCD | `data/raw` sha256-verified against the published archive; see `data/README.md` |
| Preprocessed frames | `data/cleaned_1024` + `data/transforms_1024.json` (`scripts/preprocess.py`, guarded by `scripts/record_transforms.py`) |
| Detector weights | `experiments/train_valstop/runs/yolov8s_640_seed{0,1,2}/weights/best.pt` — early-stopped on `val`; see the caveat below |
| Detection cache | `data/detections_valstop/` (`scripts/detect_batch.py --config configs/default.yaml --sleep 0`) |
| Verified GT (test, reported) | `data/gt_test_1024/` — 192 images, `benchmark.gt_dir`; verification account in `docs/GT_VAL_VERIFICATION_REPORT.md` |
| Verified GT (val, selection) | `data/gt_val_1024/` — 190 images; canonical topology is `data/gt_netlists_verified_v3` and `gt_val_1024` is the same annotation at 2x coordinates |

**The two evaluation splits swapped names on 2026-08-03.** The 190 images
every parameter was tuned on are now `val`; the 192 that never entered
selection are now `test`. Sweep with `--split val --gt-dir data/gt_val_1024`;
report with `--split test` (the default). Any artifact under `results/`
committed before that date was computed on the 190 images and is a
validation number regardless of what its `run_meta.json` calls it. Full
mapping: `data/README.md` and `data/splits/splits_meta.json` → `role_swap`.

**The detector was retrained on 2026-08-05 so early stopping reads `val`.**
The previous weights (`experiments/train_all/`) early-stopped on the split that
is now `test`, worth +0.0169 mAP@0.5 / +0.0231 mAP@0.5:0.95 measured as the
test-minus-val gap. The replacement was trained from a packet containing only
`train` and `val` — the reported test images were not on the training machine —
and its test-minus-val gap is +0.0088 / +0.0033. Both weight sets are kept;
`experiments/train_all/` is retained only so the contamination measurement can
be reproduced, and must not be used for a reported number. Retrain settings:
yolov8s, 640 px, 300 epochs, patience 50, batch 32, `deterministic=True`,
seeds 0/1/2.

Every script now takes `--split` with a default set by its role — see
`src/schematic2netlist/splits.py`. Exploratory and selection scripts default
to `val` so nothing can leak into a reported number by omission; only
scripts whose output is reported default to `test`. Reproducing the paper's
result set is one command:

```bash
./venv/bin/python scripts/regen_on_split.py --split test --fill-caches
```

It replays each ablation arm and detector seed from the frozen config
snapshot inside that run's `run_meta.json`, so every arm keeps the exact
configuration it is being compared against, then
`scripts/make_paper_tables.py` turns the output into `paper/generated/`.

**That caveat is now historical.** It read: "the detector was early-stopped on
the 192 images now called `test`, which makes detection metrics there optimistic
by a measured +0.017 mAP@0.5". The 2026-08-05 retrain removed it; the surviving
test-minus-val gap is +0.0088 mAP@0.5 / +0.0033 mAP@0.5:0.95
(`results/final/detection/seed{0,1,2}/{test,val}`). The pre-retrain measurement
is preserved in `results/detection_test192/{test,val}` solely so the
contamination estimate stays reproducible.

**Frame size is part of the configuration.** `preprocess.target_size` is
1024 and `preprocess.images_dir` names the matching frames; every script
that feeds images to the pipeline resolves the directory through
`schematic2netlist.frames.resolve_and_check` and **refuses to run** if
the frames on disk are not that size. Before this guard existed, running
a 1024 config against 512 frames scored the wrong pixels with no error
raised — and because detection boxes are stored in frame coordinates,
component alignment corrupted too. The 512 results in `results/` predate
the switch and must not be quoted alongside a 1024 number.

`data/` and `experiments/` are gitignored; the frozen split manifests and
`data/README.md` are versioned because they are part of the benchmark.

## 1. Detection (M1)

**Rebuild the YOLO dataset whenever preprocessing changes.** Labels are
the published COCO boxes projected through `data/transforms*.json`, so
regenerating `data/cleaned*` invalidates them. This bit once: labels
left over from an older preprocessing run made the committed mAP@0.5 of
0.9725 irreproducible (the same command returned 0.051) with no error
anywhere — the labels still parsed and the counts still matched.
`tests/test_yolo_labels_fresh.py` now re-derives labels from COCO and
fails if they disagree with the transforms on disk.

```bash
./venv/bin/python scripts/make_yolo_dataset.py --frame cleaned \
    --cleaned-dir data/cleaned_1024 --transforms data/transforms_1024.json \
    --out-dir data/yolo_1024
./venv/bin/python scripts/eval_detector.py --split test \
    --data data/yolo_1024/dataset.yaml \
    --weights experiments/train_valstop/runs/yolov8s_640_seed0/weights/best.pt \
    --out-dir results/final/detection/seed0/test
```

`data/yolo_cleaned` is the stale 512-px dataset and must not be used;
`data/yolo_cleaned_rebuilt` is its corrected equivalent.

→ `results/detection/{summary.json,per_class_ap.csv,seed_stats.json}`
(mAP@0.5, mAP@0.5:0.95, per-class AP with supports, 3-seed mean±std).

## 2. Primary benchmark and the C2 ablation

```bash
./venv/bin/python scripts/benchmark.py --split test --out-dir results/benchmark/seed0
./venv/bin/python scripts/make_ablation_table.py
```

`benchmark.py` runs the pipeline per verified GT image and writes
`summary.json` (bootstrap 95% CIs), `per_image.csv` and per-circuit
ledgers. `make_ablation_table.py` consolidates the committed runs into
`results/ablations/wire_method.csv` — the classical → boundary-snap →
stitching → crossover-aware progression.

Comparing two configurations is always **paired**:

```bash
./venv/bin/python scripts/compare_runs.py results/v4_stitch results/v5_stitch_crossover
```

Significance comes from a bootstrap over per-image deltas, never from
the overlap of two independent CIs.

## 3. Oracle stage attribution (C4)

```bash
./venv/bin/python scripts/oracle.py --limit 190 --out-dir results/oracle
```

→ `results/oracle/{summary.json,per_image.csv}`. Mode C injects
GT-derived connectivity; only images whose synthetic wiring passes
verification are scored, and the count of exclusions is reported with
the result. If the wire attribution ever comes out negative the script
says so loudly — that means the injected oracle is unreadable by the
stage being measured, not that wires are free.

## 4. Repair layer (C5)

```bash
./venv/bin/python scripts/benchmark_repair.py \
    --run-dir results/v5_stitch_crossover --out-dir results/repair --verify
```

→ solvability lift with a paired CI, per-issue histogram, minimality
budget, and (with `--verify`, which re-runs the pipeline per image) the
topology-preservation count and ground-choice gauge accuracy.

## 5. Port templates (C3)

```bash
./venv/bin/python scripts/build_port_templates.py
```

→ `configs/port_templates.json` (committed, derived data) and
`results/ports/template_accuracy.json` (held-out localization error
under oracle-pose and axis-only regimes).

## 6. Supporting analyses

```bash
./venv/bin/python scripts/analyze_failures.py        # results/stratified/
./venv/bin/python scripts/threshold_sensitivity.py   # results/threshold_sensitivity/
./venv/bin/python scripts/measure_runtime.py --split test \
    --out-dir results/final/runtime   # two scopes: cached + true end-to-end
./venv/bin/python scripts/explore_path_tracing.py --limit 40  # results/path_tracing_probe/
```

Run the runtime benchmark on an otherwise idle machine — timings taken
under CPU contention are not measurements.

## 7. Ground-truth box provenance

GT topology files were bootstrapped and then human-verified for **net
assignments**; their bounding boxes were not verified and are used only
to align predictions to GT.

```bash
./venv/bin/python scripts/fix_gt_boxes.py            # dry run: reports the impact
./venv/bin/python scripts/fix_gt_boxes.py --apply --out-dir data/gt_netlists_verified_v3
```

This rebuilds box geometry from the published COCO annotations, leaving
topology byte-identical (`tests/test_gt_boxes.py` enforces that). Score
against it with `--gt-dir data/gt_netlists_verified_v3` to see how much
of a result depends on box geometry.

`data/gt_val_1024` is v3 with every coordinate doubled, for the 1024-px
frames; component count, classes, terminals and net assignments are
identical, so it is the same ground truth expressed in the frame
coordinates the pipeline now runs in — not a different annotation.

The test GT (`data/gt_test_1024`) has no such lineage: it was annotated
natively in the 1024 frame, from component inventory and published COCO
boxes only, with every net traced from `null`. It is reproducible from its
own decision records — `gt_test_1024/decisions/<stem>.json` holds the
junction/crossing call at every critical site and the nets are recomputed
from those, never hand-edited.

## 8. Regenerate the manuscript's numbers

```bash
./venv/bin/python scripts/make_paper_tables.py
```

→ `paper/generated/numbers.tex` (macros used in prose) and
`paper/tables/*.tex`. Re-run after any change under `results/`. Tables
whose inputs are missing say so in the generated file rather than
silently omitting a row.

## 9. Audit and tests

```bash
./venv/bin/python scripts/audit_paper_numbers.py   # no hand-typed results
./venv/bin/python -m pytest -q
```

The audit fails if a result-shaped number appears in prose instead of
coming from a generated macro or an `\input` table. Dataset facts and
figures quoted from cited work are allowlisted in the script, each with
the reason it is not one of our results.

Tests that need gitignored data skip cleanly rather than fail, so a
clean clone still runs the suite.
