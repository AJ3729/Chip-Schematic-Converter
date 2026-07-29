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
| Detector weights | `experiments/train_all/runs/yolov8s_640_seed{0,1,2}/weights/best.pt` |
| Detection cache | `data/detections_1024/` (`scripts/detect_batch.py --images data/splits/test.txt --images-dir data/cleaned_1024`) |
| Verified GT | `data/gt_1024/` (canonical topology is `data/gt_netlists_verified_v3`; `gt_1024` is the same annotation at 2x coordinates) |

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
    --weights experiments/train_all/runs/yolov8s_640_seed0/weights/best.pt \
    --out-dir results/detection_1024
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
./venv/bin/python scripts/benchmark_runtime.py --limit 60   # results/runtime/
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

`data/gt_1024` is v3 with every coordinate doubled, for the 1024-px
frames; component count, classes, terminals and net assignments are
identical, so it is the same ground truth expressed in the frame
coordinates the pipeline now runs in — not a different annotation.

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
