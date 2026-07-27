# HANDOFF — Chip-Schematic-Converter / schematic2netlist

**For the next Claude Code session.** Read this first, then `PROGRESS.md`
(running log), then the plan. Written 2026-07-27 at the end of a long
session; context was near its limit, so this is the complete state dump.

## 1. What this project is

IEEE Access submission: **"first public benchmark on Digitize-HCD"** for
hand-drawn circuit → SPICE netlist conversion, plus a lightweight local
pipeline and a novel transparent-repair layer. The governing plan is
`~/Downloads/IEEE_ACCESS_PLAN_v2_with_repair.md` (30-day, MSP-vs-IDEAL
tagging, contributions C1–C5). It supersedes all earlier plans. The
owner is Ammaar Junaid (`[HUMAN]` items are his). **Standing user
conventions — do not break these:**

- Update `PROGRESS.md` (committed) after every phase; honest
  done/remaining/limitations. This is a hard user requirement.
- One focused commit per step; co-author trailer; push only when asked
  (pushing after milestone commits has been the recent pattern — ask if
  unsure).
- **No hand-typed numbers anywhere**: every number in the paper must
  regenerate from committed code + `results/` CSVs.
- Statistics: use **paired per-image bootstrap** (2,000 resamples) for
  config comparisons on the same images, NOT independent-CI overlap
  (that mistake was made once and corrected — see `results:` commit
  d926388b message).
- The user pushes back hard on quick fixes that dress up broken stages
  ("you're suggesting spinning the same terrible code") — measure root
  causes (oracle!) before proposing fixes; report negative results
  plainly.

## 2. Current benchmark state (the numbers that matter)

All on the 190-image test split vs **human-verified GT** (all 191 files
verified by Ammaar, strict validator passes, 0 issues).

| Config | net F1 | term-pair F1 | per-comp | strict |
|---|---|---|---|---|
| v1 old preproc + directional snap | 0.4334 | 0.1627 | 0.0239 | 0.000 |
| v2 fixed preproc + directional | 0.4310 | 0.2025 | 0.0362 | 0.000 |
| v3 + ink wires + boundary snap | 0.5853 | 0.3985 | 0.1424 | 0.026 |
| v4 + mask-hole stitching (tier 1) | 0.6065 | 0.4319 | 0.1737 | 0.053 |
| **v5 + crossover-aware CC (tier 2) = CURRENT DEFAULT** | **0.6366** | **0.4626** | **0.1841** | **0.053** |

Paired per-image bootstrap (2,000 resamples): tier 1 significant on ALL
five metrics incl. strict success (5→10 images, CI [+0.005, +0.053]);
tier 2 significant on all topology metrics, strict unchanged. Full
tables in commit 9d073d1e's message; per-image CSVs in
`results/v{3,4,5}_*/per_image.csv` (note: strict_success column stores
True/False strings in some CSVs — parse booleans).

**Detection is solved**: yolov8s 0.972 mAP@0.5 (3 seeds ±0.001),
per-class table in `results/detection/`. Detector transfers to the new
preprocessing frames without retrain (verified: 0.9747 mAP@0.5).

**Repair layer (C5) works**: DC-solvability 0.36→0.69 with ~4-5 logged
assumptions/circuit; ledger schema v1; topology provably untouched.

## 3. The session's key discoveries (why the config is what it is)

1. **Oracle attribution (`scripts/oracle.py`) is the compass.** It
   showed detection ≈ solved, snapping owned 4× the error of wires.
   Fixing snapping (boundary-crossing, class-aware terminal counts)
   doubled terminal-pair F1. ALWAYS re-run the oracle before choosing
   the next fix. **Known issue: oracle mode C is currently invalid** —
   its synthetic star-hub wire mask is read worse by boundary snapping
   than real wires (negative wire attribution = impossible). Re-render
   mode C along orthogonal routes before quoting wire-vs-snap splits.
2. **Stage interactions are real and reverse conclusions.** Ink-vs-canny
   flipped once boundary snapping landed; crossover-aware was a null
   result twice until stitching fixed net shattering, then became a
   clear win. Never conclude an upstream fix "doesn't work" while a
   downstream stage is broken.
3. **The dominant failure was rail-net shattering from self-inflicted
   mask holes** (component pad + text rectangles cut rails into 4-6
   islands, gaps 20-45 px at known locations). Tier-1 stitching
   (`stitch_wire_islands` in `wires.py`) reconnects across those known
   holes with collinearity + no-third-island + hole-explains-gap
   guards. Never stitch across a component BODY (its two leads are
   different nets).
4. **Strict success is a product over components** (~14/image median):
   per-comp 0.90 → ~23% strict. Getting "a lot more" strict requires
   per-comp in the high 0.9s. Errors are concentrated (rail fixes
   repair many components at once), which is why tier jumps are big.

## 4. Architecture and key files

```
configs/default.yaml       — SINGLE SOURCE OF TRUTH for all thresholds.
                             Current defaults: wires.method=ink,
                             stitch_masked_gaps=true,
                             snapping.strategy=boundary,
                             nodes.handle_crossovers=true (tier 2)
src/schematic2netlist/
  pipeline.py              — per-image orchestration
  preprocess.py            — annotation-aware crop, Hough skew,
                             project/unproject transforms
  wires.py                 — extract_wires (ink|canny), stitchable_mask,
                             stitch_wire_islands (tier 1)
  nodes.py                 — CC labeling + crossover-aware variant (tier 2)
  snapping.py              — snap_boundary (the big win), directional/
                             uniform legacy strategies
  netlist.py               — role-based SPICE export + model cards
  erc.py / repair.py       — C5 diagnosis + minimal ledgered repair
  benchmark.py             — align (Hungarian IoU), canonicalize
                             terminals, metric cascade, bootstrap CIs
  metrics.py, gt.py, classes.py, detect.py, simulate.py, determinism.py
scripts/
  benchmark.py             — THE evaluator (verified GT only by default)
  oracle.py                — GT-injection attribution (fix mode C!)
  sweep_wires.py           — fast config sweeps (no GED, seconds/image)
  detect_batch.py, eval_detector.py, detector_comparison.py
  make_yolo_dataset.py     — COCO→YOLO, --frame cleaned|raw
  record_transforms.py     — preprocess + containment guard (must exit 0)
  reproject_gt.py          — migrate GT bboxes between preprocess gens
  annotate_topology.py     — GT bootstrap/render/check
  make_splits.py, make_cghd_subset.py, preprocess.py (CLI), train.py
```

Data (gitignored except splits + README):
- `data/raw` = published images (sha256-verified) · `data/cleaned` =
  current preprocessed frames · `data/transforms.json` = per-image
  geometry (verified: 0/18,600 boxes off-canvas — `record_transforms.py`
  guards this, non-zero exit on regression)
- **`data/gt_netlists_verified_v2/` is the canonical GT** (bboxes in the
  current cleaned frame). `data/gt_netlists_verified/` = the user's
  read-only originals (old frame) — NEVER modify. `backups/` holds
  pre-migration copies of everything.
- `data/detections/` = per-image cache from LOCAL yolov8s (regenerate
  with `detect_batch.py` if preprocessing changes). Legacy Roboflow
  cache: `data/detections_legacy_roboflow/`.
- Weights: `experiments/train_all/runs/yolov8s_640_seed0/weights/best.pt`
  (+ seeds 1,2 + yolov8n/m). Config points at seed 0.
- Large archives (`runs.zip` etc.) are local-only, gitignored.

Results (committed: summary.json + per_image.csv + run_meta.json;
ledgers/netlists gitignored): `results/detection/`,
`results/ablations/wire_method/` (v1 baseline + crossover null),
`results/v2_newpreproc/`, `results/v3_boundary_snap/`,
`results/v4_stitch/`, `results/v5_stitch_crossover/`.

## 5. Next steps, in priority order

1. ~~Tier-1/2 bookkeeping~~ DONE (commit 9d073d1e: code + 6 tests +
   full-190 results + paired stats; config defaults = v5).
2. **Tier 3 — C3 port-heatmap keypoint CNN** (the plan's stretch, now
   justified): ViTPose-style top-down crops → K Gaussian heatmaps.
   Supervision SHIPS WITH Digitize-HCD:
   `data/digitize_hcd/extracted/Digitize-HCD Dataset/Component Port
   Location Data/<Class>/{Input Images,Output Heatmaps,XY Coordinates}`
   (thousands of crops/class). Wang et al. (Sensors 2026, 95.14%,
   doi 10.3390/s26113440) validate exactly this recipe. Train on the
   user's RunPod (RTX 3090; bundle workflow precedent in
   `README_TRAIN.md` inside the old training bundle — build a similar
   self-contained tarball). Integrate as `snapping.strategy: ports`
   with boundary fallback.
3. **Tier 4 — pin-anchored gap-tolerant path tracing** to replace
   global CC: Dijkstra over ink from pin to pin, direction-continuity
   cost, straight-through constraint inside detected crossover boxes.
4. **Fix oracle mode C** (orthogonal routing), re-run attribution after
   each tier.
5. **Remaining experimental program (MSP)**: 3-seed full benchmark of
   the best config; ablation tables via `scripts/ablate.py`; runtime/
   cost benchmark; repair evaluation (`benchmark_repair.py` not yet
   written — solvability lift, gauge accuracy, topology-preservation
   proof); CGHD zero-shot (100-img subset ready) [IDEAL].
6. **Paper (Phase F) — NOT STARTED.** IEEE Access LaTeX under `paper/`;
   the user wants to own the scholarly voice — scaffold + draft for his
   revision, don't finalize without him. Then Phase G verification
   (red-team, reproducibility drill, numbers audit).

## 6. Open issues / warts

- Oracle mode C invalid (above). — `circuit_1199` demo GT has stale
  bbox ordering; harmless (not in test split).
- Detector was trained on OLD preprocessing frames; transfers fine, but
  a retrain on new frames (+831 recovered boxes) is a known modest win.
  User said "some other time".
- `STATUS_REPORT.md` (repo root, untracked) and the session report in
  the scratchpad are point-in-time artifacts; superseded by this file.
- `results/benchmark/` (untracked) = the old circular smoke-test;
  never commit it as real numbers.
- Strict-success still low in absolute terms; per-comp acc is the
  number to move (see §3.4).
- Tests: 97 passing (`./venv/bin/python -m pytest -q`). Env: local
  venv (Python 3.11) with torch/ultralytics (MPS works for inference).

## 7. [HUMAN] gates outstanding

Mentor second-read of ambiguous GT + C5 expert-acceptance study
(~30 ledgers, [IDEAL]); ORCID + authorship decision; APC ($2,160)
awareness; iThenticate access; biographies; RunPod sessions for any
retraining. GT verification itself is DONE (all 191).
