# Project Progress — IEEE Access Submission

Working log for rebuilding this repo + paper into an IEEE Access
submission ("From Sketch to SPICE"). Updated at every phase. Plan
phases: A repo foundation → B data/GT → C models/baselines → D metrics
→ E ablations → F paper rewrite → G pre-submission verification.

---

## Phase A — Repo Foundation & Reproducibility ✅ (2026-07-16)

### Done
- **Baseline snapshot** (`4b6d42ed`): pre-restructure state committed as
  evidence, including the broken `pipeline_evaluation.csv` (every row a
  KeyError) that proves the old paper's Table 2 was not reproducible.
- **Hygiene** (`d42895cd`): untracked venv/ (8,521 files) and data/
  (2,554 images) from git; new .gitignore; legacy artifacts moved to
  `experiments/legacy/`; deleted duplicate `preprocess copy.py`.
- **Package** (`4ed945cc`): installable `schematic2netlist` (src
  layout, pyproject, MIT license, full README). Logic migrated verbatim
  from the legacy scripts; **verified byte-identical** artifacts on
  circuit_1199 (all 6 debug PNGs + netlist.sp + netlist_readable.txt).
  Both legacy pipeline variants preserved as config switches
  (`snapping.strategy`, `netlist.ground_fallback`,
  `wires.min_blob_area`) — ready-made ablation axes.
- **Config centralization** (`configs/default.yaml`): every previously
  hardcoded threshold, documented, with v1-vs-v2 differences noted.
- **Evaluation harness rebuilt** (`6a097f83`) fixing the two documented
  bugs: per-image detection caches (no more one-shared-detections.json)
  and normalized `class`/`class_name` keys. Missing detections are
  reported as such, never silently scored.
- **Determinism** (`978be683`): global seeding (random/numpy/torch);
  every run writes `run_meta.json` (config + git SHA + seed + env).
- **Tests** (`9ae990c5`): 32 unit tests — netlist writer guarantees,
  exact hand-computed metric values (pair F1, Hungarian net F1,
  GED/nGED), ngspice failure taxonomy.

### Acceptance criteria — all verified
`pip install -e .` ✓ · `run_pipeline.py --image
data/cleaned/circuit_1199.jpg` reproduces legacy outputs byte-for-byte
✓ · pytest 32/32 ✓

### Limitations / problems found
- **Two latent v1 quirks preserved verbatim** (flagged in
  `netlist.py` comments, to fix in a post-migration refactor because
  fixing them changes outputs): (1) the `"supply"` substring branch
  shadows AC supplies — they are emitted as DC sources; (2) diode and
  Zener share the `D` element prefix but use separate counters, so one
  of each can produce duplicate `D1` names (ngspice error).
- Only `circuit_1199.json` exists in the per-image detection cache;
  batch evaluation needs Phase C local inference (or Roboflow re-runs).
- The plan's repo-state facts were partially stale: data/raw and
  data/cleaned both contain 1,277 images (not 624 vs 1,279).

---

## Phase B — Data: Splits, Annotations, Ground Truth ✅ code-complete (2026-07-18) — [HUMAN] GT verification pending

### Done
- **B1 acquisition + provenance** (`7daaa81b`): Digitize-HCD downloaded
  from Mendeley (doi 10.17632/rngcz5wtv8, CC BY 4.0, 2.1 GB zip);
  **sha256 verified** against the published hash. Confirmed contents:
  1,277 original-resolution images, COCO component annotations
  (**18,600 boxes, 17 categories**), MMOCR-style text annotations
  (with transcriptions — usable for future OCR work), and per-class
  **port-location Gaussian heatmaps** + pixel coordinates (so
  contribution C4 is feasible). Mendeley v1 and v2 reference the same
  archive; v2 is a metadata revision.
- **Reconciliation** (`scripts/reconcile_data.py` →
  `data/reconciliation.json`): local `data/raw` is **byte-identical**
  to the published image set (1277/1277 matched, 0 mismatches, 0
  duplicates); `data/cleaned` are preprocessed derivatives matched by
  filename. The plan's feared 624-vs-1279 discrepancy does not exist.
- **B2 frozen splits** (committed): train 895 / val 192 / test 190
  (70.1/15.0/14.9), stratified by component-count tertile ×
  rarest-class from the published annotations; all 17 classes in every
  split (test support: 44–470). Full distributions in
  `data/splits/splits_meta.json`.
- **B3 GT tooling** (`9ad8ab43`) + **bootstrap executed**: schema v1
  loader/strict-validator (`gt.py`), annotation workflow
  (bootstrap → correct → render → check). Detection cache filled for
  all 190 test images via hosted Roboflow (0 failures); **191 GT
  files bootstrapped** (190 test + circuit_1199 demo), all overlays
  rendered, all passing validation. 12 new tests (44 total passing).
- **B4 CGHD complete** (`f7930ab8`): 3.4 GB zip downloaded from Zenodo
  and md5-verified. Layout: **25 drafter folders, 3,255 images**,
  Pascal VOC annotation XMLs (drafter folders also carry
  segmentation/instances and some `spice/` data — potential extra GT
  source worth revisiting). Upstream `classes.json` has 53 entries
  (not 59 as the plan said). `data/cghd/class_mapping.yaml` maps CGHD
  → the 17 published Digitize-HCD categories with lossy mappings
  marked; I-DC, I-AC, and V-DC (one port) have no CGHD counterpart.
  Deterministic drafter-stratified zero-shot subset extracted (4 per
  drafter × 25 = 100 images, 106 MB); frozen manifest committed at
  `data/splits/cghd_zero_shot.txt`.

### Remaining (this phase)
- **[HUMAN] GT verification pass** — the single most valuable human
  task in the project (~1–2 min/image × 190): for each
  `data/gt_netlists/<stem>.json`, check the render in
  `data/gt_netlists/renders/`, correct classes/nets/missed components,
  then set `verified: true` and `annotator`. Re-run
  `annotate_topology.py --render/--check` while editing.

### Limitations / problems
- **No drafter metadata** in the published COCO → drafter-disjoint
  splitting is impossible. Stated in data/README.md and
  splits_meta.json; must be stated in the paper's limitations (CGHD
  papers split by drafter; reviewers may raise this).
- **Class-name mismatch, pipeline vs published annotations**: the
  pipeline's snapping/netlist stages branch on Roboflow-era lowercase
  substrings ("ground", "dc supply"), but the published categories are
  GND, V-DC, I-DC, etc. — `"gnd".lower()` does NOT contain "ground",
  so a Phase C detector trained on published names would break ground
  handling. Needs a class-normalization map at detection load time
  before Phase C evaluation runs.
- **Annotation/preprocessing coordinate mismatch**: published
  annotations are in original-image coordinates; `data/cleaned` was
  produced by an unrecorded rotate/crop/resize. Phase C must train on
  `data/raw` (annotations valid there) or extend
  `scripts/preprocess.py` to record its transform matrix. The GT
  bootstrap detections/bboxes are in cleaned-image coordinates —
  consistent with the pipeline, but not with published annotations;
  GT topology (net membership) is coordinate-free, so this only
  affects bbox visual aids, not the benchmark labels.
- **Bootstrap GT quality tracks pipeline quality**: bootstrapped nets
  come from the current heuristic pipeline (~0.78 snap coverage on the
  demo image), so the human pass must fix real errors, not rubber-stamp
  — the render coloring makes disagreements visible by design.
- Roboflow-model class set ≠ published 17 (e.g. Roboflow has
  "operational amplifier"/"MOSFET Transistor"; published has
  Op-Amp/MOSFET-N/MOSFET-P/BJT-*, Wire Crossover). Test-split GT
  bootstraps therefore miss classes the hosted model can't emit —
  another reason the human verification pass matters, and why Phase C
  retrains locally on the published annotations.
- Disk: ~11 GB free with both zips resident; CGHD extraction must be
  selective. Consider deleting the Digitize zip after extraction is
  re-verified (sha256 recorded in data/README.md).

---

## Phase C — Models & Baselines 🔶 (started 2026-07-18)

### Done
- **Prerequisites cleared** (`9d92bc9f`):
  - Canonical class vocabulary = the 17 published categories
    (`configs/class_names.yaml`, `classes.py`); legacy names are
    aliases; pipeline branches on roles. Fixes GND-not-recognized-as-
    ground and both v1 netlist quirks; adds .model cards, 3-terminal
    transistor / op-amp (ideal VCVS) / one-port rail support. Wire
    Crossover excluded from masking (would sever crossing wires),
    snapping, and netlists.
  - Preprocessing transforms recorded for all 1,277 images and
    verified **byte-identical** against data/cleaned
    (`scripts/record_transforms.py` → `data/transforms.json`) —
    published annotations can now be projected into cleaned
    coordinates (also unblocks ablation E2).
  - **GT re-bootstrapped as a merge** (`bootstrap_gt_merged.py`):
    components from published COCO (complete, correctly classified,
    projected boxes), nets transferred from pipeline snapping by IoU
    ≥0.3 — 190 files, 2,563 components, 1,522 (59%) with transferred
    nets; all validate. Human pass is now mostly net-checking.
- **GT verification guide**: `docs/GT_VERIFICATION_GUIDE.md` — full
  [HUMAN] workflow, prerequisites, judgment calls, 6–8 h realistic
  budget.
- **YOLO dataset built** (`scripts/make_yolo_dataset.py`):
  COCO → YOLO labels on data/raw (annotation coordinate frame), frozen
  splits, all 17 classes, 18,600 boxes; `data/yolo/dataset.yaml`.
- **Training started on Apple M1 (8 GB, MPS)**: smoke run passed
  (yolov8n/320/1 epoch); primary run launched — **yolov8s, 640 px,
  batch 8, 100 epochs, seed 0, deterministic**, under caffeinate, to
  `experiments/train/yolov8s_640_seed0/`.

### Remaining
- Wait out seed-0 run (rough ETA 6–12 h on M1 8 GB; early-stops on
  patience). Then seeds 1 and 2 (plan: mean ± std over 3 seeds).
- Detector baselines (C2): yolov8n/m, v5s, v10s/11s, RT-DETR or
  Faster R-CNN — consider a rented GPU; M1 wall-clock makes the full
  baseline family painful locally.
- Switch pipeline detect backend to local weights; regenerate
  detection caches; per-class AP with support counts (C1).
- End-to-end baselines (C3) and runtime/cost benchmark (C4→paper C2).

### Limitations / problems
- M1 8 GB is the only compute: yolov8s@640 batch 8 fits but each run
  is overnight-scale, and the C2 baseline family (6+ detectors × 3
  seeds) is impractical locally — plan a Colab/Lambda session or use
  the plan's fallback (Roboflow for training only, exported weights).
- Merged-bootstrap net transfer covers 59% of components; the rest
  are null nets the human must fill (expected — hosted-detector
  misses and low-IoU projections).
- Op-Amp terminal order (in+, in−, out) and BJT/MOSFET (C-B-E /
  D-G-S) conventions are asserted in the guide + netlist writer;
  the annotator must follow them for SPICE elements to be sensible.
---

## Plan v2 adopted (2026-07-18) — Benchmark + Design-Intent Completion

Superseded the phase plan with `IEEE_ACCESS_PLAN_v2_with_repair.md`
(30-day, MSP-vs-IDEAL). New contribution set C1–C5; the headline
addition is **C5 — transparent minimal design-intent completion**
(the ERC + assumption-ledger repair layer), the novelty hook. Two
Day-1 decisions: standardize on the **cleaned 512-px frame**; secure a
**GPU [HUMAN]** for training. Work below is the local, GPU-free slice
of Week 1.

### Done — Week 1 local build
- **BUILD-C0 — ERC + minimal assumption ledger (C5)** (`1cbd5e0c`):
  `erc.py` diagnoses simulability issues with DC-path reasoning
  (caps/current-sources/transistor-gates don't conduct at DC);
  `repair.py` applies minimal, logged fixes under a gauge (safe) vs
  assumption (flagged, with alternatives + confidence) taxonomy —
  one shunt per floating subnet, unsnapped terminals flagged never
  auto-wired. **Integrity rule enforced**: repair only adds SPICE
  lines, never changes topology (proven byte-identical on
  circuit_1199). Human-readable + JSON ledger, schema v1.
  `simulate.py` gained diagnostics (extracts the failing node names).
- **BUILD-B — GT benchmark harness** (`3fc202c6`): `benchmark.py`
  (+library) aligns pred→GT by IoU-within-class (Hungarian; unmatched
  penalize, not dropped), **canonicalizes terminal order by
  connectivity signature** so arbitrary 2-terminal indexing can't
  unfairly penalize, computes the full cascade (terminal-pair/net/
  per-component/nGED/strict-e2e + SPICE validity + solvability lift)
  with bootstrap 95% CIs. `--include-unverified` for provisional runs.
- **BUILD-A(local) — cleaned-frame dataset + detector eval**
  (`87de8024`): `make_yolo_dataset.py --frame cleaned` projects COCO
  boxes via `transforms.json` (verified visually on circuit_1);
  17,769/18,600 boxes kept (4.5% drop out-of-frame from the content
  crop). `eval_detector.py` ready to emit mAP + per-class AP+supports
  + confusion matrix once GPU weights exist.
- Test suite: **86 passing** (was 59; +15 repair/ERC/simulate,
  +12 benchmark).

### Remaining — needs GPU or [HUMAN]
- **[HUMAN] GPU**: train yolov8s@640 on the cleaned frame, 3 seeds
  (+n/m for E1); the M1 run does not survive a session exit.
- **[HUMAN] GT verification**: 190 files → `verified:true` (the
  benchmark scores verified-only by default).
- Provisional benchmark run over unverified GT is executing now to
  smoke-test the harness end-to-end (numbers are NOT paper numbers —
  they use unverified bootstrap GT and legacy hosted detections).
- M2 (crossover-aware net assembly [MSP], U-Net [IDEAL]); M3 (port
  localization); ablations; repair evaluation at scale; paper.

### Limitations / problems (new this session)
- Benchmark wall-clock: ~8 s/image (two ngspice calls + GED), ~25 min
  for 190. Fine as a one-off; consider caching / `--no-spice` for
  iteration.
- Cleaned-frame decision drops 4.5% of boxes that fall outside the
  content crop — documented, small, but must be stated in the paper.
- The repair layer's DC-conduction model is a documented ERC
  approximation (e.g. op-amp treated as weakly linking its terminals
  for reachability); good enough for diagnosis, stated as such.

### CGHD segmentation finding — the U-Net crux is NOT well-supported
Verified the plan's central M2 assumption ("CGHD ships segmentation
maps, train a wire-segmentation U-Net on those"). It does not hold:
- **Only 253 of 3,255 CGHD images (7.8%) have a segmentation map.**
- The maps are **binary foreground/ink masks** (background ~97%, ink
  ~3%), NOT wire-vs-component segmentation — they do not separate
  wires from symbols/text, which is the actual hard problem.
- They are cross-domain (some are P&ID-style, not standard circuits).

Consequence: the **U-Net wire-segmentation path (BUILD-C2, IDEAL) is
weakly supported and high-risk** — 253 cross-domain binary masks won't
train a robust wire-vs-symbol segmenter for Digitize-HCD. This is
exactly the risk the plan named ("if the U-Net fights, fall back").
The reliable M2 contribution is **BUILD-C1 crossover-aware net
assembly, already built and unit-tested**, which needs no segmentation
supervision (uses the detected Wire Crossover class). Recommend: treat
U-Net as a documented negative/limitation, not a headline; the C2
ablation becomes classical vs crossover-aware.

Silver lining for M3 (ports): `classes_ports.json` in CGHD gives
**canonical normalized port positions per class** (e.g. resistor
connectors at [0,0.5] and [1,0.5]) — a deterministic template prior
for pin localization needing no training, complementing Digitize-HCD's
per-component port heatmaps + XY coords. The M3 path is well-supported.

## Phase E — Ablations (axes prepared in config; not started)
## Phase F — Paper Rewrite (front-matter not started)
## Phase G — Pre-submission Verification (not started)
