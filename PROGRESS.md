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
- **B4 CGHD**: upstream `classes.json` fetched (53 entries, not the
  59 the plan said); `data/cghd/class_mapping.yaml` maps CGHD → the
  17 published Digitize-HCD categories with lossy mappings marked;
  I-DC, I-AC, and V-DC (one port) have no CGHD counterpart. 3.4 GB
  zip (CC BY 4.0) downloading; zero-shot subset extraction pending.

### Remaining (this phase)
- **[HUMAN] GT verification pass** — the single most valuable human
  task in the project (~1–2 min/image × 190): for each
  `data/gt_netlists/<stem>.json`, check the render in
  `data/gt_netlists/renders/`, correct classes/nets/missed components,
  then set `verified: true` and `annotator`. Re-run
  `annotate_topology.py --render/--check` while editing.
- CGHD zip: verify md5, selectively extract annotations + a subset of
  images, build the zero-shot evaluation list (finishes when the
  background download lands).

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

## Phase C — Models & Baselines (not started)
## Phase D — Evaluation Harness vs GT (metrics implemented + tested in A; harness wiring not started)
## Phase E — Ablations (axes prepared in config; not started)
## Phase F — Paper Rewrite (not started)
## Phase G — Pre-submission Verification (not started)
