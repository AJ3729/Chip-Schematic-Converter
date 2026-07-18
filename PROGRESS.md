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

## Phase B — Data: Splits, Annotations, Ground Truth 🔶 (in progress, 2026-07-18)

### Done
- **B3 GT tooling** (`9ad8ab43`): schema v1 for GT topology graphs
  (`data/gt_netlists/<stem>.json`: components + terminal→net mapping,
  ground net "0", verified/annotator fields), loader + strict
  validator, and `scripts/annotate_topology.py` with the
  bootstrap → correct → render → check workflow. Smoke-tested on
  circuit_1199. 12 new tests (44 total).
- **B1 acquisition in progress**: Digitize-HCD downloading from
  Mendeley (doi 10.17632/rngcz5wtv8; single 2.1 GB zip, published
  sha256 to verify against). Confirmed v1 and v2 reference the same
  zip file — the "v2 adds heatmaps" assumption from the plan is
  checked against the zip contents below.
- **B4 scoped**: CGHD is Zenodo record 10056817, single 3.4 GB zip,
  CC-BY-4.0; download queued after Digitize-HCD.

### Remaining (this phase)
- Verify zip sha256, extract, inspect annotation formats (COCO boxes,
  text polygons, port heatmaps?).
- Hash-reconcile the download against local `data/raw` and
  `data/cleaned`; document provenance + counts in `data/README.md`.
- `scripts/make_splits.py`: stratified 70/15/15 splits (component
  count × class presence; drafter-disjoint if metadata exists) and
  commit the frozen manifests.
- Bootstrap GT for the full test split (needs detections — see
  problem below).
- CGHD class-mapping table + zero-shot subset.
- **[HUMAN]** GT verification pass (~1–2 min/image × ~190 test images):
  correct each bootstrapped GT JSON, then set `verified: true` +
  `annotator`. The tool renders overlays to make this fast.

### Limitations / problems
- **Detection-cache bottleneck**: bootstrapping GT for the test split
  requires per-image detections, but only circuit_1199 is cached. The
  options are (a) train the local YOLO first (Phase C) and bootstrap
  from it, or (b) batch-call the hosted Roboflow model on the ~190
  test images. (b) preserves the plan's ordering (GT before training)
  and needs only the existing ROBOFLOW_API_KEY.
- **Annotation/preprocessing coordinate mismatch**: published
  Digitize-HCD annotations are in original-image coordinates, but
  `data/cleaned` images went through rotate/crop/resize that was not
  recorded. Training/eval on published annotations must either run on
  raw images or re-run preprocessing with transforms recorded.
  `scripts/preprocess.py` does not yet store its transform matrix —
  needs a small extension before Phase C training.
- Disk: ~18 GB free vs ~2.1+3.4 GB zips plus extractions; manageable
  but CGHD extraction should be selective (subset only).

---

## Phase C — Models & Baselines (not started)
## Phase D — Evaluation Harness vs GT (metrics implemented + tested in A; harness wiring not started)
## Phase E — Ablations (axes prepared in config; not started)
## Phase F — Paper Rewrite (not started)
## Phase G — Pre-submission Verification (not started)
