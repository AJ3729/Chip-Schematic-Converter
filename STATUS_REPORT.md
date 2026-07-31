# Status Report — IEEE Access Submission

**Point-in-time snapshot. Generated 2026-07-23. Not committed (per request).**
Running log lives in [PROGRESS.md](PROGRESS.md); this is a focused "where
are we / what's left / what you must do" summary.

**Active plan:** `IEEE_ACCESS_PLAN_v2_with_repair.md` (30-day,
MSP-vs-IDEAL, contributions C1–C5 with the new C5 design-intent-repair
hook). This supersedes all earlier plans.

**One-line status:** Every task that can be done **without a GPU and
without your ground-truth verification is built and tested** (91 tests
passing). The project is now blocked on two [HUMAN] gates — a GPU and
your GT verification pass — which unblock everything downstream.

---

## 1. Where we are in the plan

| Plan arc | Status |
| --- | --- |
| Phase A (repo foundation) | ✅ done (earlier sessions) |
| Phase B (data / splits / GT tooling) | ✅ code-complete; GT verification pending (you) |
| **Week 1 — finish M1, benchmark harness, ERC/ledger, start GT, paper front-matter** | 🔶 **local parts done; detector training (GPU) + GT (you) + paper front-matter pending** |
| Week 2 — M2 wires, headline ablation, repair taxonomy, Method draft | 🔶 crossover-aware wires **pulled forward + done**; rest pending |
| Week 3 — M3 ports, full experiments, repair evaluation | ⬜ not started |
| Week 4 — finish paper, verify, submit | ⬜ not started |

We are **~through the GPU-free portion of Week 1**, with the Week-2
crossover work already banked. Progress on the critical path now
requires the two gates below.

---

## 2. What is accomplished — exactly

### By contribution (the C1–C5 the paper rests on)

| Contribution | State | Evidence |
| --- | --- | --- |
| **C1 — benchmark on Digitize-HCD** | Harness ✅, splits ✅, code ✅; **verified GT ✗ (you)** | `scripts/benchmark.py`, `src/schematic2netlist/benchmark.py`, frozen `data/splits/` |
| **C2 — learned/structural connectivity** | MSP (crossover-aware) ✅ built + tested; U-Net (IDEAL) ✗ (found unsupported — see §5) | `nodes.build_wire_nodes_crossover_aware`, `tests/test_crossover.py` |
| **C3 — port/pin localization** | ⬜ not started; supervision confirmed available | Digitize-HCD port heatmaps + CGHD `classes_ports.json` |
| **C4 — metric standardization + oracle** | Metrics ✅ (cascade + CIs); oracle waterfall + runtime ⬜ | `metrics.py`, `benchmark.py`; `scripts/oracle.py` not yet written |
| **C5 — design-intent repair (novelty)** | Skeleton ✅ built + tested; full study (IDEAL) ✗ (you + mentor) | `erc.py`, `repair.py`, `tests/test_repair.py`, ledger schema v1 |

### This session's commits (6 focused commits, `12269151`→`79d2863a`)

- `1cbd5e0c` **BUILD-C0 — ERC + assumption ledger (C5).** Diagnoses why a
  recovered circuit won't simulate (DC-path reasoning), applies
  minimal, logged repairs under a gauge-vs-assumption taxonomy, emits a
  human-readable ledger. Integrity rule proven: repair never changes
  topology.
- `3fc202c6` **BUILD-B — GT benchmark harness.** Full metric cascade
  (terminal-pair/net/per-component/nGED/strict-end-to-end + SPICE
  validity + solvability-lift) with bootstrap 95% CIs; fair pred↔GT
  alignment + terminal-order canonicalization.
- `87de8024` **BUILD-A(local) — cleaned-frame dataset + detector eval.**
  COCO boxes projected into the 512-px frame (visually verified);
  `eval_detector.py` ready for GPU weights.
- `3aaadea4` **BUILD-C1 — crossover-aware net assembly (M2 MSP).** The
  deterministic wire-crossing fix: notch + reconnect opposite arms at
  detected `Wire Crossover` boxes so crossing wires stay separate.
- `f3376eb7`, `79d2863a` — PROGRESS.md updates (plan adoption + CGHD
  finding).

**Tests:** 59 → **91 passing** (+15 repair/ERC/simulate, +12 benchmark,
+5 crossover).

### Also delivered earlier this session (not code)

- `docs/GT_VERIFICATION_GUIDE.md` — your full how-to for GT verification.
- `docs/examples/circuit_1_worked_example.json` + render — a 100%-correct
  worked GT example (uncommitted, for your reference).

---

## 3. What remains

### 3a. What I can do **without you** (once a GPU exists, or now)

- **Oracle / GT-injection attribution** (`scripts/oracle.py`, C4) — can
  build the harness now; needs verified GT + trained detector to produce
  numbers.
- **Ablation runner wiring** (`scripts/ablate.py` × the metric cascade) —
  can build now; numbers need GT + detector.
- **Repair evaluation at scale** (`scripts/benchmark_repair.py`) —
  solvability-lift, assumptions/circuit, gauge-accuracy, topology-
  preservation proof. Build now; numbers need verified GT.
- **M3 port localization (MSP)** — deterministic pin templates from CGHD
  `classes_ports.json` + Digitize-HCD XY coords; buildable now.
- **Paper scaffolding** — LaTeX template, figures-from-CSV notebook. (I
  have **not** started drafting prose — see §6.)

### 3b. What is **blocked on a GPU**

- Train the local YOLOv8s detector on the cleaned frame, 3 seeds (+n/m
  for the size ablation). **This is the linchpin:** the hosted Roboflow
  model has no `Wire Crossover` class, so the crossover fix and the real
  benchmark numbers only become meaningful after local training.
- Regenerate `data/detections/` for all splits from local weights.
- Detector baselines (v5/v10/v11/RT-DETR) — IDEAL.

### 3c. What is **blocked on your GT verification**

- Every real benchmark number (C1), the oracle attribution (C4), and the
  repair evaluation (C5). The harness currently scores verified GT only;
  the run finishing now uses *unverified* bootstrap GT and is a
  smoke-test, **not** paper numbers.

---

## 4. ⭐ What YOU have to do — prioritized

These are the `[HUMAN]` gates. The first two unblock the entire critical
path; front-load the rest so Week 4 never stalls.

### Priority 1 — Secure a GPU (blocks all training)
A cloud/university GPU for the project window (Colab Pro, Lambda, a lab
machine). The M1 laptop cannot train the detector + models with iteration
in time, and a training run does not survive my session ending. **This one
fact unblocks the most work.** ~1 hour to arrange; then I drive the rest.

### Priority 2 — Ground-truth verification (blocks all benchmark science)
Verify the 190 test-split topology files, following
[docs/GT_VERIFICATION_GUIDE.md](docs/GT_VERIFICATION_GUIDE.md).
- **Time:** 6–8 h across 5–6 sittings.
- **Where:** edit `data/gt_netlists/circuit_*.json`, check against
  `data/gt_netlists/renders/`, set `"verified": true` + your name.
- **Floor:** ≥120 verified is acceptable if honestly reported; 190 is the
  target.
- **Skill:** reading sophomore-level analog schematics + tracing wires.
  Nothing ML. See the worked example in `docs/examples/`.

### Priority 3 — Administrative (front-load; needed to submit)
- **ORCID** registered and public (both authors, if two).
- **Authorship decision** per IEEE's three criteria (you; mentor Alex
  Whitehead only if he meets all three, else Acknowledgments).
- **Book your mentor** as: (a) GT second-reader for ambiguous
  crossings/transistors, and (b) [IDEAL] the C5 expert-acceptance study
  — rate ~30 repair ledgers as correct-minimal / over-repaired / wrong
  (~1–2 h). This study is what elevates C5 from "mechanism" to
  "validated" and is high-leverage.
- **APC awareness:** $2,160 on acceptance; confirm funding.
- **iThenticate access** (institutional) for the similarity check.

### Priority 4 — Two decisions I need from you now
- **Push to GitHub?** I have 6 unpushed commits. Say the word and I push
  (I won't without your go-ahead).
- **Should I start drafting the paper?** I deliberately held off — the
  manuscript is your scholarly voice. I can scaffold the template + draft
  Intro/Related Work/Dataset for your revision, or wait. Your call.

---

## 5. Key findings that changed the plan

- **The U-Net wire-segmentation crux is not supported by the data.**
  I verified CGHD's segmentation: only 253/3,255 images (7.8%) have one,
  and they are **binary ink/paper masks, not wire-vs-component**
  (some aren't even circuits). A robust wire segmenter can't be trained
  on that. **Consequence:** the reliable M2 contribution is the
  crossover-aware net assembly (already built); the U-Net becomes a
  documented limitation, not a headline. This is exactly the risk the
  plan named — now resolved before wasting a week on it.
- **Silver lining for M3:** CGHD `classes_ports.json` gives canonical
  per-class pin positions — a deterministic port prior needing no
  training, complementing Digitize-HCD's port heatmaps.
- **Cleaned-frame standardization** drops 4.5% of annotation boxes that
  fall outside the tight content crop — small, documented, must be
  stated in the paper.

---

## 6. Honest risks / open issues

- **Timeline is GPU-gated.** Until Priority 1 is resolved, the critical
  path (train → real detections → real benchmark → ablations → paper
  numbers) cannot advance. The 30-day clock effectively pauses here.
- **Provisional benchmark numbers are not science.** The run finishing
  now uses unverified GT + legacy hosted detections; treat its output as
  a harness smoke-test only.
- **Benchmark is slow** (~8 s/image; GED dominates on large graphs).
  Fine as a one-off; will add caching before iterative runs.
- **Repair ERC uses a documented DC-conduction approximation** (e.g.
  op-amp terminals treated as weakly linked for reachability). Adequate
  for diagnosis; stated as such.
- **I have not drafted paper prose** — awaiting your decision (§4).
