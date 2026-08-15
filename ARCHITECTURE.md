# Architecture

One section per pipeline stage, in execution order. Everything runs inside
`schematic2netlist.pipeline.run_pipeline`, which is the single entry point; the
CLI wrappers under `scripts/` only choose inputs and where output goes.

Frozen configuration and its hash: `FROZEN_CONFIG.md`.
Data formats and per-circuit artifact inventory: `DATA.md`.

---

## 0. Entry point

| | |
| --- | --- |
| module | `src/schematic2netlist/pipeline.py` |
| entry | `run_pipeline(image_path, cfg, detections=None) -> dict` |
| CLI | `scripts/run_pipeline.py`, `scripts/benchmark.py` |

Passing `detections=` skips the detector and uses a supplied cache. Every
experiment in the repository runs that way so results are reproducible; the
user-facing latency figure is the scope where the detector actually runs. Both
scopes are measured separately (`scripts/measure_runtime.py`).

`run_pipeline` returns a dict whose keys are the stage outputs listed below.
`determinism.py` hashes every key except `image` and `out_dir` — that is what
makes the byte-identical claim checkable.

---

## 1. Preprocessing (offline, not in `run_pipeline`)

| | |
| --- | --- |
| module | `preprocess.py`, guarded by `frames.py` |
| entry | `scripts/preprocess.py`, `scripts/record_transforms.py` |
| config | `preprocess.*` (target_size, shadow_*, speck_*, hough_*, max_skew_deg) |
| input | `data/raw/*.jpg` — the published photographs |
| output | `data/cleaned_1024/*.jpg` + `data/transforms_1024.json` |

Perspective rectification and shadow normalisation to a 1024×1024 frame. The
forward transform is recorded per image so any annotation can be projected
between photograph and frame coordinates, and back.

**Frame-size guard.** `frames.resolve_and_check` refuses to run when the frames
on disk are not `preprocess.target_size`. Before this existed, a 1024 config
run against 512 frames scored the wrong pixels silently, and because detection
boxes live in frame coordinates the component alignment corrupted too.

---

## 2. Component detection

| | |
| --- | --- |
| module | `detect.py` |
| entry | `detect(image_path, cfg)` / `detect_ultralytics(paths, cfg)` |
| config | `detect.weights`, `detect.confidence`, `detect.image_size`, `detect.cache_dir` |
| input | 1024 frame |
| output | list of `{class, x, y, width, height, confidence}` in frame coords |
| cache | `data/detections_valstop/<stem>.json` |

YOLOv8s over 17 classes. `detect()` returns the cache whenever one exists —
which is why `--time-detector` in the deprecated `scripts/benchmark_runtime.py`
timed a JSON read rather than inference.

## 3. Component class head

| | |
| --- | --- |
| module | `class_head.py` |
| entry | `reclassify(dets, gray, cfg)` — mutates `dets` in place |
| config | `detect.class_head.{enabled,weights,threshold,ensemble}` |

The detector localises well and classifies less well, and the two are
separable. A second independently seeded head is averaged with the first.

The threshold is a **risk** setting, not a tuning knob: strict success is a
product over components, so one wrong relabel destroys a circuit that was
already correct. Relabelling everything loses; the gate only acts where the
head is confident.

Because this runs *inside* `run_pipeline`, the on-disk detection cache holds
pre-correction classes. Anything reading the cache directly (the VLM task
builder, the scorer) must apply the head too or it silently scores a different
class than the benchmark does.

## 4. Text suppression

| | |
| --- | --- |
| module | `textmask.py` |
| config | `textmask.*` |
| output | a boolean mask of handwritten-label ink |

Value labels and node names are ink, and ink that is not conductor will be
traced as conductor if it is not removed first.

## 5. Conductor extraction

| | |
| --- | --- |
| module | `wires.py`, `skeleton.py` |
| config | `wires.*` (binarize, bridge_span, stitch_*, component_mask_pad, min_blob_*) |
| output | wire mask, skeleton, conductor segments |

Adaptive binarisation (Sauvola), text-mask subtraction, morphological cleanup,
skeletonisation. Detected component boxes are notched out, padded by
`component_mask_pad`, so a symbol body cannot short its own terminals. Stitching
reconnects segments broken by the notch or by a pen lift.

## 6. Node graph and intersection resolution

| | |
| --- | --- |
| module | `nodes.py`, `junction_model.py`, `vector_nodes.py` |
| config | `nodes.*` (connectivity, handle_crossovers, junction_*) |
| output | `node_map` — a labelled connected-component map over conductor pixels |

Each intersection is adjudicated join or cross. A detected **Wire Crossover**
symbol (the semicircular hop) is evidence of non-connection. This stage decides
the circuit's topology, and it is where the pipeline's remaining errors
concentrate.

## 7. Terminal snapping and port identity

| | |
| --- | --- |
| module | `snapping.py`, `ports.py`, `port_head.py` |
| config | `snapping.max_expand`, `expand_step`, `window_depth`, `snapping.port_head.*` |
| output | per component, an ordered list of node ids |

Snapping expands a window around each detection until the class's expected
number of boundary crossings is found — that fixes **which nets** a component
touches.

`ports.py` then assigns *which pin is which* from lead geometry alone, which
cannot read an arrowhead. `port_head.py` adds a small per-class heatmap CNN
that re-permutes the pin list by Hungarian assignment on window-max heatmap
response.

**The invariant that makes this safe:** the port head can only permute a list.
The set of nets is fixed by snapping and passed through untouched, so no
topology metric can move — verified bit-identical with the head on and off.

## 8. Constraint-triggered connectivity repair

| | |
| --- | --- |
| module | `connectivity_repair.py` |
| config | `connectivity_repair.*` |

Fires only when a structural constraint is violated (a component shorted
through its own body, a one-terminal net). Runs before netlist export and is
part of reconstruction, **not** part of the design-intent repair in stage 10.

## 9. Netlist export

| | |
| --- | --- |
| module | `netlist.py`, `erc.py` |
| entry | `export_spice_netlist(...)` |
| config | `netlist.ground_fallback`, `netlist.placeholders` |
| output | SPICE deck |

Cards are written off the **ordered** terminal list — `Q<c> <b> <e>`,
`M<d> <g> <s> <s>`, `E<out> 0 <in+> <in->`. This is why pin order is a
correctness property and not a cosmetic one.

No OCR: component values are uniform placeholders from
`netlist.placeholders`. "Simulatable" therefore means structurally correct
under identical placeholder values, not numerically matching the drawn circuit.

## 10. Design-intent repair

| | |
| --- | --- |
| module | `repair.py`, `repair_eval.py` |
| config | `repair.{enabled,max_assumptions,strategies,shunt_r}` |
| output | repaired deck + a ledger of every intervention |

Attempts to make an unsolvable deck solvable — add a ground reference, tie a
floating node — and **counts and logs every intervention**. Every topology
metric is computed before this stage runs; repair never touches strict success.

## 11. Simulation

| | |
| --- | --- |
| module | `simulate.py` |
| config | `simulate.{ngspice_binary,timeout_s}` |
| output | `(ok, category, diagnostics)`; categories include `floating_node`, `singular_matrix`, `timeout`, `parse_error` |

---

## Evaluation modules (not pipeline stages)

| module | role |
| --- | --- |
| `benchmark.py` | the metric cascade: Hungarian component matching at IoU 0.3 within class, net correspondence, terminal-pair / net / per-component / nGED / strict success, bootstrap CIs |
| `metrics.py` | metric primitives |
| `gt.py` | ground-truth loading and `gt_to_components` |
| `splits.py` | split-role convention — `val` selects, `test` reports |
| `determinism.py` | global seeding and run metadata |
| `stats/` | McNemar, bootstrap, Holm, kappa (task A3) |

**`benchmark.canonicalize_terminals` sorts terminals by a connectivity
signature computed identically on both sides.** That is deliberate — net names
are arbitrary — but it is also exactly why a pin swap is invisible to every
topology metric, and why the pin-aware scorer (task D2) is needed.
