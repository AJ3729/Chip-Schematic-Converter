# Data: schemas, manifests, and per-circuit inventory

What is stored, in what format, and where. `ARCHITECTURE.md` covers the code;
this covers the artifacts. Anything a task needs and cannot find is listed
under **Missing artifacts** at the end.

---

## 1. Split manifests

Frozen and versioned, because they are part of the benchmark rather than
derived from it.

| manifest | n | sha256 |
| --- | --- | --- |
| `data/splits/train.txt` | 895 | `8c96f53bc49ac95e600a7c72098947e8482dea046f3246e0713597021cf2beb3` |
| `data/splits/val.txt` | 190 | `e2e065c1463b9dbc4a14972daace604040c481de860065feb56bb453e99136a3` |
| `data/splits/test.txt` | 192 | `35fc06791eda8be9a6b5a27f655e4aecf9f1f0de44d3f6532e9c9521774ef37f` |

**`val` selects, `test` reports.** The two exchanged names on 2026-08-03; every
`results/` artifact committed before that date is a validation number whatever
its metadata says. Mapping: `data/splits/splits_meta.json` → `role_swap`.

---

## 2. Ground-truth netlist schema

`data/gt_test_1024/<stem>.json` (192 files), `data/gt_val_1024/` (190).

```json
{
  "schema_version": 1,
  "image": "circuit_1013.jpg",
  "source": "coco_geometry+manual_topology",
  "verified": true,
  "annotator": "...",
  "verified_via": "...",
  "notes": "...",
  "bbox_frame": "cleaned_1024",
  "components": [
    {"id": 0,
     "class": "Resistor",
     "bbox": [403.2, 382.0, 119.8, 34.4],
     "terminals": [{"index": 0, "net": "n1"},
                   {"index": 1, "net": "n2"}]}
  ]
}
```

* `bbox` is `[center_x, center_y, width, height]` in **1024-frame** coordinates.
* `terminals` is **ordered**; `index` is the position in the class's canonical
  port order. This ordering is what the pin-aware metric (D2) scores and what
  the published topology metrics canonicalise away.
* `net` is an arbitrary label. Only the grouping is meaningful, except that the
  net a GND symbol touches is always exactly `"0"`.

### Decision records

`data/gt_test_1024/decisions/<stem>.json` (192 files):

```json
{"sites": {"0": "junction", "13": "crossing", ...},
 "notes": "free-text account of the circuit and every judgement call"}
```

`sites` maps intersection-site id to one of `junction`, `crossing`,
`edge_group`, `none`. Across the test split: 2,047 sites adjudicated — 1,708
junction, 194 crossing, 62 edge group, 83 none.

### The as-drawn rule

Ground truth records the topology **visibly drawn**. A floating node stays
floating, a missing ground stays missing, a shorted source is recorded as
shorted. Any intervention that would make a circuit easier to simulate is
recorded separately from topology and never folded into it.

---

## 3. CGHD annotation schema (target for the C1 tool)

New annotations are written to `data/cghd/annotations/incoming/<stem>.json` and
validated by `scripts/sync_board.py`. The schema extends the Digitize-HCD one
with the fields the plan requires:

```json
{
  "schema_version": 1,
  "image": "<cghd stem>.jpg",
  "source": "cghd_geometry+manual_topology",
  "drafter": "<cghd drafter id>",
  "drawing_group": "<id shared by every photo of one physical drawing>",
  "annotator": "<name>",
  "annotation_seconds": 812,
  "pass": 1,
  "components": [
    {"id": 0, "class": "Resistor", "bbox": [cx, cy, w, h],
     "terminals": [{"index": 0, "net": "n1"}, {"index": 1, "net": "n2"}],
     "allow_self_short": false}
  ],
  "sites": [{"id": 0, "xy": [512, 300], "kind": "junction"}],
  "interventions": [
    {"type": "assumed_ground", "target": "n3",
     "note": "no GND symbol drawn; the bottom rail is clearly the return"}
  ]
}
```

Three fields carry the plan's requirements and must not be dropped:

* **`interventions`** is the as-drawn rule made machine-readable. Repairs the
  annotator *would* apply go here, never into `components`. Task D8 scores the
  pipeline's declared repairs against this field.
* **`drawing_group`** enables the capture-invariance experiment (B7): several
  photographs of one physical drawing need exactly one ground-truth netlist.
* **`annotation_seconds`** lets the paper report annotation cost, and
  `pass` distinguishes the double-annotated subset for self-agreement (E4).

### Validation performed on ingest

Schema conformance; every terminal assigned to exactly one net; no component
shorted through its own body unless `allow_self_short` is set explicitly; every
site adjudicated to one of the four kinds; component and net counts sane
relative to detector output. **Sanity checks flag, they never auto-correct** —
an annotation that disagrees with a prediction is a finding, not an error.

---

## 4. Per-circuit inventory (test split, 192 circuits)

| artifact | path | coverage |
| --- | --- | --- |
| reference topology | `data/gt_test_1024/<stem>.json` | 192/192 |
| decision record | `data/gt_test_1024/decisions/<stem>.json` | 192/192 |
| detection cache | `data/detections_valstop/<stem>.json` | 192/192 |
| predicted + reference decks | `results/final/op_agreement/netlists/<stem>.{pred,gt}.sp` | 192/192 (384 files) |
| per-node simulation output | `results/final/op_agreement/cache/<stem>.json` | 192/192 |
| repair ledger | `results/final/benchmark/seed0/ledgers/<stem>.json` | 192/192 |

Repair ledger shape: `{schema_version, image, solvable_before, solvable_after,
num_assumptions, num_gauge, entries}` where `entries` is the per-intervention
list that task D7 classifies.

### The 494 perturbation swaps

Stored in `results/final/op_agreement/summary.json` under
`acceptance.<policy>.controlled_gt_perturbation`, and in full in
`results/final/op_agreement/acceptance_test.json`.

**Detection rates are already recorded per swap kind** — `by_swap_kind` carries
`n_swaps`, `detected`, `detection_rate`, `undetected_because_terminals_
equipotential`, and `detection_rate_on_detectable` for each of
`bjt_collector_emitter`, `diode_reversal`, `mosfet_drain_source`,
`opamp_input_swap`, `idc_direction`, `iac_direction`. Task D3's first bullet is
therefore computable from stored artifacts with no re-simulation.

The passive control (665 swaps on resistors, capacitors, inductors, detection
rate exactly 0) is in the same block.

---

## 5. CGHD, as it exists on disk

| | |
| --- | --- |
| archive | `data/cghd/cghd-zenodo-12.zip`, 3.43 GB |
| class table | `data/cghd/cghd_classes.json`, **53 classes** |
| draft mapping | `data/cghd/class_mapping.yaml` (predates this plan; B2 supersedes it) |
| extracted subset | `data/cghd/subset/` |
| annotation inbox | `data/cghd/annotations/{incoming,accepted,rejected}/` |

The archive name records Zenodo version 12. Task B1 must confirm whether a
newer release exists rather than assuming this is current, and record the exact
version, DOI, release date and license in `data/cghd/PROVENANCE.md`.

CGHD ships 53 classes against Digitize-HCD's 17, so a mapping to
`{in-vocabulary, OUT_OF_VOCABULARY, AMBIGUOUS}` is required before any circuit
can be fairly scored (task B2). Circuits containing out-of-vocabulary
components are excluded from the evaluable pool, with the exclusion counted and
reported.

---

## 6. Missing artifacts

Recorded here so that tasks needing them are BLOCKED against a named path
rather than silently skipped. Current state from `scripts/sync_board.py`:

| artifact | blocks | who produces it |
| --- | --- | --- |
| `spec/pin_symmetry.yaml` | D2, D6 | author — template ready at `reports/pending_review/pin_symmetry_template.yaml` |
| `data/cghd/annotations/normalized_interventions.json` | D8 | author, after annotation |
| `data/cghd/annotations/accepted/*.json` | E2, E3 | author, via the C1 tool |
| `data/cghd/annotations/double/*.json` | E4 | author, double-annotation pass |
| `spec/qualitative_circuit.txt` | F2 | author, after reviewing D6's residual set |
| `data/cghd/PROVENANCE.md` | — | task B1 |
| `spec/class_map_cghd.yaml` | — | task B2 |

Nothing in this table blocks more than the tasks named beside it. 24 tasks are
READY at the time of writing.
