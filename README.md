# schematic2netlist

Hand-drawn schematics are how circuits are first designed, taught, and
discussed, yet they are disconnected from modern simulation tools —
recreating each sketch in EDA software is slow and error-prone. This
project bridges that gap: a machine-learning-based system that
interprets photographed handwritten circuit schematics and generates
simulation-ready digital netlists, aiming to accelerate prototyping and
make hardware design more accessible to students and hobbyists.

Concretely, it is a deterministic, fully local pipeline that converts
photographs of **hand-drawn analog circuit schematics** into **SPICE
netlists**:

```
photo ─► preprocess ─► component detection ─► text masking ─► wire
extraction ─► node inference (connected components) ─► terminal
snapping ─► SPICE netlist ─► ngspice validation
```

Connectivity is treated **topologically rather than geometrically**:
wire pixels are grouped into connected components (one component = one
electrical node), and component terminals are snapped to nodes with
configurable strategies. No OCR is performed; component values are
placeholders.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .            # core pipeline
pip install -e '.[dev]'     # + pytest
pip install -e '.[train]'   # + ultralytics (local YOLO training/inference)
pip install -e '.[roboflow]'  # + hosted Roboflow fallback (needs ROBOFLOW_API_KEY in .env)
```

`ngspice` must be on PATH for simulation checks (`brew install ngspice`
on macOS).

## Usage

Run the full pipeline on one image (uses the per-image detection cache
in `data/detections/`, falling back to the configured detector backend):

```bash
python scripts/run_pipeline.py --image data/cleaned/circuit_1199.jpg
```

Artifacts (wire masks, overlays, `netlist.sp`, `netlist_readable.txt`,
`run_meta.json`) are written to `experiments/runs/<image-stem>/`.

Batch-evaluate a directory of images:

```bash
python scripts/evaluate.py --images-dir data/cleaned --limit 100
```

Preprocess raw photos, train a detector, run an ablation axis:

```bash
python scripts/preprocess.py --raw-dir data/raw --clean-dir data/cleaned
python scripts/train.py --data data/dataset.yaml --model yolov8s.pt
python scripts/ablate.py --axis wires.min_blob_area --values 10,20,40,60,100
```

## Configuration

Every pipeline threshold lives in [configs/default.yaml](configs/default.yaml)
— nothing is hardcoded. The two legacy pipeline variants survive as
config choices (`snapping.strategy: directional|uniform`,
`netlist.ground_fallback: most_connected|fail`, `wires.min_blob_area`),
which are ablation axes for the experimental program.

## Repository layout

```
configs/default.yaml        every threshold, documented
src/schematic2netlist/      the installable package
  preprocess.py             deskew/shadow/binarize/crop/resize
  detect.py                 local Ultralytics (batch) / Roboflow / cached
  textmask.py               heuristic text masking (ablation axis)
  wires.py                  non-wire masking + wire extraction
  nodes.py                  connected-component node inference
  snapping.py               BOTH terminal-snapping strategies (v1/v2)
  netlist.py                node naming + SPICE export
  simulate.py               ngspice runner + failure taxonomy
  metrics.py                coverage stats + GT metrics (pair F1, net F1, nGED)
  pipeline.py               per-image orchestration
  determinism.py            seeding + run metadata (config, git SHA, env)
scripts/                    thin CLIs: run_pipeline, evaluate, preprocess, train, ablate
tests/                      pytest unit tests (netlist writer, metrics, ngspice parser)
data/                       (gitignored) raw/, cleaned/, detections/, splits/, gt_netlists/
experiments/                (gitignored) runs, evaluations, legacy artifacts
```

## Data

The image corpus is [Digitize-HCD](https://doi.org/10.17632/rngcz5wtv8)
(Mendeley Data, CC BY 4.0). `data/` is not versioned; splits, provenance
and annotation formats will be documented in `data/README.md` as part of
the benchmark release.

## Reproducibility

- Each run directory contains `run_meta.json`: full config, git SHA,
  seed, and environment versions.
- `seed` in the config seeds `random`, `numpy`, and `torch`/Ultralytics.
- Evaluation numbers are only ever produced by `scripts/evaluate.py`
  from committed code + frozen splits — never hand-typed.

## License

MIT — see [LICENSE](LICENSE).
