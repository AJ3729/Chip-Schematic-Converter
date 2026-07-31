# Overnight GPU jobs — REDIRECTED by the Phase-0 oracles

Phase 0 measured the ceiling of each planned fix before spending GPU
time. Both assumptions failed, in opposite directions
(`results/comparisons/phase0_*.csv`):

- **Perfect text masking is worth ≈ nothing** (terminal-pair −0.007,
  strict −0.005 ns). The heuristic's misses are real (10.5% of boxes)
  but electrically benign, and box-masking trades phony wires for wire
  holes that stitching repairs. The text detector's BEST CASE is this
  null → **deprioritized to optional**.
- **Perfect crossover boxes made strict success WORSE** (−0.026, sig;
  0 gained / 5 lost, including images at terminal-pair F1 1.0000 →
  0.52). Drawn hops have an ink gap and are already electrically
  severed; notching them welds what the drafter kept apart. **The
  bottleneck is the notch surgery, not crossover detection.** A guard
  now skips boxes whose arms are already separate (nodes.py); two
  verification benchmarks are running.

## Job 1 (primary, if the 3090 is available): crossing classifier

Dataset is DONE: `data/crossings_synth` — 112,956 self-labeled patches
(train 32.8k crossover / 59.9k junction; val 7.4k / 12.8k) rendered over
real train/val layouts, exact electrical labels by construction, test
split never read. A CPU run is already training locally
(`experiments/junction/synth128`); the 3090 trains a stronger one in
minutes:

```bash
./venv/bin/python scripts/train_junction.py \
  --data data/crossings_synth --size 128 --epochs 30 --batch 256 \
  --device 0 --out experiments/junction/synth128_gpu
```

**Integration note:** the classifier's consumer will be the vector
tracer (continuation decisions), NOT notch surgery — Phase 0 showed
even perfect inputs to the notch path lose strict success.

## Job 2 (optional, only if the GPU is otherwise idle): 18-class text detector

Ceiling measured ≈ 0 on topology metrics, but the run is cheap and the
Text class is still useful for the demo/OCR later:

```bash
./venv/bin/python scripts/train.py \
  --data data/yolo_1024_text/dataset.yaml \
  --model yolov8s.pt --imgsz 640 --epochs 100 --batch 32 --device 0 \
  --project experiments/train_text/runs --name yolov8s_text_seed0
```

Do NOT expect benchmark gains from Job 2; the GT-text oracle bounds it.
