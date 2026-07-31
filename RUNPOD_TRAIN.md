# Crossing classifier on RunPod (Jupyter) — copy/paste runbook

Trains the all-degree wire-crossing classifier. Everything needed is in
one tarball: a packed dataset (`.npz`, loads in seconds) and the training
script. No repo checkout, no project install on the pod.

**Target signal:** `balanced_acc` past **0.90** with `crossover_recall`
healthy. Watch recall, not accuracy — junctions outnumber crossings ~2:1,
so a model that never splits anything still posts high plain accuracy.
That trap has already cost this project a debugging session.

> **Always train into a fresh `--out`, and verify before you ship.**
> `best.pt` and `val_probs.npy` are written whenever validation improves;
> `summary.json` is written once at the end. A second run into the same
> directory overwrites the weights at its own epoch 1 (`best` resets, so
> epoch 1 always "improves") while `summary.json` still describes the
> finished run. Interrupt it and the tarball carries a finished run's
> metrics next to a half-trained run's weights.
>
> This already happened on the v5 run: it returned `summary.json` saying
> balanced accuracy **0.8019** at epoch 57, alongside weights worth
> **0.6746** — roughly epoch 4 of a fresh run. A transfer evaluation ran
> on the wrong model before the mismatch was caught. The trainer now
> refuses a non-empty `--out` (use `--force` to override) and stamps its
> metrics into `best.pt`, and
> `scripts/check_junction_checkpoint.py` catches it in seconds.

---

## 1. On your Mac — build the tarball

```bash
cd ~/Documents/Chip-Schematic-Converter
./venv/bin/python scripts/pack_crossing_dataset.py \
  --data data/crossings_v3 --out /tmp/crossings_v3.npz
mkdir -p /tmp/xtrain && cp /tmp/crossings_v3.npz scripts/train_junction.py /tmp/xtrain/
tar -czf ~/Desktop/xtrain.tgz -C /tmp xtrain
ls -lh ~/Desktop/xtrain.tgz
```

---

## 2. Start the pod

RunPod → Deploy → **RTX 5090** → template **RunPod PyTorch 2.x with CUDA
12.8+** (Blackwell needs it; see Cell 2). 20 GB volume is plenty. When it is running, click **Connect →
Jupyter Lab** (port 8888).

---

## 3. Upload through Jupyter

In Jupyter Lab's left-hand file browser, navigate to `/workspace`, then
drag `xtrain.tgz` from your Desktop into the file list. Wait for the
upload progress bar to finish.

Then **File → New → Notebook** (Python 3) and run the cells below.

---

## 4. Notebook cells

**Cell 1 — unpack and check the GPU**

```python
%cd /workspace
!tar -xzf xtrain.tgz
%cd /workspace/xtrain
!ls -la
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

**Cell 2 — install cv2 AND verify the GPU is actually usable**

`train_junction.py` imports cv2 at module level (only the PNG path uses
it, but the import still has to resolve).

The second half matters on a **5090**: it is Blackwell (compute
capability **sm_120**), which older PyTorch builds cannot emit kernels
for. A CUDA 11.8 / 12.1 template will import torch fine, report the GPU
by name, and then fail at the first conv with *"no kernel image is
available for execution on the device"*. Check before spending time:

```python
!pip install -q opencv-python-headless
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability sm_%d%d" % torch.cuda.get_device_capability(0))
print("arch list", torch.cuda.get_arch_list())
# the real test: run an actual conv on the device
import torch.nn as nn
try:
    nn.Conv2d(1, 8, 3).cuda()(torch.zeros(2, 1, 32, 32, device="cuda"))
    print("OK — conv executes on the GPU")
except Exception as e:
    print("FAILED:", type(e).__name__, e)
```

`sm_120` must appear in `arch list`. If the conv fails, upgrade in place
and restart the kernel:

```python
!pip install -q --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# Kernel -> Restart Kernel, then re-run Cell 1 and this cell
```

**Cell 3 — train**

Runs in the foreground so you see per-epoch output live. The Jupyter
kernel keeps it alive; if your browser disconnects, reconnect and the
output resumes.

```python
!python train_junction.py \
  --data crossings_v3.npz --size 128 \
  --epochs 60 --batch 256 --lr 3e-4 --device cuda \
  --out out_synth128
```

On a 5090 expect roughly **10 s/epoch**, so ~10 minutes for 60 epochs.
One line per epoch with `balanced_acc`, `crossover_recall`,
`crossover_precision`. The model is only 72k parameters; the bottleneck
is CPU-side batch assembly, not the GPU.

**Because it is this cheap, do not stop at one run.** Spend the surplus
on epochs, seeds, and patch size — not on a bigger network. Inference
runs on the Mac's CPU at ~20 intersections per image, so network capacity
is charged to every future benchmark run, while training time is paid
once:

```python
for size, ep, lr in [(128, 100, 3e-4), (128, 100, 1e-4), (96, 100, 3e-4)]:
    !python train_junction.py --data crossings_v3.npz --size {size} \
      --epochs {ep} --lr {lr} --batch 256 --device cuda \
      --out out_s{size}_lr{lr}
```

(96 px needs a dataset packed at that size — `pack_crossing_dataset.py
--size 96` — so run that variant only if you packed one.) Then pick the
checkpoint with the best validation `balanced_acc` and send that one.

**Cell 3-alt — if you would rather close the browser**

Detach it, then poll from another cell whenever you like:

```python
!nohup python train_junction.py --data crossings_v3.npz --size 128 \
  --epochs 60 --batch 256 --lr 3e-4 --device cuda --out out_synth128 \
  > train.log 2>&1 &
print("started")
```

```python
!tail -20 train.log
```

**Cell 4 — inspect the result before downloading**

```python
import json, pathlib
p = pathlib.Path("out_synth128")
print(sorted(x.name for x in p.iterdir()))
h = json.loads((p / "history.json").read_text())
best = max(h, key=lambda r: r["balanced_acc"])
print("best epoch:", best)
```

**Cell 5 — package the output for download**

```python
!tar -czf /workspace/xtrain_out.tgz -C /workspace/xtrain out_synth128
!ls -lh /workspace/xtrain_out.tgz
```

Then in the Jupyter file browser, navigate to `/workspace`, right-click
`xtrain_out.tgz` → **Download**.

---

## 5. Back on your Mac

```bash
cd ~/Documents/Chip-Schematic-Converter
mkdir -p experiments/junction
tar -xzf ~/Downloads/xtrain_out.tgz -C /tmp
cp -r /tmp/out_synth128 experiments/junction/synth128_gpu
ls -la experiments/junction/synth128_gpu/
```

**Verify the artefacts agree before evaluating** — this is the check that
catches a mismatched `best.pt` / `summary.json` pair, and it costs seconds
against the ~10 minutes a transfer evaluation takes on the wrong model:

```bash
./venv/bin/python scripts/check_junction_checkpoint.py \
  --run experiments/junction/synth128_gpu --data /tmp/crossings_v5.npz
```

It must print `ARTEFACTS CONSISTENT`. If check 2 fails, the weights are
not the model `summary.json` describes — retrain into a fresh `--out`
rather than trusting either number.

Only then run the measurement that actually decides anything:

```bash
./venv/bin/python scripts/eval_crossing_transfer.py \
  --weights experiments/junction/synth128_gpu/best.pt --limit 60
```

Then **stop the pod** in the RunPod console so it stops billing.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| kernel dies, or the cell ends with no output and no error | out of memory. Use `--batch 128`. (This bit us locally: the loader used to build float32 arrays needing 8.6 GB and was OOM-killed silently — now uint8, but batch size can still do it.) |
| `ModuleNotFoundError: cv2` | run Cell 2 |
| `balanced_acc` stuck at exactly 0.5000 | degenerate one-class model. Try `--lr 1e-4`. Never use MPS for this model — it fails this way silently. |
| `Invalid device string: '0'` | use `--device cuda`. `--device 0` is the Ultralytics spelling (scripts/train.py wants it) but raw PyTorch does not accept it. The trainer now converts it, but older copies of the script do not. |
| `--size` mismatch error | the message prints the packed patch size; pass that value |
| `crossover_recall` near 0 while accuracy looks high | class imbalance won. The loss is already class-weighted, so report back rather than shipping it. |
| upload is slow | the tarball is one `.npz`, not ~150k PNGs, precisely to avoid this. If it is still slow, check you are uploading the `.tgz` and not the source directory. |

## What to send back

`experiments/junction/synth128_gpu/` with `best.pt`, `history.json`, and
whatever summary the script wrote. Integration reads `best.pt` through
`nodes.junction_weights`.
