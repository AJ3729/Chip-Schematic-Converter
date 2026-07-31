#!/usr/bin/env python3
"""Train the junction-vs-crossover patch classifier (M2 / C2).

At every place two strokes meet, net assembly must decide whether they
CONNECT or merely CROSS. Thresholds cannot make that call — sweeping
every stitching guard bought ~+0.02 terminal-pair F1 against the +0.47
the oracle attributes to connectivity — so the decision is learned from
CGHD's `junction` and `crossover` annotations.

The model is deliberately small (a few hundred thousand parameters on
64x64 binary patches). It runs at every wire intersection of every
image, so inference cost is the binding constraint, not capacity; and
the task is local and geometric rather than semantic.

Three things this script refuses to fudge:

- **Class imbalance is handled, not hidden.** Junctions outnumber
  crossovers ~13:1. Loss is class-weighted, and the reported headline
  is balanced accuracy plus per-class recall — plain accuracy would
  read 93% for a model that never predicts `crossover` at all.
- **Validation is drafter-disjoint** (enforced upstream in
  build_junction_dataset.py), so the score measures transfer to unseen
  handwriting rather than memorised strokes.
- **The operating point is chosen, not assumed.** Net assembly pays
  asymmetric costs for the two errors: calling a crossover a junction
  welds two nets together, which is usually worse than splitting one.
  The script sweeps the decision threshold and reports the trade-off
  so the pipeline can pick deliberately.

Usage (CPU smoke test):
    python scripts/train_junction.py --data data/junctions --epochs 2

Usage (GPU):
    python scripts/train_junction.py --data data/junctions_full \
        --epochs 30 --device cuda --out experiments/junction/run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ("junction", "crossover")     # index 1 = crossover = positive class


def load_split(root: Path, split: str, size: int):
    import cv2

    X, y = [], []
    for label, cls in enumerate(CLASSES):
        for p in sorted((root / split / cls).glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if im.shape[0] != size:
                im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
            X.append(im)
            y.append(label)
    if not X:
        raise SystemExit(f"no patches under {root/split}")
    # Keep patches as uint8 and scale per BATCH, not here. Materializing
    # float32 costs 4x: the 128-px synthetic set (92,689 train + 20,267
    # val patches) needs 8.6 GB as float32 and was silently OOM-killed on
    # a 16 GB machine, leaving an empty output directory and no error.
    # uint8 brings the same set to 2.1 GB.
    return np.stack(X), np.array(y, dtype=np.int64)


def build_model(size: int):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 16), block(16, 32), block(32, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(64, 2),
    )


def metrics_at(probs: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    rec_cross = tp / (tp + fn) if tp + fn else 0.0
    rec_junc = tn / (tn + fp) if tn + fp else 0.0
    prec_cross = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": round(thr, 3),
        "balanced_acc": round((rec_cross + rec_junc) / 2, 4),
        "crossover_recall": round(rec_cross, 4),
        "crossover_precision": round(prec_cross, 4),
        "junction_recall": round(rec_junc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/junctions")
    ap.add_argument("--out", default="experiments/junction/run1")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an --out that already holds best.pt")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # MPS is deliberately NOT auto-selected. On this model it trains to a
    # degenerate classifier — balanced accuracy pinned at exactly 0.5000,
    # predicting all one class while the loss falls — whereas identical
    # code on CPU reaches ~0.90. The failure is silent and looks like a
    # data problem, which cost a debugging session, so it must be opted
    # into explicitly.
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Accept the Ultralytics-style device spec ("0", "1", ...) as well as
    # PyTorch's own. scripts/train.py takes --device 0 because that is what
    # Ultralytics wants, so passing the same flag here is the natural
    # mistake, and torch answers it with a bare "Invalid device string: '0'"
    # AFTER loading the whole dataset.
    if dev.isdigit():
        dev = f"cuda:{dev}"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device} requested but CUDA is not "
                         f"available (torch {torch.__version__}, "
                         f"cuda {torch.version.cuda})")
    if dev == "mps":
        print("[WARN] MPS produces a degenerate model here (see comment); "
              "CPU or CUDA strongly recommended", flush=True)
    root = Path(args.data)
    out = Path(args.out)
    if (out / "best.pt").exists() and not args.force:
        raise SystemExit(
            f"{out}/best.pt already exists. A second run here overwrites the "
            f"weights the moment its first epoch 'improves' on nothing, while "
            f"summary.json keeps describing the OLD run until this one "
            f"finishes — so an interrupted re-run ships a finished run's "
            f"metrics next to a half-trained run's weights. Use a fresh "
            f"--out, or --force if you really mean to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    # A packed .npz (scripts/pack_crossing_dataset.py) loads in seconds
    # where ~150k individual PNGs take minutes — the difference matters on
    # a rented GPU, where startup time is billed.
    if root.suffix == ".npz":
        z = np.load(root, allow_pickle=False)
        Xtr, ytr = z["X_train"], z["y_train"]
        Xva, yva = z["X_val"], z["y_val"]
        if Xtr.shape[1] != args.size:
            raise SystemExit(
                f"{root} holds {Xtr.shape[1]}px patches but --size is "
                f"{args.size}; pass --size {Xtr.shape[1]}")
    else:
        Xtr, ytr = load_split(root, "train", args.size)
        Xva, yva = load_split(root, "val", args.size)
    # load_split returns patches grouped by class; shuffle validation so
    # that any subsampling downstream stays class-representative rather
    # than silently scoring one class (a mistake this project has made).
    vperm = np.random.default_rng(args.seed).permutation(len(yva))
    Xva, yva = Xva[vperm], yva[vperm]
    print(f"device {dev} | train {len(ytr)} ({int((ytr==1).sum())} crossover) "
          f"| val {len(yva)} ({int((yva==1).sum())} crossover)")

    # class weights invert the imbalance so the rare class is not ignored
    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=dev)
    print(f"class weights (junction, crossover): "
          f"{weights[0]:.2f}, {weights[1]:.2f}")

    model = build_model(args.size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=weights)

    # uint8 on the host; scaled to float per batch on the device
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva).unsqueeze(1)

    best = {"balanced_acc": -1.0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr_t))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = Xtr_t[idx].to(dev).float().div_(255.0)
            yb = ytr_t[idx].to(dev)
            # augmentation: circuits are drawn at any orientation, and a
            # junction stays a junction under flips and 90-degree turns
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[3])
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                xb = torch.rot90(xb, k, dims=[2, 3])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xva_t), 512):
                vb = Xva_t[i:i + 512].to(dev).float().div_(255.0)
                chunks.append(torch.softmax(model(vb), dim=1)[:, 1].cpu())
            probs = torch.cat(chunks).numpy()
        m = metrics_at(probs, yva, 0.5)
        m["epoch"] = epoch
        m["train_loss"] = round(total / len(ytr_t), 4)
        history.append(m)
        print(f"epoch {epoch:3d}  loss {m['train_loss']:.4f}  "
              f"balanced acc {m['balanced_acc']:.4f}  "
              f"crossover recall {m['crossover_recall']:.4f}  "
              f"junction recall {m['junction_recall']:.4f}", flush=True)
        if m["balanced_acc"] > best["balanced_acc"]:
            best = dict(m)
            # Stamp the metrics INTO the checkpoint. best.pt, val_probs.npy
            # and summary.json are written at three different moments, so a
            # second run into the same --out silently leaves a mismatched
            # trio: a finished run's summary next to a half-trained run's
            # weights. That happened on the v5 pod run and cost a full
            # transfer evaluation on the wrong model, which read as "the
            # data did not transfer" when the 0.80 weights were never
            # tested at all. Anything reading best.pt can now check.
            torch.save({"state_dict": model.state_dict(), "size": args.size,
                        "classes": CLASSES, "epoch": epoch,
                        "val_metrics": dict(m), "data": str(root),
                        "seed": args.seed}, out / "best.pt")
            np.save(out / "val_probs.npy", probs)

    probs = np.load(out / "val_probs.npy")
    sweep = [metrics_at(probs, yva, t) for t in np.arange(0.1, 0.95, 0.05)]
    summary = {
        "device": dev, "params": n_params, "epochs": args.epochs,
        "data": str(root), "seed": args.seed,
        "train_counts": {c: int((ytr == i).sum()) for i, c in enumerate(CLASSES)},
        "val_counts": {c: int((yva == i).sum()) for i, c in enumerate(CLASSES)},
        "best_at_0.5": best, "threshold_sweep": sweep, "history": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nbest balanced accuracy {best['balanced_acc']:.4f} "
          f"(epoch {best['epoch']}), {n_params} params")
    print("\noperating points — net assembly pays more for a false JUNCTION "
          "(welds two nets) than a false crossover (splits one):")
    print(f"  {'thr':>5s} {'bal acc':>8s} {'cross rec':>10s} "
          f"{'cross prec':>11s} {'junc rec':>9s}")
    for s in sweep:
        print(f"  {s['threshold']:5.2f} {s['balanced_acc']:8.4f} "
              f"{s['crossover_recall']:10.4f} {s['crossover_precision']:11.4f} "
              f"{s['junction_recall']:9.4f}")
    print(f"\nwrote {out}/best.pt + summary.json")


if __name__ == "__main__":
    main()
