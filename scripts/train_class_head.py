#!/usr/bin/env python3
"""Train a dedicated component-class head, and compare it against the detector.

The comparison is the whole point. A previous attempt at this lost to the
detector on every group, so an accuracy number in isolation proves nothing: the
model is scored on the SAME test components the detector labels, and reported
head to head, overall and on the confusable pairs that carry the error.

Design follows the diagnosis of why the earlier attempt failed -- 72k parameters
at 64 px, against confusions decided by a few pixels of arrow. Here the input is
128 px (about 3.5x the linear detail the detector sees on a 60 px component at
image_size 640) and the network is a compact residual stack of ~1.4M parameters.

Two choices are not free parameters and should not be "tuned away":

  rotation only, never reflection.  A mirrored MOSFET-N IS a MOSFET-P. Flipping
                                    teaches the model the two are the same class.
  class-balanced sampling.          The confusable classes are also the rarest
                                    (MOSFET-N 223, MOSFET-P 226 against Resistor
                                    2240), so uniform sampling spends its
                                    capacity where there is no error.

Early stopping is on the 192-image VALIDATION split, which exists for
classification even though it does not for net topology -- so this is a genuine
train/val/test protocol rather than the test-set tuning the rest of the project
is forced into.

Usage:
    python scripts/train_class_head.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFUSABLE = [("MOSFET-N", "MOSFET-P"), ("BJT-NPN", "BJT-PNP"),
              ("Inductor", "Resistor"), ("I-DC", "I-AC"),
              ("V-DC", "V-AC"), ("Diode", "Zener Diode")]


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential() if stride == 1 and cin == cout else
                   nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                 nn.BatchNorm2d(cout)))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))


class Net(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 5, 2, 2, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.MaxPool2d(2))
        self.s1 = Block(32, 64, 2)
        self.s2 = Block(64, 128, 2)
        self.s3 = Block(128, 192, 2)
        self.head = nn.Linear(192, n_cls)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.stem(x)
        x = self.s3(self.s2(self.s1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.drop(x))


def augment(batch: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Rotations and mild photometric jitter. NO reflections -- see module docs."""
    k = int(torch.randint(0, 4, (1,), generator=gen).item())
    if k:
        batch = torch.rot90(batch, k, dims=(2, 3))
    if torch.rand(1, generator=gen).item() < 0.7:
        ang = (torch.rand(1, generator=gen).item() * 2 - 1) * 12.0
        th = torch.tensor(ang * np.pi / 180.0)
        c, s = torch.cos(th), torch.sin(th)
        m = torch.tensor([[c, -s, 0.0], [s, c, 0.0]],
                         dtype=batch.dtype).unsqueeze(0).repeat(len(batch), 1, 1)
        grid = F.affine_grid(m, batch.shape, align_corners=False)
        batch = F.grid_sample(batch, grid, align_corners=False,
                              padding_mode="border")
    if torch.rand(1, generator=gen).item() < 0.5:
        batch = batch * (0.85 + 0.3 * torch.rand(1, generator=gen).item())
    return batch.clamp(0, 1)


def load(split, root):
    d = np.load(root / f"{split}.npz", allow_pickle=True)
    return d["X"], d["y"]


def evaluate(model, X, y, dev, bs=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().div(255).unsqueeze(1).to(dev)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/class_crops")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="experiments/class_head")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = ROOT / args.data
    names = json.loads((root / "summary.json").read_text())["names"]
    Xtr, ytr = load("train", root)
    Xva, yva = load("val", root)
    Xte, yte = load("test", root)
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  "
          f"{len(names)} classes\n")

    dev = torch.device(args.device if (args.device != "mps"
                                       or torch.backends.mps.is_available())
                       else "cpu")
    model = Net(len(names)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model {n_par/1e6:.2f}M parameters on {dev}\n")

    # class-balanced sampling: the confusable classes are the rarest
    cnt = Counter(ytr.tolist())
    w = np.array([1.0 / cnt[int(c)] for c in ytr])
    w = w / w.sum()
    gen = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps)

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        idx = np.random.choice(len(Xtr), size=steps * args.batch, p=w)
        tot = 0.0
        for i in range(steps):
            b = idx[i * args.batch:(i + 1) * args.batch]
            xb = torch.from_numpy(Xtr[b]).float().div(255).unsqueeze(1)
            xb = augment(xb, gen).to(dev)
            yb = torch.from_numpy(ytr[b]).to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        pv = evaluate(model, Xva, yva, dev)
        # balanced accuracy: the rare confusable classes must not be drowned out
        accs = [float((pv[yva == c] == c).mean())
                for c in range(len(names)) if (yva == c).any()]
        bal = float(np.mean(accs))
        flag = ""
        if bal > best:
            best, best_state, bad = bal, {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}, 0
            flag = "  <- best"
        else:
            bad += 1
        print(f"  epoch {ep+1:3d}/{args.epochs}  loss {tot/steps:.4f}  "
              f"val balanced acc {bal:.4f}{flag}", flush=True)
        if bad >= 10:
            print("  early stop")
            break

    model.load_state_dict(best_state)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "names": names,
                "val_balanced_acc": best, "size": 128}, out / "best.pt")

    pte = evaluate(model, Xte, yte, dev)
    overall = float((pte == yte).mean())
    per = {names[c]: float((pte[yte == c] == c).mean())
           for c in range(len(names)) if (yte == c).any()}
    bal_te = float(np.mean(list(per.values())))
    print(f"\n=== TEST ({len(yte)} crops) ===")
    print(f"  overall accuracy   {overall:.4f}")
    print(f"  balanced accuracy  {bal_te:.4f}")
    print(f"\n  {'class':22s} {'n':>5s} {'acc':>7s}")
    for k, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"  {k:22s} {int((yte == names.index(k)).sum()):5d} {v:7.4f}")
    print(f"\n  CONFUSABLE PAIRS (where the strict-success error lives):")
    for a, b in CONFUSABLE:
        if a not in names or b not in names:
            continue
        ia, ib = names.index(a), names.index(b)
        m = (yte == ia) | (yte == ib)
        if not m.any():
            continue
        acc = float((pte[m] == yte[m]).mean())
        print(f"    {a:14s} vs {b:14s} n={int(m.sum()):4d}  acc {acc:.4f}")
    (out / "test_report.json").write_text(json.dumps({
        "overall": overall, "balanced": bal_te, "per_class": per,
        "val_balanced_acc": best, "n_params": n_par,
    }, indent=2) + "\n")
    print(f"\nwrote {out}/best.pt and test_report.json")
    print(f"\nNEXT: score the pipeline's own detections with this head, price the "
          f"relabels with\nscripts/audit_relabels.py, and only then spend a "
          f"benchmark. An accuracy number here\nis not evidence until it beats "
          f"the DETECTOR on the same components.")


if __name__ == "__main__":
    main()
