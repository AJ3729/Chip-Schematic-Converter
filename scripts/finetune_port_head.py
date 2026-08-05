#!/usr/bin/env python3
"""Fine-tune the port heads on the frames the pipeline actually reads.

The heads trained on the published crops reach 0.87-0.90 order accuracy there
and collapse to 0.30 inside the pipeline, because the crops come from raw
photographs and the pipeline reads deskewed, downscaled `cleaned_1024` frames.
`scripts/build_indomain_ports.py` carries the published port labels onto those
frames; this fine-tunes on the result.

Small data (~1,600 samples over five classes), so it starts from the
published-crop checkpoint rather than from scratch, uses a low learning rate,
and validates on a held-out slice of the SAME domain -- which is the only
validation number that predicts what the pipeline will do.

Usage:
    python scripts/finetune_port_head.py --all --epochs 80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IN = ROOT / "data/port_indomain"
OUTDIR = ROOT / "experiments/port_head"

PORTS = {
    "BJT-NPN": ["Base", "Collector", "Emitter"],
    "BJT-PNP": ["Base", "Collector", "Emitter"],
    "MOSFET-N": ["Drain", "Gate", "Source"],
    "MOSFET-P": ["Drain", "Gate", "Source"],
    "Op-Amp": ["In+", "In-", "Out"],
}
PREFIX = {"BJT-NPN": "bjt_npn", "BJT-PNP": "bjt_pnp", "MOSFET-N": "mosfet_n",
          "MOSFET-P": "mosfet_p", "Op-Amp": "opamp"}


def build_net(k):
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True))
    return nn.Sequential(blk(1, 32), blk(32, 32), nn.MaxPool2d(2),
                         blk(32, 64), blk(64, 64), blk(64, 128), blk(128, 128),
                         nn.Conv2d(128, k, 1))


def heatmaps(coords, HS, sigma=None):
    if sigma is None:
        sigma = max(1.2, HS * 0.05)
    yy, xx = np.mgrid[0:HS, 0:HS].astype(np.float32)
    out = np.zeros((len(coords), HS, HS), np.float32)
    for i, (fx, fy) in enumerate(coords):
        out[i] = np.exp(-(((xx - fx * (HS - 1)) ** 2 + (yy - fy * (HS - 1)) ** 2)
                          / (2 * sigma ** 2)))
    return out


def augment(g, c, rng):
    k = rng.integers(0, 4)
    if k:
        g = np.rot90(g, k)
        for _ in range(k):
            c = np.stack([c[:, 1], 1.0 - c[:, 0]], 1)
    if rng.random() < 0.5:
        g = g[:, ::-1]
        c = np.stack([1.0 - c[:, 0], c[:, 1]], 1)
    g = g * rng.uniform(0.85, 1.2) + rng.uniform(-0.25, 0.25)
    if rng.random() < 0.3:
        g = g + rng.normal(0, 0.12, g.shape).astype(np.float32)
    return np.ascontiguousarray(g, np.float32), np.ascontiguousarray(c, np.float32)


def order_accuracy(hm, coords):
    from scipy.optimize import linear_sum_assignment
    HS = hm.shape[-1]
    ok = 0
    for h, c in zip(hm, coords):
        k = len(c)
        cost = np.zeros((k, k), np.float32)
        for p in range(k):
            for t, (fx, fy) in enumerate(c):
                x = int(np.clip(round(fx * (HS - 1)), 0, HS - 1))
                y = int(np.clip(round(fy * (HS - 1)), 0, HS - 1))
                cost[p, t] = -h[p, y, x]
        _, col = linear_sum_assignment(cost)
        ok += int((col == np.arange(k)).all())
    return ok / max(1, len(coords))


def run(cls, epochs, seed=0):
    f = IN / f"{cls}.npz"
    if not f.exists():
        print(f"  {cls}: no in-domain data"); return None
    d = np.load(f)
    X, Y = d["X"], d["Y"]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    idx = rng.permutation(len(X))
    nval = max(40, int(0.2 * len(X)))
    vi, ti = idx[:nval], idx[nval:]

    ck_path = OUTDIR / f"{PREFIX[cls]}.pt"
    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    HS = ck["HS"]
    net = build_net(len(PORTS[cls]))
    state = {k[4:] if k.startswith("net.") else k: v for k, v in ck["state"].items()}
    net.load_state_dict(state)
    net = net.to(dev)

    Xv = torch.from_numpy(X[vi][:, None]).float().to(dev)
    Yv = Y[vi]
    with torch.no_grad():
        pre = order_accuracy(net(Xv).cpu().numpy(), Yv)
    print(f"  {cls}: {len(X)} in-domain ({len(ti)} train / {nval} val)  "
          f"BEFORE fine-tune = {pre:.4f}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    best = {"acc": pre, "epoch": -1}
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(ti))
        for b in range(0, len(perm), 32):
            sel = ti[perm[b:b + 32]]
            gs, cs = zip(*(augment(X[j], Y[j], rng) for j in sel))
            xb = torch.from_numpy(np.stack(gs)[:, None]).float().to(dev)
            cb = torch.from_numpy(np.stack(cs)).to(dev)
            hb = torch.from_numpy(
                np.stack([heatmaps(c, HS) for c in cs])).to(dev)
            out = net(xb)
            B, K = cb.shape[:2]
            grid = torch.stack([(cb[..., 0] * 2 - 1).clamp(-1, 1),
                                (cb[..., 1] * 2 - 1).clamp(-1, 1)], -1).unsqueeze(2)
            logits = F.grid_sample(out, grid, align_corners=True).squeeze(-1).transpose(1, 2)
            tgt = torch.arange(K, device=dev).expand(B, K)
            loss = F.mse_loss(out, hb) + 0.5 * F.cross_entropy(
                logits.reshape(-1, K), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = order_accuracy(net(Xv).cpu().numpy(), Yv)
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": ep}
            torch.save({"state": net.state_dict(), "ports": PORTS[cls],
                        "S": ck["S"], "HS": HS, "MARGIN": ck["MARGIN"],
                        "finetuned_indomain": True},
                       OUTDIR / f"{PREFIX[cls]}_indomain.pt")
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    ep{ep:3d} val_order_acc={acc:.4f}"
                  f"{'  *' if acc == best['acc'] else ''}", flush=True)
    print(f"  {cls}: {pre:.4f} -> {best['acc']:.4f} (epoch {best['epoch']})", flush=True)
    return {"class": cls, "n": len(X), "before": pre, "after": best["acc"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    todo = list(PORTS) if a.all else [a.cls]
    t0 = time.time()
    res = [r for c in todo if (r := run(c, a.epochs))]
    (OUTDIR / "finetune_report.json").write_text(json.dumps(
        {"results": res, "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    print("\n  class        n   before -> after")
    for r in res:
        print(f"  {r['class']:10s} {r['n']:4d}   {r['before']:.4f} -> {r['after']:.4f}")


if __name__ == "__main__":
    main()
