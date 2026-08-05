#!/usr/bin/env python3
"""Learn WHICH boundary crossing is which port, from the published port data.

The pipeline already finds where a component's leads cross its bounding box.
What it cannot do is name them: `ports.py` selects a pose from wire geometry
and never looks at the symbol, so it cannot read the arrowhead that marks a
BJT emitter, the +/- glyphs on an op-amp, or the gate bar on a MOSFET. Terminal
ORDER is therefore close to a coin flip on one axis, and `netlist.py` writes
`Q<c> <b> <e>` / `M<d> <g> <s>` / `E<out> 0 <in+> <in->` straight off that
order -- so a reversed transistor is emitted as a reversed transistor.

Digitize-HCD ships the supervision for this and the project was not using it:
per-class crops with NAMED port coordinates (Base/Collector/Emitter,
In+/In-/Out, Drain/Gate/Source). Those coordinates sit ON the component's box
boundary, which is exactly the quantity the pipeline computes.

PROVENANCE. Those crops were cut from the archive's 1,277 photographs, 192 of
which are this project's test split, and they carry no filename provenance. So
this trains only on the manifest produced by `scripts/port_provenance.py`,
which matched every crop against every COCO component region and kept a crop
only when it either matched a TRAIN image or matched nothing in the corpus at
all. See that script for why the "matched nothing" bucket is safe: the port
corpus is roughly 6x larger than the published diagram set, and matched crops
map to distinct instances 1:1, so the unmatched majority comes from
photographs this project never evaluates on.

Model: a small CNN over the box interior predicting one heatmap per named
port. At inference the heatmaps are sampled at the crossings the pipeline
already found and assigned by Hungarian matching, so the model decides ORDER
and never geometry -- which keeps the fix topology-neutral by construction.

Usage:
    python scripts/train_port_head.py --class Op-Amp --epochs 40
    python scripts/train_port_head.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

AR = ROOT / "data/digitize_hcd/extracted/Digitize-HCD Dataset"
PORT = AR / "Component Port Location Data"
MANIFEST = ROOT / "results/port_provenance/train_safe_crops.json"
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

S = 96          # network input side (set by --size)
HS = 48         # heatmap side
MARGIN = 0.12   # outward expansion of the box, so boundary ports sit inside


def green_box(im: np.ndarray):
    b, g, r = im[:, :, 0].astype(int), im[:, :, 1].astype(int), im[:, :, 2].astype(int)
    ys, xs = np.nonzero((g - np.maximum(b, r)) > 40)
    return None if len(xs) < 20 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def build_sample(path: Path, xy_path: Path, names: list[str]):
    """Crop to the annotated box (expanded), return image + port coords in it."""
    im = cv2.imread(str(path))
    if im is None or not xy_path.exists():
        return None
    bb = green_box(im)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * MARGIN), int(h * MARGIN)
    X0, Y0 = max(0, x0 - mx), max(0, y0 - my)
    X1, Y1 = min(im.shape[1], x1 + mx), min(im.shape[0], y1 + my)
    sub = im[Y0:Y1, X0:X1]
    if sub.size == 0 or X1 - X0 < 8 or Y1 - Y0 < 8:
        return None

    pts = {}
    for line in xy_path.read_text().strip().splitlines():
        parts = line.rsplit(maxsplit=2)
        if len(parts) != 3:
            continue
        nm, px, py = parts[0].strip(), float(parts[1]), float(parts[2])
        pts[nm] = ((px - X0) / (X1 - X0), (py - Y0) / (Y1 - Y0))
    if not all(n in pts for n in names):
        return None
    # the green annotation is burned into the pixels; grayscale + contrast
    # normalisation removes most of it and all of the colour-treatment variance
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g, np.array([pts[n] for n in names], np.float32)


def heatmaps(coords: np.ndarray, sigma: float | None = None) -> np.ndarray:
    # sigma MUST scale with the heatmap: fixed at 1.6 the target shrinks
    # relative to a larger map, the regression gets sparser, and a higher
    # input resolution scores WORSE than a lower one for reasons that have
    # nothing to do with resolution.
    if sigma is None:
        sigma = max(1.2, HS * 0.05)
    yy, xx = np.mgrid[0:HS, 0:HS].astype(np.float32)
    out = np.zeros((len(coords), HS, HS), np.float32)
    for i, (fx, fy) in enumerate(coords):
        cx, cy = fx * (HS - 1), fy * (HS - 1)
        out[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return out


class PortNet(nn.Module):
    """Small fully-convolutional heatmap regressor. 64x64 -> K x 32x32."""

    def __init__(self, k: int):
        super().__init__()
        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            blk(1, 32), blk(32, 32), nn.MaxPool2d(2),      # 32
            blk(32, 64), blk(64, 64),
            blk(64, 128), blk(128, 128),
            nn.Conv2d(128, k, 1),
        )

    def forward(self, x):
        return self.net(x)


def augment(g: np.ndarray, c: np.ndarray, rng):
    """Rotations/flips (with the coords), plus photometric jitter.

    Rotation matters more than usual here: the published crops come from
    photographs and the pipeline's crops come from deskewed 1024 frames, so
    the model has to be orientation-robust rather than memorise one pose.
    """
    k = rng.integers(0, 4)
    if k:
        g = np.rot90(g, k)
        for _ in range(k):
            c = np.stack([c[:, 1], 1.0 - c[:, 0]], 1)
    if rng.random() < 0.5:
        g = g[:, ::-1]
        c = np.stack([1.0 - c[:, 0], c[:, 1]], 1)
    g = g * rng.uniform(0.8, 1.25) + rng.uniform(-0.3, 0.3)
    if rng.random() < 0.3:
        g = g + rng.normal(0, 0.15, g.shape).astype(np.float32)
    return np.ascontiguousarray(g, np.float32), np.ascontiguousarray(c, np.float32)


def load_class(cls: str):
    man = json.loads(MANIFEST.read_text())
    keep = [r["crop"] for r in man["crops"][cls] if r["verdict"].startswith("KEEP")]
    names = PORTS[cls]
    xs, ys = [], []
    for nm in keep:
        stem = Path(nm).stem
        s = build_sample(PORT / cls / "Input Images" / nm,
                         PORT / cls / "XY Coordinates" / f"{stem}.txt", names)
        if s is not None:
            xs.append(s[0]); ys.append(s[1])
    return np.stack(xs), np.stack(ys)


def order_accuracy(pred_hm, coords) -> float:
    """The metric that matters: do the ports come out in the RIGHT ORDER?

    Localisation error is not the objective -- the pipeline already knows where
    the crossings are. What it needs is the assignment. So score exactly that:
    sample each predicted heatmap at every true port location and Hungarian-
    assign; correct iff the identity permutation wins.
    """
    from scipy.optimize import linear_sum_assignment
    ok = 0
    for hm, c in zip(pred_hm, coords):
        k = len(c)
        cost = np.zeros((k, k), np.float32)
        for pi in range(k):
            for ti, (fx, fy) in enumerate(c):
                x, y = int(round(fx * (HS - 1))), int(round(fy * (HS - 1)))
                x, y = np.clip(x, 0, HS - 1), np.clip(y, 0, HS - 1)
                cost[pi, ti] = -hm[pi, y, x]
        r, cidx = linear_sum_assignment(cost)
        ok += int((cidx == np.arange(k)).all())
    return ok / max(1, len(coords))


def train_one(cls: str, epochs: int, seed: int = 0) -> dict:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, Y = load_class(cls)
    n = len(X)
    idx = rng.permutation(n)
    nval = max(64, int(0.15 * n))
    vi, ti = idx[:nval], idx[nval:]
    print(f"  {cls}: {n} samples -> train {len(ti)} / val {nval}, device={dev}", flush=True)

    net = PortNet(len(PORTS[cls])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 3e-3, total_steps=epochs * max(1, len(ti) // 64 + 1))

    Xv = torch.from_numpy(X[vi][:, None]).to(dev)
    Yv = Y[vi]
    best = {"order_acc": -1.0}
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(ti))
        tot = 0.0
        for b in range(0, len(perm), 64):
            sel = ti[perm[b:b + 64]]
            gs, cs = [], []
            for j in sel:
                g, c = augment(X[j], Y[j], rng)
                gs.append(g); cs.append(c)
            xb = torch.from_numpy(np.stack(gs)[:, None]).to(dev)
            hb = torch.from_numpy(np.stack([heatmaps(c) for c in cs])).to(dev)
            out = net(xb)
            # MSE alone optimises LOCALISATION, but the pipeline already knows
            # where the crossings are -- what it needs is the ASSIGNMENT. On an
            # op-amp In+ and In- sit a few pixels apart, well inside one heatmap
            # sigma, so a purely generative target cannot separate them. The
            # cross-entropy term is the discriminative objective: at each true
            # port location the K channel responses must peak on the right port.
            cb = torch.from_numpy(np.stack(cs)).to(dev)          # B,K,2
            B, K = cb.shape[:2]
            gx = (cb[..., 0] * 2 - 1).clamp(-1, 1)
            gy = (cb[..., 1] * 2 - 1).clamp(-1, 1)
            grid = torch.stack([gx, gy], -1).unsqueeze(2)        # B,K,1,2
            sampled = F.grid_sample(out, grid, align_corners=True)  # B,K,K,1
            logits = sampled.squeeze(-1).transpose(1, 2)         # B, site, port
            tgt = torch.arange(K, device=dev).expand(B, K)
            loss = F.mse_loss(out, hb) + 0.5 * F.cross_entropy(
                logits.reshape(-1, K), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss.detach()) * len(sel)
        net.eval()
        with torch.no_grad():
            pv = net(Xv).cpu().numpy()
        acc = order_accuracy(pv, Yv)
        if acc > best["order_acc"]:
            best = {"order_acc": acc, "epoch": ep, "loss": tot / len(ti)}
            torch.save({"state": net.state_dict(), "ports": PORTS[cls],
                        "S": S, "HS": HS, "MARGIN": MARGIN},
                       OUTDIR / f"{PREFIX[cls]}.pt")
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    ep{ep:3d} loss={tot/len(ti):.5f} val_order_acc={acc:.4f}"
                  f"{'  *' if acc==best['order_acc'] else ''}", flush=True)
    print(f"  {cls}: BEST val order accuracy = {best['order_acc']:.4f} "
          f"(epoch {best['epoch']})", flush=True)
    return {"class": cls, "n": n, "n_val": nval, **best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=None,
                    help="network input side; the +/- glyph that separates an "
                         "op-amp's inputs is only a few pixels wide, so this is "
                         "the resolution knob that matters most")
    args = ap.parse_args()

    if args.size:
        globals()["S"] = args.size
        globals()["HS"] = args.size // 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    todo = list(PORTS) if args.all else [args.cls]
    res = []
    t0 = time.time()
    for c in todo:
        res.append(train_one(c, args.epochs))
    (OUTDIR / "training_report.json").write_text(json.dumps(
        {"results": res, "seconds": round(time.time() - t0, 1),
         "manifest": str(MANIFEST)}, indent=1))
    print(f"\nwrote {OUTDIR}/training_report.json  ({time.time()-t0:.0f}s)")
    for r in res:
        print(f"  {r['class']:10s} n={r['n']:5d}  val order acc = {r['order_acc']:.4f}")


if __name__ == "__main__":
    main()
