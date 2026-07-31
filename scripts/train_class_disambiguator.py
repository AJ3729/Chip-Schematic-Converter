#!/usr/bin/env python3
"""Fine-grained relabelling of the component classes the detector confuses.

This is where "train on real annotated data" actually pays. The crossing
version of that idea is closed: perfect GT crossover boxes make strict success
WORSE (0.3263 against 0.3526), perfect per-box decisions give 0.0 headroom,
and no model on 4822 real causally-labelled sites clears 0.70 precision. But
20% of the test images cannot reach strict success at all because a GT
component is unmatched, and 76% of those are CLASS CONFUSION rather than
missed detection -- the box is present and localizes to IoU >= 0.3, only the
label is wrong.

The confusions are near-symmetric visual pairs differing in one detail:

  MOSFET-N / MOSFET-P    arrow direction on the body connection
  Resistor / Inductor    zigzag against coils
  I-DC / I-AC            straight line against a sine inside the circle
  BJT-NPN / BJT-PNP      arrow direction on the emitter
  Diode / Zener Diode    straight cathode bar against a bent one
  V-DC / V-AC            bars against a sine

Unlike the crossing task this is well posed. The input is the component symbol
crop rather than an ambiguous patch of wire; the classes carry thousands of
real human-labelled boxes on the TRAIN split; and training on the same
preprocessed 1024-px frames the pipeline consumes removes the mask-domain
shift that sank the render-trained classifiers. A 17-way detector spends its
capacity separating resistors from op-amps; a per-group model only has to
answer the one question the detector gets wrong.

One model per confusable GROUP, trained on train, model selected on val,
reported on test. Groups are kept separate rather than pooled into one 17-way
head because the decision at inference is conditional: the detector has
already narrowed the class to a group, and the only question is which member.

Usage:
    python scripts/train_class_disambiguator.py --extract
    python scripts/train_class_disambiguator.py --train --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.classes import canonical_class
from schematic2netlist.preprocess import project_bbox

COCO = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
        "Component Symbol and Text Label Data/component_annotations.json")

# Groups the DETECTOR actually confuses, from results/blockers/strict_blockers.json
GROUPS = {
    "mosfet": ["MOSFET-N", "MOSFET-P"],
    "rl": ["Resistor", "Inductor"],
    "isrc": ["I-DC", "I-AC"],
    "bjt": ["BJT-NPN", "BJT-PNP"],
    "diode": ["Diode", "Zener Diode"],
    "vsrc": ["V-DC", "V-AC"],
}
SIZE = 64
PAD_FRAC = 0.12


def build_model(size: int, n_out: int):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
    return nn.Sequential(
        block(1, 16), block(16, 32), block(32, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(64, n_out),
    )


def extract(out_dir: Path):
    coco = json.loads(Path(COCO).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    tf = json.loads(Path("data/transforms_1024.json").read_text())

    split_of = {}
    for sp in ("train", "val", "test"):
        for l in open(f"data/splits/{sp}.txt"):
            if l.strip():
                split_of[l.strip()] = sp

    member_group = {}
    for gname, members in GROUPS.items():
        for i, m in enumerate(members):
            member_group[m] = (gname, i)

    buf = defaultdict(list)      # (group, split) -> [(patch, label)]
    counts = Counter()
    for img_id, file_name in img_by_id.items():
        sp = split_of.get(file_name)
        stem = Path(file_name).stem
        meta = tf.get(stem)
        if sp is None or meta is None:
            continue
        frame = cv2.imread(f"data/cleaned_1024/{file_name}", cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        H, W = frame.shape
        for a in anns.get(img_id, []):
            cls = canonical_class(cats[a["category_id"]])
            if cls not in member_group:
                continue
            gname, lab = member_group[cls]
            cx, cy, bw, bh = project_bbox(meta, *a["bbox"])
            pad = PAD_FRAC * max(bw, bh)
            x1 = int(max(0, cx - bw / 2 - pad)); x2 = int(min(W, cx + bw / 2 + pad))
            y1 = int(max(0, cy - bh / 2 - pad)); y2 = int(min(H, cy + bh / 2 + pad))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = frame[y1:y2, x1:x2]
            patch = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            buf[(gname, sp)].append((patch, lab))
            counts[(gname, sp, cls)] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for gname in GROUPS:
        payload = {}
        for sp in ("train", "val", "test"):
            items = buf.get((gname, sp), [])
            if not items:
                continue
            payload[f"X_{sp}"] = np.stack([p for p, _ in items]).astype(np.uint8)
            payload[f"y_{sp}"] = np.array([l for _, l in items], dtype=np.int64)
        if "X_train" not in payload:
            continue
        np.savez_compressed(out_dir / f"{gname}.npz", **payload)
        print(f"  {gname:8s} " + "  ".join(
            f"{sp}={len(buf.get((gname,sp),[]))}" for sp in ("train","val","test")))
    print(f"\nper-class counts:")
    for g in GROUPS:
        for cls in GROUPS[g]:
            row = "  ".join(f"{sp}={counts[(g,sp,cls)]}"
                            for sp in ("train", "val", "test"))
            print(f"  {g:8s} {cls:16s} {row}")
    print(f"\nwrote {out_dir}")


def train_group(path: Path, epochs: int, lr: float, batch: int, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    z = np.load(path)
    Xtr, ytr = z["X_train"], z["y_train"]
    Xva = z["X_val"] if "X_val" in z else Xtr[:0]
    yva = z["y_val"] if "y_val" in z else ytr[:0]
    Xte = z["X_test"] if "X_test" in z else Xtr[:0]
    yte = z["y_test"] if "y_test" in z else ytr[:0]
    if len(yva) < 10 or len(yte) < 5:
        return None

    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                           dtype=torch.float32)
    model = build_model(SIZE, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=weights)

    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1)
    ytr_t = torch.from_numpy(ytr)

    def evaluate(X, y):
        model.eval()
        with torch.no_grad():
            pr = []
            for i in range(0, len(y), 512):
                xb = torch.from_numpy(X[i:i+512]).unsqueeze(1).float().div_(255.0)
                pr.append(model(xb).argmax(1).numpy())
        pred = np.concatenate(pr) if pr else np.zeros(0, int)
        # balanced accuracy: the groups are imbalanced (3186 Resistor vs 2206
        # Inductor), and plain accuracy would reward always guessing the
        # majority member -- the exact trap the junction work already hit
        accs = []
        for c in (0, 1):
            m = y == c
            if m.sum():
                accs.append((pred[m] == c).mean())
        return float(np.mean(accs)) if accs else 0.0, pred

    best = (-1.0, None)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(perm), batch):
            idx = perm[i:i+batch]
            xb = Xtr_t[idx].float().div_(255.0)
            yb = ytr_t[idx]
            # flips only: a MOSFET's N/P difference is an ARROW DIRECTION, so
            # a horizontal flip does not change the class but a 90-degree
            # rotation composed with a flip can look like the other member.
            # Rotations by 90 are safe (the symbol is drawn at any
            # orientation); reflections are NOT, and augmenting with them
            # would teach the model to ignore the only cue that matters.
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                xb = torch.rot90(xb, k, dims=[2, 3])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        va, _ = evaluate(Xva, yva)
        if va > best[0]:
            best = (va, {k: v.clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    te, pred = evaluate(Xte, yte)
    cm = np.zeros((2, 2), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    return {"val_balanced_acc": round(best[0], 4),
            "test_balanced_acc": round(te, 4),
            "n_train": int(len(ytr)), "n_val": int(len(yva)),
            "n_test": int(len(yte)), "confusion": cm.tolist(),
            "state": best[1]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data-dir", default="data/class_groups")
    ap.add_argument("--out-dir", default="experiments/class_disambiguator")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data_dir)
    if args.extract:
        print("extracting component crops from cleaned_1024 frames")
        extract(data)
        return
    if not args.train:
        ap.error("pass --extract or --train")

    import torch
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"{'group':10s} {'members':34s} {'n_train':>8s} "
          f"{'val_bacc':>9s} {'TEST_bacc':>10s}")
    for gname, members in GROUPS.items():
        p = data / f"{gname}.npz"
        if not p.exists():
            continue
        r = train_group(p, args.epochs, args.lr, args.batch, args.seed)
        if r is None:
            print(f"{gname:10s} {' / '.join(members):34s} "
                  f"{'--':>8s} {'too few val/test':>20s}")
            continue
        torch.save({"state_dict": r.pop("state"), "size": SIZE,
                    "classes": members, "group": gname},
                   out / f"{gname}.pt")
        results[gname] = {**r, "members": members}
        print(f"{gname:10s} {' / '.join(members):34s} {r['n_train']:8d} "
              f"{r['val_balanced_acc']:9.4f} {r['test_balanced_acc']:10.4f}")

    print(f"\nconfusion matrices on TEST (rows = true, cols = predicted):")
    for g, r in results.items():
        print(f"  {g:8s} {r['members']}  {r['confusion']}")
    print(f"\nThe bar to beat is the DETECTOR on the same components. It")
    print(f"relabels 67 of 2815 detections wrongly (2.38%), concentrated in")
    print(f"these groups, and that costs 20% of images their shot at strict")
    print(f"success. A group model only helps if it is right where the")
    print(f"detector is wrong, so integration must be measured end-to-end,")
    print(f"not judged on these numbers alone.")
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
