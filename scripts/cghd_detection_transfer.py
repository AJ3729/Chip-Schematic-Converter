#!/usr/bin/env python3
"""Zero-shot detection transfer to CGHD (task B5).

CGHD ships bounding boxes and classes, so detection is scoreable with zero
human annotation. The frozen detector runs over the evaluable pool and is
scored under the COCO protocol, broken down by drafter.

TWO CONSTRAINTS THAT SHAPE THE SCORING, both from the corpus rather than a
choice:

1. We evaluate only on classes the detector was TRAINED on. CGHD's 53 classes
   are mapped through spec/class_map_cghd.yaml; anything outside the 17-class
   vocabulary is excluded from both sides, and the exclusion is counted.

2. CGHD does not annotate transistor POLARITY -- its labels are
   `transistor.bjt` and `transistor.fet` with no NPN/PNP or n-/p-channel
   distinction. Both sides are therefore collapsed to `bjt` and `fet` for
   scoring. **Polarity cannot be evaluated on CGHD at all**; that result rests
   on Digitize-HCD alone, and every output of this script says so.

Usage:
    python scripts/cghd_detection_transfer.py
    python scripts/cghd_detection_transfer.py --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from stats.bootstrap import bootstrap_mean  # noqa: E402

IMG = ROOT / "data/cghd_1024/images"
ANN = ROOT / "data/cghd_1024/annotations"
MAP = ROOT / "spec/class_map_cghd.yaml"
OUT = ROOT / "results/cghd_detection_transfer.json"
WEIGHTS = "experiments/train_valstop/runs/yolov8s_640_seed{}/weights/best.pt"

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


def coarsen(cls: str, groups: dict[str, list[str]]) -> str:
    """Collapse pipeline classes into the granularity CGHD can express."""
    if cls.startswith("COARSE:"):
        return cls.split(":", 1)[1]
    for g, members in groups.items():
        if cls in members:
            return g
    return cls


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b are (n,4) and (m,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):          # monotone envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(grid, mrec, mpre, left=mpre[0], right=0.0)))


def evaluate(preds: dict, gts: dict, iou_thr: float) -> dict[str, float]:
    """Per-class AP at one IoU threshold. Greedy matching by descending score."""
    classes = sorted({c for g in gts.values() for c in g["cls"]})
    aps: dict[str, float] = {}
    for c in classes:
        rows = []
        n_gt = 0
        for stem, g in gts.items():
            gmask = [i for i, x in enumerate(g["cls"]) if x == c]
            n_gt += len(gmask)
            p = preds.get(stem, {"cls": [], "box": np.zeros((0, 4)), "score": []})
            pmask = [i for i, x in enumerate(p["cls"]) if x == c]
            if not pmask:
                continue
            gb = g["box"][gmask] if gmask else np.zeros((0, 4))
            pb = p["box"][pmask]
            ps = np.asarray(p["score"])[pmask]
            M = iou_matrix(pb, gb)
            taken = set()
            for j in np.argsort(-ps):
                best, bi = 0.0, -1
                for k in range(len(gb)):
                    if k in taken:
                        continue
                    if M[j, k] > best:
                        best, bi = M[j, k], k
                hit = best >= iou_thr and bi >= 0
                if hit:
                    taken.add(bi)
                rows.append((ps[j], hit))
        if n_gt == 0:
            continue
        if not rows:
            aps[c] = 0.0
            continue
        rows.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in rows])
        fp = np.cumsum([not r[1] for r in rows])
        rec = tp / n_gt
        prec = tp / np.maximum(tp + fp, 1e-9)
        aps[c] = average_precision(rec, prec)
    return aps


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    groups = yaml.safe_load(MAP.read_text())["coarse_groups"]
    files = sorted(ANN.glob("*.json"))[: a.limit] if a.limit else sorted(ANN.glob("*.json"))
    if not files:
        sys.exit(f"no adapted annotations under {ANN}; run adapters/cghd_to_pipeline.py")

    gts: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for f in files:
        d = json.loads(f.read_text())
        stem = f.stem
        cls = [coarsen(c["class"], groups) for c in d["components"]]
        box = np.array([c["bbox_xyxy"] for c in d["components"]], dtype=float)
        gts[stem] = {"cls": cls, "box": box}
        meta[stem] = {"drafter": d["drafter"], "group": d["drawing_group"]}
    print(f"CGHD evaluable pool: {len(gts)} images, "
          f"{sum(len(g['cls']) for g in gts.values())} boxes, "
          f"{len({m['drafter'] for m in meta.values()})} drafters")

    from ultralytics import YOLO
    per_seed: dict[str, dict] = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        model = YOLO(str(ROOT / WEIGHTS.format(s)))
        names = model.names
        preds: dict[str, dict] = {}
        paths = [str(IMG / f"{k}.jpg") for k in gts]
        for i in range(0, len(paths), 32):
            chunk = paths[i:i + 32]
            for path, r in zip(chunk, model.predict(chunk, verbose=False,
                                                    imgsz=640, conf=0.001)):
                stem = Path(path).stem
                b = r.boxes
                if b is None or len(b) == 0:
                    preds[stem] = {"cls": [], "box": np.zeros((0, 4)), "score": []}
                    continue
                preds[stem] = {
                    "cls": [coarsen(names[int(c)], groups) for c in b.cls.tolist()],
                    "box": b.xyxy.cpu().numpy(),
                    "score": b.conf.cpu().numpy().tolist(),
                }
            print(f"  seed{s}: {min(i+32, len(paths))}/{len(paths)}", flush=True)

        aps = {t: evaluate(preds, gts, float(t)) for t in IOU_THRESHOLDS}
        map50 = float(np.mean(list(aps[0.5].values()))) if aps[0.5] else 0.0
        map5095 = float(np.mean([np.mean(list(v.values()))
                                 for v in aps.values() if v]))
        # per-drafter mAP@0.5
        by_drafter = {}
        for d in sorted({m["drafter"] for m in meta.values()}):
            sub = {k: v for k, v in gts.items() if meta[k]["drafter"] == d}
            subp = {k: preds[k] for k in sub}
            v = evaluate(subp, sub, 0.5)
            by_drafter[str(d)] = {"n_images": len(sub),
                                  "map50": float(np.mean(list(v.values()))) if v else 0.0}
        per_seed[str(s)] = {
            "map50": map50, "map50_95": map5095,
            "per_class_ap50": aps[0.5],
            "per_drafter_map50": by_drafter,
        }
        print(f"  seed{s}: mAP@0.5={map50:.4f}  mAP@0.5:0.95={map5095:.4f}")

    m50 = [v["map50"] for v in per_seed.values()]
    m5095 = [v["map50_95"] for v in per_seed.values()]
    dcis = bootstrap_mean([v["map50"] for v in
                           next(iter(per_seed.values()))["per_drafter_map50"].values()
                           ] if False else m50, seed=0)

    # Digitize-HCD reference, from the committed artifacts
    hcd = [json.loads((ROOT / f"results/final/detection/seed{s}/test/summary.json")
                      .read_text()) for s in (0, 1, 2)]
    out = {
        "_what": "Zero-shot detection transfer to CGHD. The pipeline was frozen "
                 "before CGHD was touched; nothing here is tuned on it.",
        "_polarity_caveat": (
            "CGHD does not annotate transistor polarity: its labels are "
            "transistor.bjt and transistor.fet with no NPN/PNP or n-/p-channel "
            "distinction, and the Pascal VOC XML carries no sub-attribute. "
            "Both sides are collapsed to bjt/fet for scoring. POLARITY CANNOT "
            "BE EVALUATED ON CGHD. Every polarity and pin-order result in this "
            "work rests on Digitize-HCD alone."),
        "_vocabulary_caveat": (
            "Scored only on classes the detector was trained on. CGHD classes "
            "outside the 17-class vocabulary are excluded from both sides via "
            "spec/class_map_cghd.yaml; see results/cghd_coverage.json for the "
            "exclusion counts."),
        "cghd_version": 12,
        "protocol": "COCO 101-point interpolated AP, greedy matching by "
                    "descending confidence, conf floor 0.001, imgsz 640",
        "n_images": len(gts),
        "n_boxes": int(sum(len(g["cls"]) for g in gts.values())),
        "n_drafters": len({m["drafter"] for m in meta.values()}),
        "per_seed": per_seed,
        "cghd_map50_mean": float(np.mean(m50)),
        "cghd_map50_std": float(np.std(m50, ddof=1)) if len(m50) > 1 else 0.0,
        "cghd_map50_95_mean": float(np.mean(m5095)),
        "cghd_map50_95_std": float(np.std(m5095, ddof=1)) if len(m5095) > 1 else 0.0,
        "digitize_hcd_test_map50_mean": float(np.mean([h["map50"] for h in hcd])),
        "digitize_hcd_test_map50_95_mean": float(np.mean([h["map50_95"] for h in hcd])),
    }
    out["transfer_delta_map50"] = out["cghd_map50_mean"] - out["digitize_hcd_test_map50_mean"]
    out["transfer_delta_map50_95"] = out["cghd_map50_95_mean"] - out["digitize_hcd_test_map50_95_mean"]
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"\nCGHD          mAP@0.5 {out['cghd_map50_mean']:.4f} "
          f"+/- {out['cghd_map50_std']:.4f}   "
          f"mAP@0.5:0.95 {out['cghd_map50_95_mean']:.4f}")
    print(f"Digitize-HCD  mAP@0.5 {out['digitize_hcd_test_map50_mean']:.4f}"
          f"                mAP@0.5:0.95 {out['digitize_hcd_test_map50_95_mean']:.4f}")
    print(f"transfer delta mAP@0.5 {out['transfer_delta_map50']:+.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
