#!/usr/bin/env python3
"""Qualitative figures: what the failures actually look like.

Every other figure in the paper is an aggregate, and aggregates cannot show
that a 0.55 net-$F_1$ circuit is one fused rail rather than fifty small
errors. This renders the failure modes side by side with ground truth, and
picks WHICH circuits to show from the per-image metrics rather than by eye,
so the gallery is not a curated best case.

Selection rule, one circuit per mode, from the reported run's per_image.csv:

    weld     over-merge — terminal-pair recall high, precision low. The
             pipeline found the conductors and then fused them.
    split    under-merge — precision high, recall low. Conductors survive
             but a net is broken into fragments.
    missing  a GT component with no matching detection at all.
    solved   strict success, for contrast; without it a reader cannot tell
             a hard failure from a hard dataset.

Each panel shows the cleaned frame with every component box coloured by the
net its first terminal sits on: ground truth on the left, prediction on the
right. Two boxes sharing a colour share a net, so an over-merge reads as
colours collapsing and a split reads as one colour splintering.

Usage:
    python scripts/make_paper_qualitative.py
    python scripts/make_paper_qualitative.py --run results/paper_test/seeds/seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.config import load_config          # noqa: E402
from schematic2netlist.detect import load_cached_detections  # noqa: E402
from schematic2netlist.gt import load_gt                  # noqa: E402
from schematic2netlist.pipeline import run_pipeline       # noqa: E402

FIG = ROOT / "paper" / "figures"

# BGR, distinguishable at print size and stable per net ORDER (not per net
# name — predicted and GT net names are independent labellings, so matching
# them by name would colour the same conductor differently on the two sides
# for no reason)
PALETTE = [(60, 60, 220), (30, 160, 30), (220, 130, 20), (200, 30, 200),
           (20, 170, 200), (150, 90, 40), (200, 60, 120), (80, 140, 60),
           (40, 90, 220), (120, 80, 200), (0, 130, 130), (90, 90, 90)]
GROUND = (25, 25, 25)


def num(r: dict, k: str) -> float:
    v = r[k]
    return 1.0 if v == "True" else 0.0 if v == "False" else float(v)


def pick(rows: list[dict]) -> dict[str, dict]:
    """Choose one circuit per failure mode, by metric rather than by eye."""
    def ok(r):          # ignore circuits too small to show anything
        return int(r["n_gt"]) >= 8

    cand = [r for r in rows if ok(r)]
    out: dict[str, dict] = {}

    weld = [r for r in cand
            if num(r, "terminal_pair_recall") - num(r, "terminal_pair_precision") > 0.15]
    if weld:
        out["over-merge (weld)"] = max(
            weld, key=lambda r: num(r, "terminal_pair_recall")
            - num(r, "terminal_pair_precision"))

    split = [r for r in cand
             if num(r, "terminal_pair_precision") - num(r, "terminal_pair_recall") > 0.15]
    if split:
        out["under-merge (split)"] = max(
            split, key=lambda r: num(r, "terminal_pair_precision")
            - num(r, "terminal_pair_recall"))

    miss = [r for r in cand if int(r["unmatched_gt"]) > 0]
    if miss:
        out["missing detection"] = max(miss, key=lambda r: int(r["unmatched_gt"]))

    solved = [r for r in cand if num(r, "strict_success") == 1.0]
    if solved:
        out["solved (strict success)"] = max(solved, key=lambda r: int(r["n_gt"]))
    return out


def colour_for(net, order):
    if net in ("0", 0):
        return GROUND
    return PALETTE[order.index(net) % len(PALETTE)] if net in order else (150,) * 3


def net_order(entries):
    order, seen = [], set()
    for nets in entries:
        for n in nets:
            if n and n not in seen and n not in ("0", 0):
                seen.add(n)
                order.append(n)
    return order


def draw(frame, boxes, title):
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    vis = (vis * 0.35 + 255 * 0.65).astype(np.uint8)   # fade so colour reads
    order = net_order([nets for _, nets in boxes])
    for (x1, y1, x2, y2), nets in boxes:
        col = colour_for(nets[0] if nets else None, order)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        # a component whose terminals sit on DIFFERENT nets gets a second
        # inner box in the second net's colour; a self-shorted component
        # therefore shows as a single flat colour, which is the tell
        if len(set(n for n in nets if n)) > 1:
            c2 = colour_for(sorted(set(nets), key=lambda n: nets.index(n))[1], order)
            cv2.rectangle(vis, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), c2, 2)
    bar = np.full((26, vis.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def panel(stem: str, mode: str, cfg: dict, gt_dir: Path, images_dir: Path,
          det_dir: Path, metrics: dict) -> np.ndarray | None:
    frame_p = images_dir / f"{stem}.jpg"
    gt_p = gt_dir / f"{stem}.json"
    det_p = det_dir / f"{stem}.json"
    if not (frame_p.exists() and gt_p.exists() and det_p.exists()):
        return None
    gray = cv2.imread(str(frame_p), cv2.IMREAD_GRAYSCALE)
    gt = load_gt(str(gt_p))

    gt_boxes = []
    for c in gt["components"]:
        bx, by, bw, bh = c["bbox"]
        gt_boxes.append((
            (int(bx - bw / 2), int(by - bh / 2), int(bx + bw / 2), int(by + bh / 2)),
            [t.get("net") for t in c["terminals"]]))

    res = run_pipeline(frame_p, cfg, detections=load_cached_detections(str(det_p)))
    # Pipeline components carry connectivity, not geometry: `id` indexes into
    # res["detections"], and non-electrical detections (crossovers) have no
    # component at all, so the two lists are different lengths. Reading a
    # "bbox" key off a component silently yields nothing.
    dets = res["detections"]
    pr_boxes = []
    for c in res["components"]:
        d = dets[c["id"]] if 0 <= c["id"] < len(dets) else None
        if d is None:
            continue
        bx, by, bw, bh = d["x"], d["y"], d["width"], d["height"]
        pr_boxes.append((
            (int(bx - bw / 2), int(by - bh / 2), int(bx + bw / 2), int(by + bh / 2)),
            list(c.get("node_names") or [])))

    left = draw(gray, gt_boxes, f"ground truth  ({len(gt_boxes)} components)")
    right = draw(gray, pr_boxes, f"predicted  ({len(pr_boxes)} components)")
    sep = np.full((left.shape[0], 6, 3), 255, np.uint8)
    row = np.hstack([left, sep, right])

    cap = np.full((30, row.shape[1], 3), 255, np.uint8)
    txt = (f"{mode} | {stem}: net F1 {metrics['net_f1']:.2f}, "
           f"tp precision {metrics['p']:.2f}, recall {metrics['r']:.2f}")
    cv2.putText(cap, txt, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (150, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([cap, row])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/paper_test/seeds/seed0")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="fig_failure_gallery")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_dir = Path(args.gt_dir or cfg["benchmark"]["gt_dir"])
    images_dir = Path(cfg["preprocess"]["images_dir"])
    det_dir = Path(cfg["detect"]["cache_dir"])

    with (ROOT / args.run / "per_image.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    chosen = pick(rows)
    if not chosen:
        raise SystemExit("no circuits matched the selection rules")

    panels = []
    for mode, r in chosen.items():
        stem = Path(r["image"]).stem
        m = {"net_f1": num(r, "net_f1"),
             "p": num(r, "terminal_pair_precision"),
             "r": num(r, "terminal_pair_recall")}
        print(f"  {mode:26s} {stem}  netF1={m['net_f1']:.3f} "
              f"P={m['p']:.3f} R={m['r']:.3f}")
        p = panel(stem, mode, cfg, gt_dir, images_dir, det_dir, m)
        if p is not None:
            panels.append(p)

    w = max(p.shape[1] for p in panels)
    padded = [np.pad(p, ((0, 0), (0, w - p.shape[1]), (0, 0)),
                     constant_values=255) for p in panels]
    grid = np.vstack([np.pad(p, ((0, 14), (0, 0), (0, 0)), constant_values=255)
                      for p in padded])

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{args.out}.png"
    cv2.imwrite(str(out), grid)
    print(f"wrote {out.relative_to(ROOT)} ({grid.shape[1]}x{grid.shape[0]})")

    manifest = {"run": args.run, "gt_dir": str(gt_dir),
                "selected": {k: Path(v["image"]).stem for k, v in chosen.items()},
                "rule": "chosen by per-image metrics, not by eye; "
                        "see scripts/make_paper_qualitative.py"}
    (FIG / f"{args.out}_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
