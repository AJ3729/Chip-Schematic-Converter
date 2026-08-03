"""Which split a script reads, and why that is never a default worth guessing.

Until 2026-08-03 every tuned parameter in this project was selected by
sweeping the split it was then reported on. The two evaluation splits
exchanged names to fix that (data/README.md -> "the 2026-08-03 role swap"),
but the fix only holds if every script says out loud which split it wants.

THE CONVENTION

    val   190 images, data/gt_val_1024.  Anything that could influence a
          design decision reads this: sweeps, oracles, diagnostics, ablation
          exploration, failure analysis. Looking at these images is free.

    test  192 images, data/gt_test_1024.  Only final measurement reads this:
          the benchmark, the detector evaluation, the reported oracle
          attribution. Every look costs a little of the split's value.

A script that hardcoded a path could not express that distinction, and two
dozen of them hardcoded data/splits/test.txt — which silently became the
reporting split the moment the names moved. Route through here instead, and
declare the default with add_split_arg() so `--help` states the choice.

CAVEAT ON THE DETECTOR. The YOLO weights in experiments/ were early-stopped
on the 192 images that are now `test` (patience 50, ultralytics `split: val`),
so detection metrics measured on test are optimistic by a measured
+0.017 mAP@0.5 / +0.023 mAP@0.5:0.95 against the 190 the detector never saw.
The topology pipeline consumes those detections and inherits the bias.
Removing it needs a retrain that early-stops on val.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SPLITS_DIR = Path("data/splits")

#: GT directory that goes with each split, for --gt-dir defaults.
GT_DIRS = {
    "test": "data/gt_test_1024",
    "val": "data/gt_val_1024",
}


def split_path(split: str, splits_dir: str | Path | None = None) -> Path:
    """Resolve a split name (or an explicit .txt path) to a manifest file."""
    p = Path(split)
    if p.suffix == ".txt":
        return p
    return Path(splits_dir or SPLITS_DIR) / f"{split}.txt"


def load_split(split: str, splits_dir: str | Path | None = None) -> list[str]:
    """Image filenames in a split, in manifest order."""
    path = split_path(split, splits_dir)
    if not path.exists():
        raise SystemExit(
            f"no such split: {path}. Known splits: "
            f"{', '.join(sorted(p.stem for p in Path(splits_dir or SPLITS_DIR).glob('*.txt')))}"
        )
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_stems(split: str, splits_dir: str | Path | None = None) -> list[str]:
    return [Path(n).stem for n in load_split(split, splits_dir)]


def add_split_arg(ap: argparse.ArgumentParser, default: str) -> None:
    """Add --split/--splits-dir, stating in --help why this default.

    `default` must be "val" for anything exploratory and "test" only for a
    number that is actually reported. Passing anything else is a mistake
    worth catching at import time rather than in a results table.
    """
    if default not in ("val", "test"):
        raise ValueError(f"split default must be 'val' or 'test', got {default!r}")
    why = ("selection/diagnosis — reads val so it cannot influence a reported "
           "number" if default == "val" else
           "reported measurement — reads the held-out test split")
    ap.add_argument("--split", default=default, help=f"{why} (default: {default})")
    ap.add_argument("--splits-dir", default=None,
                    help=f"directory of split manifests (default: {SPLITS_DIR})")
