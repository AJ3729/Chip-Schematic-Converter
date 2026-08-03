#!/usr/bin/env python3
"""Report derived data artifacts that are older than what they derive from.

Regenerating ``data/cleaned`` invalidates everything projected through
``data/transforms.json`` — detection labels, detection caches, projected
GT — and none of those consumers check. That is not hypothetical: the
YOLO labels were left four days behind a preprocessing change, and the
committed detection mAP of 0.9725 silently became 0.051 on re-run, with
no error at any layer because the labels still parsed and the instance
counts still matched.

This is a timestamp check, so it is a smoke alarm, not a proof: a
regenerated-but-identical artifact reports stale, and a hand-edited one
can report fresh. `tests/test_yolo_labels_fresh.py` does the real
content check for the labels specifically. Use this to find which
artifacts to look at after any preprocessing change.

Exit status is 1 if anything is stale, so it can gate a run.

Usage:
    python scripts/audit_data_freshness.py
    python scripts/audit_data_freshness.py --quiet   # only problems
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# (artifact glob, [source globs it must not predate], why it matters)
DEPS: list[tuple[str, list[str], str]] = [
    ("data/cleaned/*.jpg", ["data/raw/*.jpg"],
     "preprocessed frames"),
    ("data/transforms.json", ["data/cleaned/*.jpg"],
     "records how raw maps into the 512 frame"),
    ("data/detections/*.json", ["data/cleaned/*.jpg"],
     "detection cache is in frame coordinates"),
    ("data/yolo_cleaned/labels/test/*.txt", ["data/transforms.json"],
     "COCO boxes projected through the transforms — THIS IS THE ONE THAT BROKE"),
    # ^ knowingly stale, see KNOWN_STALE below
    ("data/yolo_cleaned_rebuilt/labels/test/*.txt", ["data/transforms.json"],
     "corrected 512 detection labels"),
    ("data/gt_netlists_verified_v3/*.json", ["data/transforms.json"],
     "GT boxes rebuilt from published COCO"),

    ("data/cleaned_1024/*.jpg", ["data/raw/*.jpg"],
     "preprocessed frames (1024, the default)"),
    ("data/transforms_1024.json", ["data/cleaned_1024/*.jpg"],
     "records how raw maps into the 1024 frame"),
    ("data/detections_1024/*.json", ["data/cleaned_1024/*.jpg"],
     "detection cache for the default config"),
    ("data/yolo_1024/labels/test/*.txt", ["data/transforms_1024.json"],
     "detection labels for the default config"),
    ("data/gt_val_1024/*.json", ["data/gt_netlists_verified_v3/*.json"],
     "validation-split GT expressed in 1024 coordinates"),
    # The test GT is annotated natively in the 1024 frame, so it has no
    # upstream to go stale against.
    ("data/gt_test_1024/*.json", [], "test-split GT (benchmark.gt_dir)"),
]

# tolerance: artifacts written in the same batch race by seconds
SLACK_S = 300

# Retained deliberately and known to be stale. Reported, but does not fail
# the audit — a check that always exits nonzero gets ignored, which would
# cost exactly the signal this script exists to give.
KNOWN_STALE = {"data/yolo_cleaned/labels/test/*.txt"}


def newest(pattern: str) -> tuple[float, str] | None:
    files = glob.glob(pattern)
    if not files:
        return None
    f = max(files, key=os.path.getmtime)
    return os.path.getmtime(f), f


def stamp(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="only report problems")
    args = ap.parse_args()

    stale, known_stale, missing = [], [], []
    for pattern, sources, why in DEPS:
        got = newest(pattern)
        if got is None:
            missing.append((pattern, why))
            if not args.quiet:
                print(f"  --  {pattern:44s} (absent) — {why}")
            continue
        t_art, f_art = got
        worst = None
        for s in sources:
            src = newest(s)
            if src and (worst is None or src[0] > worst[0]):
                worst = src
        if worst and t_art + SLACK_S < worst[0]:
            known = pattern in KNOWN_STALE
            (known_stale if known else stale).append(
                (pattern, f_art, t_art, worst[1], worst[0], why))
            print(f"  {'known' if known else 'STALE'} {pattern:44s} "
                  f"{stamp(t_art)} < {stamp(worst[0])} "
                  f"({os.path.basename(worst[1])})")
        elif not args.quiet:
            print(f"  ok    {pattern:44s} {stamp(t_art)}")

    if stale:
        print(f"\n{len(stale)} STALE artifact(s). Regenerate before trusting any "
              f"number that depends on them:")
        for pattern, _, _, src, _, why in stale:
            print(f"  - {pattern}\n      why it matters: {why}")
        print("\n  frames        scripts/preprocess.py + scripts/record_transforms.py"
              "\n  detections    scripts/detect_batch.py"
              "\n  YOLO labels   scripts/make_yolo_dataset.py  (then eval_detector.py)"
              "\n  GT boxes      scripts/fix_gt_boxes.py --apply")
        return 1

    print(f"\nNo unexpected staleness"
          f"{f'; {len(known_stale)} known-stale artifact(s) retained for provenance' if known_stale else ''}"
          f"{f'; {len(missing)} absent' if missing else ''}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
