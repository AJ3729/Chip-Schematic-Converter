#!/usr/bin/env python3
"""Fill the per-image detection cache for a set of images.

Runs the configured detection backend (hosted Roboflow until local
weights exist in Phase C) for every image that is not already cached in
detect.cache_dir. Never overwrites existing cache entries.

Usage:
    python scripts/detect_batch.py --images data/splits/test.txt
    python scripts/detect_batch.py --images-dir data/cleaned --limit 20 --backend roboflow
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from schematic2netlist.config import load_config, set_by_dotted_key
from schematic2netlist.detect import cache_path_for_image, detect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=None,
                    help="split .txt file listing image filenames")
    ap.add_argument("--images-dir", default=None,
                    help="defaults to the CONFIG's preprocess.images_dir. It used "
                         "to default to a hardcoded data/cleaned, which is the "
                         "512-era frame set: a cache generated without this flag "
                         "was silently computed on the wrong frame generation, in "
                         "a different coordinate frame, and every box in it was "
                         "misplaced. scripts/check_cache_alignment.py catches it, "
                         "but the default should not create the hazard.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default=None,
                    help="override detect.backend (roboflow | ultralytics)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="pause between API calls (rate-limit kindness)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg = set_by_dotted_key(cfg, "detect.backend", args.backend)

    images_dir = Path(args.images_dir
                      or cfg["preprocess"]["images_dir"])
    if args.images:
        names = Path(args.images).read_text().split()
        images = [images_dir / n for n in names]
    else:
        images = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
    if args.limit:
        images = images[: args.limit]

    cached, fetched, failed = 0, 0, []
    for i, img in enumerate(images):
        if cache_path_for_image(img, cfg).exists():
            cached += 1
            continue
        try:
            dets = detect(img, cfg)
            fetched += 1
            print(f"[{i + 1}/{len(images)}] {img.name}: {len(dets)} detections")
            time.sleep(args.sleep)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            failed.append((img.name, f"{type(e).__name__}: {e}"))
            print(f"[{i + 1}/{len(images)}] {img.name}: FAILED {type(e).__name__}")

    print(f"\n[OK] {fetched} fetched, {cached} already cached, {len(failed)} failed")
    for name, err in failed[:10]:
        print(f"     {name}: {err}")


if __name__ == "__main__":
    main()
