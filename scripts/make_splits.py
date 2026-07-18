#!/usr/bin/env python3
"""Freeze stratified train/val/test splits (Phase B2).

Stratification: component-count tertile × rarest-class-present, computed
from the published COCO annotations. If per-image drafter/writer
metadata exists in the COCO file, splits are made drafter-disjoint
(whole drafters assigned to one split, greedy by size) and the achieved
ratios are reported; otherwise that limitation is recorded in the
metadata file.

Outputs (committed to git — they are part of benchmark contribution C1):
    data/splits/train.txt, val.txt, test.txt   one image filename per line
    data/splits/splits_meta.json               seed, strategy, counts, distributions

Usage:
    python scripts/make_splits.py --coco data/digitize_hcd/annotations.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DRAFTER_KEYS = ("drafter", "writer", "author", "annotator", "artist")


def load_coco(path: str | Path) -> tuple[list[dict], dict[int, str], dict[int, list[int]]]:
    """Returns (images, category_id->name, image_id->[category ids])."""
    with open(path) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    per_image: dict[int, list[int]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        per_image[ann["image_id"]].append(ann["category_id"])
    return coco["images"], cats, per_image


def find_drafter_key(images: list[dict]) -> str | None:
    for key in DRAFTER_KEYS:
        if any(key in img for img in images):
            return key
    return None


def tertile_thresholds(counts: list[int]) -> tuple[int, int]:
    s = sorted(counts)
    return s[len(s) // 3], s[2 * len(s) // 3]


def stratum_of(img_id, per_image, cats, class_freq, t1, t2) -> str:
    cat_ids = per_image.get(img_id, [])
    n = len(cat_ids)
    tert = "T1" if n <= t1 else ("T2" if n <= t2 else "T3")
    if not cat_ids:
        return f"{tert}|none"
    rarest = min(set(cat_ids), key=lambda c: class_freq[c])
    return f"{tert}|{cats[rarest]}"


def allocate(bucket: list, rng: random.Random) -> dict[str, list]:
    """Split one shuffled bucket by RATIOS using largest-remainder."""
    rng.shuffle(bucket)
    n = len(bucket)
    n_train = round(n * RATIOS["train"])
    n_val = round(n * RATIOS["val"])
    return {
        "train": bucket[:n_train],
        "val": bucket[n_train : n_train + n_val],
        "test": bucket[n_train + n_val :],
    }


def split_stratified(images, cats, per_image, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    class_freq = Counter(c for ids in per_image.values() for c in ids)
    t1, t2 = tertile_thresholds([len(per_image.get(i["id"], [])) for i in images])

    buckets: dict[str, list] = defaultdict(list)
    for img in images:
        buckets[stratum_of(img["id"], per_image, cats, class_freq, t1, t2)].append(img)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in sorted(buckets):
        alloc = allocate(buckets[key], rng)
        for name in splits:
            splits[name].extend(i["file_name"] for i in alloc[name])
    return splits


def split_drafter_disjoint(images, drafter_key: str, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups: dict[str, list] = defaultdict(list)
    for img in images:
        groups[str(img.get(drafter_key, "unknown"))].append(img)
    drafters = sorted(groups, key=lambda d: len(groups[d]), reverse=True)
    rng.shuffle(drafters)  # tie-break order deterministically

    total = len(images)
    targets = {k: v * total for k, v in RATIOS.items()}
    sizes = {k: 0 for k in RATIOS}
    splits: dict[str, list[str]] = {k: [] for k in RATIOS}
    for d in sorted(drafters, key=lambda d: len(groups[d]), reverse=True):
        # assign the drafter to the split furthest below its target
        deficit = {k: targets[k] - sizes[k] for k in RATIOS}
        dest = max(deficit, key=deficit.get)
        splits[dest].extend(i["file_name"] for i in groups[d])
        sizes[dest] += len(groups[d])
    return splits


def class_distribution(split_files, images, per_image, cats) -> dict[str, int]:
    by_name = {i["file_name"]: i["id"] for i in images}
    dist: Counter = Counter()
    for fn in split_files:
        for c in per_image.get(by_name[fn], []):
            dist[cats[c]] += 1
    return dict(sorted(dist.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", required=True, help="published COCO annotations JSON")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])

    images, cats, per_image = load_coco(args.coco)
    drafter_key = find_drafter_key(images)

    if drafter_key:
        splits = split_drafter_disjoint(images, drafter_key, seed)
        strategy = f"drafter_disjoint (key: {drafter_key})"
    else:
        splits = split_stratified(images, cats, per_image, seed)
        strategy = (
            "stratified by component-count tertile x rarest-class "
            "(no drafter metadata in the published annotations — "
            "drafter-disjoint splitting is not possible; stated as a "
            "limitation per plan B2)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, files in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(sorted(files)) + "\n")

    meta = {
        "seed": seed,
        "ratios": RATIOS,
        "strategy": strategy,
        "coco_source": str(args.coco),
        "counts": {k: len(v) for k, v in splits.items()},
        "achieved_ratios": {
            k: round(len(v) / len(images), 4) for k, v in splits.items()
        },
        "class_distribution": {
            k: class_distribution(v, images, per_image, cats)
            for k, v in splits.items()
        },
    }
    with open(out_dir / "splits_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] splits written to {out_dir}: "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"[INFO] strategy: {strategy}")


if __name__ == "__main__":
    main()
