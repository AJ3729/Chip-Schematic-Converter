#!/usr/bin/env python3
"""Build the CGHD zero-shot evaluation subset (Phase B4).

Deterministically samples N images per drafter from the CGHD zip
(drafter-stratified — CGHD ships one folder per drafter), extracts only
those images plus their Pascal-VOC annotation XMLs and the root class
metadata, and writes the frozen manifest to data/splits/cghd_zero_shot.txt
(versioned, like the Digitize-HCD split manifests).

Usage:
    python scripts/make_cghd_subset.py --zip data/cghd/cghd-zenodo-12.zip --per-drafter 4
"""

from __future__ import annotations

import argparse
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed

IMG_RE = re.compile(r"^(drafter_\d+)/images/([^/]+\.(?:jpe?g|png))$", re.I)
ROOT_META = ("classes.json", "classes_color.json", "classes_ports.json",
             "classes_discontinuous.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", dest="zip_path", default="data/cghd/cghd-zenodo-12.zip")
    ap.add_argument("--out-dir", default="data/cghd/subset")
    ap.add_argument("--manifest", default="data/splits/cghd_zero_shot.txt")
    ap.add_argument("--per-drafter", type=int, default=4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    rng = random.Random(seed)

    zf = zipfile.ZipFile(args.zip_path)
    by_drafter: dict[str, list[tuple[str, str]]] = defaultdict(list)
    names = set(zf.namelist())
    for name in sorted(names):
        m = IMG_RE.match(name)
        if not m:
            continue
        drafter, img = m.groups()
        xml = f"{drafter}/annotations/{Path(img).stem}.xml"
        if xml in names:  # only images that actually have annotations
            by_drafter[drafter].append((name, xml))

    selected: list[tuple[str, str]] = []
    for drafter in sorted(by_drafter, key=lambda d: int(d.split("_")[1])):
        pool = by_drafter[drafter]
        rng.shuffle(pool)
        selected.extend(pool[: args.per_drafter])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for meta in ROOT_META:
        if meta in names:
            zf.extract(meta, out_dir)
    for img_name, xml_name in selected:
        zf.extract(img_name, out_dir)
        zf.extract(xml_name, out_dir)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(img for img, _ in selected) + "\n"
    )

    print(f"[OK] {len(selected)} images from {len(by_drafter)} drafters "
          f"extracted to {out_dir}")
    print(f"[OK] frozen manifest written to {manifest}")


if __name__ == "__main__":
    main()
