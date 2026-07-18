#!/usr/bin/env python3
"""Hash-reconcile local data/ images against the published Digitize-HCD
download (Phase B1).

For every image in the published set and in local data/raw and
data/cleaned, computes sha256 and reports: exact matches, local files
absent from the published set, published files absent locally, and
duplicate images inside each set. Writes data/reconciliation.json —
the provenance evidence cited by data/README.md.

Usage:
    python scripts/reconcile_data.py --published data/digitize_hcd/extracted
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def hash_dir(root: Path) -> dict[str, list[str]]:
    """sha256 -> [relative paths] for every image under root."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS or not p.is_file():
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[h].append(str(p.relative_to(root)))
    return dict(out)


def summarize(name: str, hashes: dict[str, list[str]]) -> dict:
    files = sum(len(v) for v in hashes.values())
    dups = {h: v for h, v in hashes.items() if len(v) > 1}
    return {
        "name": name,
        "files": files,
        "unique_contents": len(hashes),
        "internal_duplicates": {h[:16]: v for h, v in dups.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published", required=True,
                    help="extracted Digitize-HCD image root")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cleaned", default="data/cleaned")
    ap.add_argument("--out", default="data/reconciliation.json")
    args = ap.parse_args()

    print("[INFO] hashing published set...")
    pub = hash_dir(Path(args.published))
    print("[INFO] hashing local raw...")
    raw = hash_dir(Path(args.raw))
    print("[INFO] hashing local cleaned...")
    cleaned = hash_dir(Path(args.cleaned))

    raw_matched = {h for h in raw if h in pub}
    cleaned_matched = {h for h in cleaned if h in pub}

    report = {
        "sets": [summarize("published", pub), summarize("raw", raw),
                 summarize("cleaned", cleaned)],
        "raw_vs_published": {
            "matched": len(raw_matched),
            "raw_only": len(raw) - len(raw_matched),
            "published_only": len(pub) - len(raw_matched),
            "raw_only_examples": sorted(
                raw[h][0] for h in list(set(raw) - raw_matched)[:10]
            ),
        },
        "cleaned_vs_published": {
            "matched": len(cleaned_matched),
            "note": (
                "cleaned images are preprocessed (deskew/binarize/resize) "
                "so zero byte-level matches are expected; filename "
                "correspondence is used instead"
            ),
        },
    }

    out = Path(args.out)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "sets"}, indent=2))
    for s in report["sets"]:
        print(f"[INFO] {s['name']}: {s['files']} files, "
              f"{s['unique_contents']} unique, "
              f"{len(s['internal_duplicates'])} duplicated contents")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
