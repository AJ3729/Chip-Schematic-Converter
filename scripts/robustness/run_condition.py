#!/usr/bin/env python3
"""Run one corruption condition end to end, into fresh directories.

Nothing existing is written. Every path this touches lives under
data/robustness/ or results/robustness/, and the per-condition config is
generated rather than edited, so configs/default.yaml is left alone. In
particular --out is ALWAYS passed to record_transforms: its default is
data/transforms.json, which is the file the whole corpus depends on.

Stages, each the same script the published results were produced by:
    corrupt -> record_transforms (preprocess + exact transform)
            -> detect_batch      -> benchmark

Usage:
    python scripts/robustness/run_condition.py --condition clean --split test
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import corruptions as C  # noqa: E402

DATA = ROOT / "data/robustness"
RES = ROOT / "results/robustness"
CFGD = ROOT / "configs/robustness"


def gen_config(cond: str, clean_dir: Path, det_dir: Path, gt_dir: str,
               impulse_k: int = 0) -> Path:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    if impulse_k:
        cfg["preprocess"]["impulse_median_ksize"] = impulse_k
    cfg["detect"]["cache_dir"] = str(det_dir.relative_to(ROOT))
    cfg["preprocess"]["images_dir"] = str(clean_dir.relative_to(ROOT))
    cfg["benchmark"]["gt_dir"] = gt_dir
    CFGD.mkdir(parents=True, exist_ok=True)
    p = CFGD / f"{cond}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def sh(cmd: list[str], label: str) -> float:
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t
    if r.returncode != 0:
        print(f"    !! {label} FAILED ({dt:.0f}s)")
        print("    " + "\n    ".join((r.stderr or r.stdout).strip().splitlines()[-12:]))
        raise SystemExit(2)
    print(f"    {label}: {dt:.0f}s")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--gt-dir", default="data/gt_test_1024")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--preprocess-v2", action="store_true",
                    help="drive preprocessing through the impulse-prefilter copy")
    ap.add_argument("--impulse-k", type=int, default=5,
                    help="median kernel for --preprocess-v2 (0 disables)")
    ap.add_argument("--source-cond", default=None,
                    help="reuse another condition's corrupted raw scans")
    ap.add_argument("--lossless", action="store_true",
                    help="write corrupted scans as PNG, so the delivered "
                         "perturbation is not attenuated by JPEG re-encoding")
    ap.add_argument("--prune-raw", type=int, default=0,
                    help="after preprocessing, keep only N corrupted scans "
                         "(they are deterministic, so this is free to redo)")
    a = ap.parse_args()

    cond = a.condition
    stems = [l.strip() for l in
             (ROOT / a.splits_dir / f"{a.split}.txt").read_text().split() if l.strip()]
    if a.limit:
        stems = stems[:a.limit]

    raw_dir = DATA / "raw" / (a.source_cond or cond)
    clean_dir = DATA / "cleaned" / cond
    det_dir = DATA / "detections" / cond
    tf_path = DATA / "transforms" / f"{cond}.json"
    out_dir = RES / cond
    for d in (raw_dir, clean_dir, det_dir, tf_path.parent, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[{cond}] {len(stems)} images")
    t0 = time.time()

    # 1. corrupt a COPY of the raw scans
    n = 0
    for name in ([] if a.source_cond else stems):
        stem = Path(name).stem
        ext = "png" if a.lossless else "jpg"
        dst = raw_dir / f"{stem}.{ext}"
        if dst.exists():
            continue
        img = cv2.imread(str(ROOT / "data/raw" / f"{stem}.jpg"))
        if img is None:
            print(f"    !! missing raw {stem}")
            continue
        # PNG takes no quality argument; JPEG q95 still loses ~29% of an
        # additive perturbation, which is the whole reason --lossless exists.
        cv2.imwrite(str(dst), C.apply(cond, img, stem),
                    [] if a.lossless else [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        n += 1
    print(f"    corrupt: {n} written ({len(stems) - n} already present)")

    cfg = gen_config(cond, clean_dir, det_dir, a.gt_dir,
                     a.impulse_k if a.preprocess_v2 else 0)

    # 2. preprocess + record the exact transform (never the shared file)
    #
    # record_transforms hard-fails when any published annotation box lands
    # outside the canvas, because in normal use that means preprocessing has a
    # bug. Under a severe geometric corruption it means the corruption worked:
    # rotating a page by 10 degrees genuinely pushes content near the edge out
    # of frame. That is a result, not a crash, so it is RECORDED and the run
    # continues -- but only for that specific failure, and only after the
    # transforms file was actually written. Any other non-zero exit still stops
    # the condition.
    pre = subprocess.run(
        [sys.executable,
         "scripts/robustness/record_transforms_v2.py" if a.preprocess_v2
         else "scripts/record_transforms.py",
         "--raw-dir", str(raw_dir.relative_to(ROOT)),
         "--clean-dir", str(clean_dir.relative_to(ROOT)),
         "--out", str(tf_path.relative_to(ROOT)),
         "--config", str(cfg.relative_to(ROOT)),
         "--write-images", "--no-annotation-aware"],
        cwd=ROOT, capture_output=True, text=True)
    blob = (pre.stdout or "") + (pre.stderr or "")
    guard = re.search(r"annotation containment: (\d+) of (\d+) boxes outside "
                      r"canvas, across (\d+) image", blob)
    offcanvas = None
    if guard:
        offcanvas = {"boxes_outside": int(guard.group(1)),
                     "boxes_total": int(guard.group(2)),
                     "images_affected": int(guard.group(3))}
    if pre.returncode != 0:
        containment_only = ("[FAIL] annotations are being cropped out of frame" in blob
                            and tf_path.exists())
        if not containment_only:
            print("    !! preprocess FAILED")
            print("    " + "\n    ".join(blob.strip().splitlines()[-12:]))
            raise SystemExit(2)
        print(f"    preprocess: OK, containment guard tripped "
              f"({offcanvas['boxes_outside']}/{offcanvas['boxes_total']} boxes "
              f"off-canvas across {offcanvas['images_affected']} images) -- recorded")
    else:
        print("    preprocess: ok")

    # Corrupted scans are ~100 MB per condition at full resolution and are
    # only needed again for the heatmap overlays. They regenerate exactly from
    # the seed, so keeping 192 of them per condition buys nothing but disk.
    if a.prune_raw and not a.source_cond:
        keep = {Path(x).stem for x in stems[:a.prune_raw]}
        freed = 0
        for f in list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")):
            if f.stem not in keep:
                freed += f.stat().st_size
                f.unlink()
        print(f"    pruned raw: kept {len(keep)}, freed {freed / 1e6:.0f} MB")

    # 3. detect on the corrupted frames
    sh([sys.executable, "scripts/detect_batch.py",
        "--images-dir", str(clean_dir.relative_to(ROOT)),
        "--config", str(cfg.relative_to(ROOT))], "detect")

    # 4. score against the UNCHANGED ground truth
    sh([sys.executable, "scripts/benchmark.py",
        "--split", a.split, "--splits-dir", a.splits_dir,
        "--images-dir", str(clean_dir.relative_to(ROOT)),
        "--gt-dir", a.gt_dir,
        "--out-dir", str(out_dir.relative_to(ROOT)),
        "--config", str(cfg.relative_to(ROOT))], "benchmark")

    if offcanvas is not None:
        (out_dir / "offcanvas.json").write_text(json.dumps(offcanvas, indent=1) + "\n")
    s = json.loads((out_dir / "summary.json").read_text())
    strict = s.get("topology", {}).get("strict_success", {}).get("mean")
    print(f"[{cond}] strict_success = {strict}   total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
