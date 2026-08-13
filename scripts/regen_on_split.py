#!/usr/bin/env python3
"""Re-run the paper's result set against a different split.

The 2026-08-03 role swap made every committed result a validation number
(data/README.md). Re-deriving them on the test split is not a matter of
re-running one command, because each arm of the ablation and each detector
seed is a SEPARATE run with its own frozen config — and those configs are
only preserved inside each run's ``run_meta.json``.

So this replays them: read each historical run's config snapshot, point the
GT at the requested split, and re-run. That keeps every arm byte-identical
in configuration to the one it is being compared against, which a hand-built
set of YAML files would not.

It refuses to run an arm whose detection cache does not cover the split,
rather than silently scoring whichever subset happens to be present — a
partial arm looks like a real result and is not.

Usage:
    python scripts/regen_on_split.py --split test --out results/paper_test
    python scripts/regen_on_split.py --split test --only seeds
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.splits import GT_DIRS, load_stems  # noqa: E402

# label -> historical run whose config snapshot defines the arm.
# Order is the cumulative ablation order the paper reports.
ABLATION = [
    ("v1_classical_directional", "results/ablations_1024/canny_mk4"),
    ("v2_ink_boundary_snap", "results/ablations_1024/abl_ink_boundary_v2"),
    ("v3_plus_stitching", "results/ablations_1024/abl_stitch_v2"),
    ("v4_plus_crossover", "results/ablations_1024/abl_crossover_v2"),
    ("v5_plus_crossover_DEFAULT", "results/benchmark_1024/parity19"),
    ("v6_plus_bridge_span7", "results/benchmark_1024/span7"),
    ("v7_plus_connectivity_repair", "results/benchmark_1024/span7_cr"),
    ("v8_plus_snap_expand80", "results/benchmark_1024/snapexp80"),
    ("v9_plus_blob_thresholds", "results/sweeps_bench/wires_binarize/otsu"),
    ("v10_plus_sauvola", "results/sweeps_bench/wires_binarize/sauvola"),
    ("v11_plus_class_head", "results/benchmark_1024/head95"),
    ("v12_plus_head_ensemble", "results/benchmark_1024_final/seed0"),
]

SEEDS = [(f"seed{i}", f"results/benchmark_1024_final/seed{i}") for i in range(3)]

# The three-seed runs differ ONLY by detector weights, but their config
# snapshots all record seed0's weights path — the seed1/seed2 caches were
# built by overriding weights at detection time without updating the config
# that got written. Replaying from the snapshot alone therefore fills those
# caches with seed0 boxes and produces three identical "seeds". Pin the
# weights here instead of trusting the snapshot.
SEED_WEIGHTS = {
    "seed0": "experiments/train_all/runs/yolov8s_640_seed0/weights/best.pt",
    "seed1": "experiments/train_all/runs/yolov8s_640_seed1/weights/best.pt",
    "seed2": "experiments/train_all/runs/yolov8s_640_seed2/weights/best.pt",
}


def config_of(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    cfg = meta.get("config")
    if not cfg:
        raise SystemExit(f"{run_dir}/run_meta.json has no config snapshot")
    return cfg


def cache_covers(cfg: dict, stems: list[str]) -> tuple[int, str]:
    cache = ROOT / cfg["detect"]["cache_dir"]
    have = sum(1 for s in stems if (cache / f"{s}.json").exists())
    return have, cfg["detect"]["cache_dir"]


def fill_cache(cfg: dict, split: str, tmp: Path) -> None:
    """Run detection for an arm whose cache is short, using that arm's own
    weights — otherwise the arm would be scored on another model's boxes."""
    p = tmp / "detect_cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run(
        [sys.executable, "scripts/detect_batch.py",
         "--images", f"data/splits/{split}.txt", "--config", str(p), "--sleep", "0"],
        cwd=ROOT, check=True)


def run_arm(label: str, src: Path, split: str, out_root: Path,
            stems: list[str], tmp: Path, fill: bool, no_spice: bool,
            detector: tuple[str, str] | None = None) -> dict | None:
    cfg = config_of(src)
    if label in SEED_WEIGHTS:
        cfg["detect"]["weights"] = str(ROOT / SEED_WEIGHTS[label])
    if detector is not None:
        # Pin EVERY arm to one detector. Without this the ablation confounds
        # two things: the pipeline stage being added, and whatever detector
        # that arm's historical snapshot happened to record. Pinning makes it
        # a pipeline-stage ablation on a fixed front end, which is what the
        # cumulative table claims to be.
        w, cache = detector
        cfg["detect"]["weights"] = str(ROOT / w)
        cfg["detect"]["cache_dir"] = cache
    have, cache = cache_covers(cfg, stems)
    if have < len(stems):
        if not fill:
            print(f"  {label:30s} SKIP — {cache} covers {have}/{len(stems)}; "
                  f"pass --fill-caches to detect the rest")
            return None
        print(f"  {label:30s} cache {cache} short ({have}/{len(stems)}), detecting…")
        fill_cache(cfg, split, tmp)

    cfg_path = tmp / f"{label}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = out_root / label
    cmd = [sys.executable, "scripts/benchmark.py", "--split", split,
           "--config", str(cfg_path), "--gt-dir", GT_DIRS[split],
           "--out-dir", str(out)]
    if no_spice:
        cmd.append("--no-spice")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {label:30s} FAILED\n{r.stdout[-800:]}{r.stderr[-800:]}")
        return None
    s = json.loads((out / "summary.json").read_text())
    t = s["topology"]
    print(f"  {label:30s} n={s['scored']:3d}  strict={t['strict_success']['mean']:.4f}  "
          f"tpF1={t['terminal_pair_f1']['mean']:.4f}  netF1={t['net_f1']['mean']:.4f}")
    return {"label": label, "source_run": str(src), "out_dir": str(out), **s}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default=None, help="default results/paper_<split>")
    ap.add_argument("--only", choices=["seeds", "ablation"], default=None)
    ap.add_argument("--fill-caches", action="store_true",
                    help="run detection for arms whose cache is short")
    ap.add_argument("--no-spice", action="store_true")
    ap.add_argument("--pin-detector", metavar="WEIGHTS:CACHE_DIR", default=None,
                    help="force every arm onto one detector, e.g. "
                         "'experiments/train_valstop/runs/yolov8s_640_seed0/"
                         "weights/best.pt:data/detections_valstop'. Makes the "
                         "ablation a pipeline-stage ablation on a fixed front "
                         "end; without it each arm keeps its snapshot's detector")
    args = ap.parse_args()

    detector = None
    if args.pin_detector:
        w, _, cache = args.pin_detector.rpartition(":")
        if not w or not cache:
            raise SystemExit("--pin-detector must be WEIGHTS:CACHE_DIR")
        if not (ROOT / w).exists():
            raise SystemExit(f"weights not found: {w}")
        detector = (w, cache)
        print(f"pinning every arm to {w}\n  cache {cache}\n")

    stems = load_stems(args.split)
    out_root = Path(args.out or f"results/paper_{args.split}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"replaying onto split={args.split} ({len(stems)} images) "
          f"-> {out_root}\n")

    done = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for group, arms in (("seeds", SEEDS), ("ablation", ABLATION)):
            if args.only and args.only != group:
                continue
            print(f"{group}:")
            done[group] = [
                r for label, src in arms
                if (r := run_arm(label, ROOT / src, args.split, out_root / group,
                                 stems, tmp, args.fill_caches, args.no_spice,
                                 detector))
            ]
            print()

    (out_root / "index.json").write_text(json.dumps({
        "split": args.split, "n_images": len(stems),
        "gt_dir": GT_DIRS[args.split],
        "note": ("Each arm replays a historical run's frozen config snapshot "
                 "against this split; see scripts/regen_on_split.py."),
        "pinned_detector": (
            {"weights": detector[0], "cache_dir": detector[1],
             "why": "every arm forced onto one detector so the cumulative table "
                    "measures pipeline stages, not detector drift"}
            if detector else None),
        "arms": {g: [{k: v for k, v in r.items() if k != "config"} for r in rs]
                 for g, rs in done.items()},
    }, indent=1) + "\n")
    print(f"wrote {out_root}/index.json")


if __name__ == "__main__":
    main()
