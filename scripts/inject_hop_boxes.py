#!/usr/bin/env python3
"""Inject synthesised Wire Crossover boxes at hop candidates, at a chosen operating point.

The notch-and-relink machinery already knows what to do with a crossover box; the
only thing missing is a box at the unannotated hops. So candidates are written
into a detection cache as ``Wire Crossover`` boxes and the pipeline is left
unchanged, which keeps the intervention attributable: nothing else moves.

Two operating points matter and they bracket the answer:

  --all       split at EVERY candidate. Deliberately over-eager. If even this
              does not reduce welding, the candidates are in the wrong places and
              no classifier over them can help, whatever its accuracy.
  --min-detour / --require-hops-over
              the selective end, where precision is traded for recall.

This is the honest way to bound a detector before training one. A classifier can
only ever choose a subset of the candidates it is given, so sweeping the subset
directly measures the best any classifier could do -- without the confound of
whether the classifier itself is any good.

Usage:
    python scripts/inject_hop_boxes.py --all --out data/detections_1024_hopall
    python scripts/inject_hop_boxes.py --min-detour 1.4 --out data/det_hop14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hop_candidates import hop_candidates

from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="inject every candidate (the over-eager bound)")
    ap.add_argument("--min-detour", type=float, default=1.18)
    ap.add_argument("--require-hops-over", action="store_true")
    ap.add_argument("--box", type=int, default=26,
                    help="px side of the synthesised crossover box")
    ap.add_argument("--wins", type=int, nargs="*", default=[20, 30, 42])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    idir = Path(cfg["preprocess"]["images_dir"])
    cdir = Path(cfg["detect"]["cache_dir"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(f"data/splits/{args.split}.txt")
             if l.strip()]
    if args.limit:
        names = names[: args.limit]

    n_files = n_added = 0
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        cp, ip = cdir / f"{stem}.json", idir / nm
        if not (cp.exists() and ip.exists()):
            continue
        cache = json.loads(cp.read_text())
        dets = load_cached_detections(
            str(cp), min_confidence=cfg["detect"].get("confidence"))
        res = run_pipeline(str(ip), cfg, detections=dets)
        cands = hop_candidates(res["clean_wires"], cfg,
                               wins=tuple(args.wins),
                               min_detour=args.min_detour)
        for c in cands:
            if not args.all:
                if args.require_hops_over and not c["hops_over"]:
                    continue
                if c["detour"] < args.min_detour:
                    continue
            cache["detections"].append({
                "class": "Wire Crossover", "confidence": 0.99,
                "x": float(c["x"]), "y": float(c["y"]),
                "width": float(args.box), "height": float(args.box),
                "synthetic_hop": True, "detour": c["detour"],
                "hops_over": c["hops_over"],
            })
            n_added += 1
        (out / f"{stem}.json").write_text(json.dumps(cache) + "\n")
        n_files += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(names)}] injected {n_added}", flush=True)

    print(f"\nwrote {n_files} caches to {out}")
    print(f"injected {n_added} synthetic Wire Crossover boxes "
          f"({n_added/max(n_files,1):.1f} per image)")
    print(f"\nBenchmark this cache against the baseline. If the OVER-EAGER "
          f"setting\ndoes not reduce welding, the candidates are mislocated and "
          f"no classifier\nover them can help.")


if __name__ == "__main__":
    main()
