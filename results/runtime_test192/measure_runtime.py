#!/usr/bin/env python3
"""Per-stage runtime on the full 192-image TEST split, in two scopes.

WHY THIS EXISTS RATHER THAN scripts/benchmark_runtime.py
--------------------------------------------------------
Two defects make that script unable to answer the question asked:

1. ``--time-detector`` is a NO-OP. It sets ``model = True`` and then calls
   ``schematic2netlist.detect.detect()``, which returns the per-image cache
   whenever one exists (detect.py: ``if cache.exists(): return
   load_cached_detections(...)``). Every test image is cached, so the flag
   times a JSON read and reports it as detector inference. This is why
   results/runtime_1024/summary.json has ``detector_timed: false`` and a
   0.28 ms ``detect`` stage: the flag could not have helped.

2. Its stage list is NOT the pipeline. ``time_image`` re-implements a subset
   of ``run_pipeline`` and omits two stages that are ENABLED in the shipped
   config and DO run in every benchmark: the component class head
   (detect.class_head.enabled: true -- two CPU CNNs over every component
   crop) and constraint-triggered connectivity repair
   (connectivity_repair.enabled: true, which rebuilds nodes and snapping).
   A re-implementation can also drift from the pipeline silently.

This harness instruments ``run_pipeline`` ITSELF by wrapping the callables in
its module namespace with an EXCLUSIVE (self-time) timer, so nested work --
node building rebuilt inside connectivity repair, for instance -- is charged
to the stage that actually does it and counted exactly once. The stage times
therefore sum to the measured wall clock up to a small unattributed residual,
which is reported as ``other`` rather than hidden.

TWO SCOPES, MEASURED PAIRED PER IMAGE
-------------------------------------
- ``cached``  -- detections read from disk. This is what every experiment in
  this repo does, and what results/benchmark_test192 reports.
- ``e2e``     -- the YOLO detector actually runs. This is what a user
  experiences. The cache is bypassed by patching the detect entry point; the
  model is loaded ONCE (a server amortizes it) and the one-time load cost is
  reported separately. Nothing is written to the detection cache.

The two modes are interleaved image by image so that machine load drifting
during the run affects both equally and the comparison stays paired.

Usage:
    ./venv/bin/python results/runtime_test192/measure_runtime.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2

from schematic2netlist import detect as detect_mod
from schematic2netlist import pipeline as pipeline_mod
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.netlist import export_spice_netlist
from schematic2netlist.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]

# stages measured INSIDE run_pipeline, in execution order
INNER_STAGES = [
    "load", "detect", "class_head", "textmask", "wiremask", "wires",
    "stitch", "nodes", "snapping", "conn_repair", "netlist", "repair",
]
STAGES = INNER_STAGES + ["other", "export"]

# the stage subset scripts/benchmark_runtime.py measures. class_head,
# conn_repair and the residual are absent from it, so shares quoted against
# this denominator (the published 54% stitch share among them) are shares of
# a SMALLER pipeline than the one that actually runs.
LEGACY_STAGES = [
    "load", "detect", "textmask", "wiremask", "wires", "stitch",
    "nodes", "snapping", "netlist", "repair", "export",
]

# ---------------------------------------------------------------- timers ---
_acc: dict[str, float] = defaultdict(float)
_stack: list[float] = []


def _reset() -> None:
    _acc.clear()
    _stack.clear()


def _timed(name: str, fn):
    """Wrap fn so its EXCLUSIVE (self) time accumulates under `name`.

    Time spent inside other wrapped callables is subtracted, so a stage that
    calls another stage (connectivity repair rebuilding nodes + snapping) is
    not double counted and the stage times can be summed.
    """
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        _stack.append(0.0)
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            child = _stack.pop()
            _acc[name] += dt - child
            if _stack:
                _stack[-1] += dt
    wrapper.__wrapped__ = fn
    return wrapper


def install_instrumentation() -> None:
    """Wrap the callables run_pipeline actually invokes, in its namespace."""
    p = pipeline_mod
    p.cv2 = _Cv2Proxy(cv2)                       # load (imread only)
    detect_mod.detect = _timed("detect", detect_mod.detect)
    p.class_head_reclassify = _timed("class_head", p.class_head_reclassify)
    p.detect_text_mask = _timed("textmask", p.detect_text_mask)
    p.build_non_wire_mask = _timed("wiremask", p.build_non_wire_mask)
    p.extract_wires = _timed("wires", p.extract_wires)
    p.stitchable_mask = _timed("stitch", p.stitchable_mask)
    p.stitch_wire_islands = _timed("stitch", p.stitch_wire_islands)
    p.build_wire_nodes = _timed("nodes", p.build_wire_nodes)
    p.build_wire_nodes_crossover_aware = _timed(
        "nodes", p.build_wire_nodes_crossover_aware)
    p.build_wire_nodes_learned = _timed("nodes", p.build_wire_nodes_learned)
    p.build_component_pin_nets = _timed("snapping", p.build_component_pin_nets)
    p.repair_connectivity = _timed("conn_repair", p.repair_connectivity)
    p.build_node_name_map = _timed("netlist", p.build_node_name_map)
    p.assign_node_names = _timed("netlist", p.assign_node_names)
    p.repair_circuit = _timed("repair", p.repair_circuit)


class _Cv2Proxy:
    """Charges only cv2.imread to `load`; everything else passes through.

    cvtColor is deliberately NOT wrapped: it is called from inside other
    stages too, and charging those calls to `load` would misattribute them.
    The main frame's cvtColor lands in the `other` residual.
    """

    def __init__(self, real):
        self._real = real
        self.imread = _timed("load", real.imread)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ------------------------------------------------------- real detection ---
class RealDetector:
    """YOLO inference that BYPASSES the cache and never writes to it.

    Mirrors detect.detect_ultralytics exactly (same conf, imgsz, box
    parsing) but keeps the loaded model so per-image time is inference,
    not repeated model construction. The one-time load is timed separately.
    """

    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        t0 = time.perf_counter()
        self.model = YOLO(cfg["detect"]["weights"])
        self.load_s = time.perf_counter() - t0
        self.conf = cfg["detect"]["confidence"]
        self.imgsz = cfg["detect"]["image_size"]

    def __call__(self, image_path, cfg):
        res = self.model.predict([str(image_path)], conf=self.conf,
                                 imgsz=self.imgsz, verbose=False)[0]
        dets = []
        names = res.names
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "class": canonical_class(names[int(box.cls[0])]),
                "confidence": float(box.conf[0]),
                "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                "width": x2 - x1, "height": y2 - y1,
            })
        return dets


# ------------------------------------------------------------- one image ---
def time_one(image_path: Path, cfg: dict) -> dict:
    _reset()
    t0 = time.perf_counter()
    result = run_pipeline(image_path, cfg, detections=None, out_dir=None)
    wall = time.perf_counter() - t0

    row = {s: _acc.get(s, 0.0) for s in INNER_STAGES}
    row["other"] = max(0.0, wall - sum(row.values()))

    # netlist export, timed the same way scripts/benchmark_runtime.py does it
    # (run_pipeline only writes when out_dir is given, and writing debug PNGs
    # would not be a pipeline cost)
    comps = result["components"]
    rep = result.get("repair")
    with tempfile.NamedTemporaryFile("w", suffix=".sp", delete=False) as fh:
        sp = fh.name
    t0 = time.perf_counter()
    export_spice_netlist(comps, sp, placeholders=cfg["netlist"]["placeholders"],
                         extra_lines=rep.extra_lines if rep else None)
    row["export"] = time.perf_counter() - t0
    Path(sp).unlink(missing_ok=True)

    row["total"] = sum(row[s] for s in STAGES)
    row["n_components"] = len(comps)
    row["n_detections"] = len(result["detections"])
    return row, result


def summarize(rows: list[dict]) -> dict:
    out = {}
    for stage in STAGES + ["total"]:
        vals = sorted(r[stage] for r in rows)
        out[stage] = {
            "mean_ms": round(1000 * statistics.mean(vals), 2),
            "median_ms": round(1000 * statistics.median(vals), 2),
            "p90_ms": round(1000 * vals[int(0.9 * (len(vals) - 1))], 2),
            "share_of_total_mean": None,
            "share_of_total_median": None,
        }
    tm, tmed = out["total"]["mean_ms"], out["total"]["median_ms"]
    for stage in STAGES:
        out[stage]["share_of_total_mean"] = round(
            out[stage]["mean_ms"] / tm, 4) if tm else None
        out[stage]["share_of_total_median"] = round(
            out[stage]["median_ms"] / tmed, 4) if tmed else None
    return out


COLD_PROBE = """
import time, sys
t0 = time.perf_counter()
from pathlib import Path
from schematic2netlist.config import load_config
from schematic2netlist.detect import detect_ultralytics
cfg = load_config(None)
t_import = time.perf_counter() - t0
t1 = time.perf_counter()
detect_ultralytics([Path(sys.argv[1])], cfg)
print(t_import, time.perf_counter() - t1)
"""


def measure_cold_start(image: Path, repeats: int = 3) -> dict:
    """What one image costs from a cold `python -c` -- the CLI case.

    Steady-state per-image figures amortize interpreter start, the torch and
    ultralytics imports, and the first forward pass. A user converting ONE
    schematic pays all of it, and it dominates everything else, so it is
    measured rather than waved at. Each repeat is a fresh interpreter.
    """
    imports, first = [], []
    for _ in range(repeats):
        r = subprocess.run([sys.executable, "-c", COLD_PROBE, str(image)],
                           cwd=REPO_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return {"error": r.stderr[-400:]}
        a, b = r.stdout.strip().split()[-2:]
        imports.append(float(a))
        first.append(float(b))
    return {
        "repeats": repeats,
        "import_and_config_median_ms": round(1000 * statistics.median(imports),
                                             1),
        "first_detect_call_median_ms": round(1000 * statistics.median(first),
                                             1),
        "total_median_ms": round(
            1000 * statistics.median(i + f for i, f in zip(imports, first)), 1),
        "note": "fresh interpreter per repeat; includes the torch/ultralytics "
                "import, model construction and the first forward pass",
    }


def thirds_stability(rows: list[dict]) -> dict:
    """Did machine load drift during the run, or is the spread just content?

    Other agents were active throughout, so the timings could in principle be
    a record of the machine getting busier rather than of the pipeline. Split
    the run into thirds and compare a CONTENT-NORMALISED cost (class-head ms
    per detection) and a content-light stage (wire extraction). If those hold
    steady while raw per-image totals move, the movement is image content, and
    the ordering of the stage table is safe to read.
    """
    def block(sub):
        big = [r for r in sub if r["n_detections"] > 8]
        return {
            "n": len(sub),
            "median_n_detections": statistics.median(
                r["n_detections"] for r in sub),
            "median_total_ms": round(
                1000 * statistics.median(r["total"] for r in sub), 2),
            "median_wires_ms": round(
                1000 * statistics.median(r["wires"] for r in sub), 2),
            "median_class_head_ms_per_detection": round(
                1000 * statistics.median(
                    r["class_head"] / r["n_detections"] for r in big), 3)
            if big else None,
        }
    k = len(rows) // 3
    return {
        "first_third": block(rows[:k]),
        "middle_third": block(rows[k:2 * k]),
        "last_third": block(rows[2 * k:]),
        "reading": "raw per-image totals move with image content; the "
                   "content-normalised costs should not move if machine load "
                   "was stable enough for the stage ordering to be trusted",
    }


def loadavg() -> list[float]:
    try:
        return list(os.getloadavg())
    except OSError:  # pragma: no cover
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--out-dir", default="results/runtime_test192")
    ap.add_argument("--config", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = (Path(args.splits_dir) / f"{args.split}.txt").read_text().split()
    if args.limit:
        names = names[: args.limit]
    images_dir = resolve_and_check(args.images_dir, names, cfg)

    install_instrumentation()
    real = RealDetector(cfg)
    print(f"detector model load (one-time): {real.load_s * 1000:.0f} ms",
          flush=True)

    cached_detect = detect_mod.detect          # already instrumented
    e2e_detect = _timed("detect", real)

    cold = measure_cold_start(images_dir / names[0])
    print(f"cold start, one image from a fresh interpreter: "
          f"{cold.get('total_median_ms')} ms", flush=True)

    load_start = loadavg()
    rows_cached: list[dict] = []
    rows_e2e: list[dict] = []
    det_shipped: list[float] = []
    det_agree = {"images": 0, "same_count": 0, "same_boxes_1px": 0,
                 "same_classes": 0}

    for i, nm in enumerate(names):
        # --- cached-detection scope ---
        detect_mod.detect = cached_detect
        r_c, res_c = time_one(images_dir / nm, cfg)
        # --- true end-to-end scope (detector runs, model held in memory) ---
        detect_mod.detect = e2e_detect
        r_e, res_e = time_one(images_dir / nm, cfg)
        # --- the detector call EXACTLY as the shipped code invokes it:
        # detect() -> detect_ultralytics([one image]) -> YOLO(weights) is
        # reconstructed on every call, so this includes model construction.
        t0 = time.perf_counter()
        detect_mod.detect_ultralytics([images_dir / nm], cfg)
        dt_shipped = time.perf_counter() - t0

        if i < args.warmup:      # discard: first calls pay import/JIT costs
            continue
        det_shipped.append(dt_shipped)
        for r, bucket in ((r_c, rows_cached), (r_e, rows_e2e)):
            r["image"] = nm
            bucket.append(r)

        # does the live detector reproduce the cache it is being compared to?
        a = sorted((d["class"], round(d["x"], 1), round(d["y"], 1))
                   for d in res_c["detections"])
        b = sorted((d["class"], round(d["x"], 1), round(d["y"], 1))
                   for d in res_e["detections"])
        det_agree["images"] += 1
        det_agree["same_count"] += int(len(a) == len(b))
        det_agree["same_boxes_1px"] += int(
            len(a) == len(b) and all(
                x[0] == y[0] and abs(x[1] - y[1]) < 1 and abs(x[2] - y[2]) < 1
                for x, y in zip(a, b)))
        det_agree["same_classes"] += int(
            sorted(x[0] for x in a) == sorted(y[0] for y in b))

        print(f"[{len(rows_cached)}/{len(names) - args.warmup}] {nm} "
              f"cached {r_c['total'] * 1000:7.0f} ms | "
              f"e2e {r_e['total'] * 1000:7.0f} ms", flush=True)

    load_end = loadavg()

    for mode, rows in (("cached", rows_cached), ("e2e", rows_e2e)):
        with (out_dir / f"per_image_{mode}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["image"] + STAGES
                               + ["total", "n_components", "n_detections"])
            w.writeheader()
            w.writerows(rows)

    s_cached, s_e2e = summarize(rows_cached), summarize(rows_e2e)

    # stitch's share of the LEGACY denominator, so the published 54% can be
    # checked on its own terms as well as against the real pipeline
    legacy_tot = [sum(r[s] for s in LEGACY_STAGES) for r in rows_cached]
    legacy_stitch = {
        "n_images": len(rows_cached),
        "legacy_subset_total_mean_ms": round(
            1000 * statistics.mean(legacy_tot), 2),
        "legacy_subset_total_median_ms": round(
            1000 * statistics.median(legacy_tot), 2),
        "stitch_share_mean": round(
            statistics.mean(r["stitch"] for r in rows_cached)
            / statistics.mean(legacy_tot), 4),
        "stitch_share_median": round(
            statistics.median(r["stitch"] for r in rows_cached)
            / statistics.median(legacy_tot), 4),
        "note": "denominator excludes class_head, conn_repair and the "
                "unattributed residual, matching scripts/benchmark_runtime.py",
    }

    summary = {
        "split": args.split,
        "n_images": len(rows_cached),
        "warmup_discarded": args.warmup,
        "supersedes": {
            "file": "results/runtime_1024/summary.json",
            "why": "that run records detector_timed: false, so its 0.28 ms "
                   "'detect' stage is a cache read and its ~46 ms total is "
                   "not an end-to-end latency. Its stage list also omits the "
                   "class head and connectivity repair, both enabled in the "
                   "shipped config, and it covers 60 images of the split now "
                   "called val.",
            "quote_instead": "the e2e scope below for user-facing latency; "
                             "the cached scope only when describing what the "
                             "experiments in this repository cost to run",
        },
        "scopes": {
            "cached": {
                "detector_timed": False,
                "what_it_measures": "detections read from the per-image JSON "
                                    "cache; the scope every experiment in this "
                                    "repo runs at",
                "stages": s_cached,
            },
            "e2e": {
                "detector_timed": True,
                "what_it_measures": "YOLOv8s inference actually runs on the "
                                    "1024 px frame at imgsz 640; the scope a "
                                    "user experiences",
                "detector_model_load_once_ms": round(real.load_s * 1000, 1),
                "stages": s_e2e,
            },
        },
        "e2e_over_cached_total": {
            "mean_x": round(s_e2e["total"]["mean_ms"]
                            / s_cached["total"]["mean_ms"], 2),
            "median_x": round(s_e2e["total"]["median_ms"]
                              / s_cached["total"]["median_ms"], 2),
            "median_delta_ms": round(s_e2e["total"]["median_ms"]
                                     - s_cached["total"]["median_ms"], 2),
        },
        "detector_only_ms": {
            "inference_model_preloaded": {
                "mean": s_e2e["detect"]["mean_ms"],
                "median": s_e2e["detect"]["median_ms"],
                "p90": s_e2e["detect"]["p90_ms"],
                "note": "what a served system pays per image",
            },
            "as_shipped_model_rebuilt_per_call": {
                "mean": round(1000 * statistics.mean(det_shipped), 2),
                "median": round(1000 * statistics.median(det_shipped), 2),
                "p90": round(1000 * sorted(det_shipped)[
                    int(0.9 * (len(det_shipped) - 1))], 2),
                "note": "detect() -> detect_ultralytics([one image]) "
                        "reconstructs YOLO(weights) on every call",
            },
            "model_construction_ms_warm_process": round(
                real.load_s * 1000, 1),
            "cold_start_one_image_cli": cold,
        },
        "detector_vs_cache_agreement": det_agree,
        "stitch_share_of_cached_total": {
            "mean": s_cached["stitch"]["share_of_total_mean"],
            "median": s_cached["stitch"]["share_of_total_median"],
        },
        "stitch_share_legacy_scope": legacy_stitch,
        "concurrency_caveat":
            "Other agents were running on this machine throughout. Wall-clock "
            "timings are NOISY and inflated relative to an idle host; PREFER "
            "THE MEDIANS. The cached and e2e scopes were measured interleaved "
            "image by image, so their RATIO is paired and far more robust than "
            "either absolute figure.",
        "load_stability_check": thirds_stability(rows_cached),
        "load_average_1_5_15": {"start": load_start, "end": load_end},
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    write_run_metadata(out_dir, cfg, seed, extra={
        "split": args.split,
        "n_timed": len(rows_cached),
        "warmup_discarded": args.warmup,
        "scopes_measured": ["cached", "e2e"],
        "detector_timed": "e2e scope only",
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "load_average_start": load_start,
        "load_average_end": load_end,
        "harness": "results/runtime_test192/measure_runtime.py",
        "why_not_benchmark_runtime": (
            "scripts/benchmark_runtime.py --time-detector is a no-op: it calls "
            "detect.detect(), which returns the cache when one exists. Its "
            "stage list also omits class_head and connectivity_repair, both "
            "enabled in the shipped config."),
    })

    def show(title, s):
        print(f"\n{title} over {len(rows_cached)} images")
        print(f"  {'stage':11s} {'mean ms':>9s} {'median':>9s} {'p90':>9s} "
              f"{'share(med)':>11s}")
        for stage in STAGES:
            v = s[stage]
            print(f"  {stage:11s} {v['mean_ms']:9.2f} {v['median_ms']:9.2f} "
                  f"{v['p90_ms']:9.2f} {v['share_of_total_median']:11.1%}")
        v = s["total"]
        print(f"  {'TOTAL':11s} {v['mean_ms']:9.2f} {v['median_ms']:9.2f} "
              f"{v['p90_ms']:9.2f}")

    show("CACHED DOWNSTREAM (detector NOT run)", s_cached)
    show("TRUE END-TO-END (detector run)", s_e2e)
    print(f"\ne2e / cached: {summary['e2e_over_cached_total']['median_x']}x "
          f"(median), +{summary['e2e_over_cached_total']['median_delta_ms']:.0f} ms")
    print(f"detector alone: preloaded {s_e2e['detect']['median_ms']:.0f} ms | "
          f"as shipped (model rebuilt per call) "
          f"{summary['detector_only_ms']['as_shipped_model_rebuilt_per_call']['median']:.0f} ms | "
          f"warm model construction {real.load_s * 1000:.0f} ms | "
          f"cold start (one image, fresh interpreter) "
          f"{cold.get('total_median_ms')} ms")
    print(f"stitch share, full pipeline (median): "
          f"{s_cached['stitch']['share_of_total_median']:.1%}")
    print(f"stitch share, legacy stage subset (median): "
          f"{legacy_stitch['stitch_share_median']:.1%}")
    print(f"\nwrote {out_dir}/summary.json + per_image_{{cached,e2e}}.csv")


if __name__ == "__main__":
    main()
