#!/usr/bin/env python3
"""Is the pipeline deterministic? Measure it, do not assume it.

The pipeline SHOULD be deterministic by construction -- seeded, no sampling --
but "should" is not a measurement, and determinism is a claim that survives
independently of the host, unlike latency. It is also the axis on which a
frontier-model baseline cannot compete: a hosted model is sampled, so a single
run of it is one draw from a distribution we never observe, while a single run
of this pipeline is the whole distribution IF that is verified.

WHAT IS MEASURED, over the full test split, 5 independent runs
--------------------------------------------------------------
1. exact-output agreement -- fraction of circuits whose emitted SPICE netlist
   is BYTE-identical across all runs. Uses the real export_spice_netlist, both
   the base netlist and the repaired one.
2. topology changes -- circuits whose predicted topology differs between runs.
   Compared as a NAMING-INVARIANT partition of component terminals into nets,
   so a mere renumbering of nodes is not counted as a change and a genuine
   rewiring is.
3. headline metric variance across runs -- strict success, net F1,
   terminal-pair F1, nGED.
4. invalid-output frequency -- pipeline exceptions and netlists with no
   emitted element lines.
5. full-result agreement -- a digest of the ENTIRE run_pipeline return dict
   (detections, components, node_map and clean_wires label images,
   node_name_map, coverage, repair ledger, class-head report, connectivity
   repair report, junction info), excluding only the keys that legitimately
   vary (the input path and the output directory). A stable netlist cannot
   hide a wobbling intermediate behind it.

WHY SUBPROCESSES, AND WHY DIFFERENT HASH SEEDS
----------------------------------------------
Five loops inside one interpreter is a weak test. Python fixes its string-hash
seed at interpreter startup, so set-iteration and dict-insertion order are
CONSTANT within a process no matter how many times you loop -- any ordering
nondeterminism would be invisible. determinism.set_global_seed sets
PYTHONHASHSEED in os.environ, which has no effect on the already-running
interpreter, so in real use the hash seed is whatever the shell supplied
(unset = randomized per process). Each run here is therefore a FRESH
interpreter with an EXPLICITLY DIFFERENT PYTHONHASHSEED (0..4): reproducible,
and strictly harder to pass than repeating one process.

COST
----
Runs with SPICE disabled by default (--spice to enable): ngspice adds two
subprocess launches per circuit per run and answers a question the byte
comparison already settles -- byte-identical netlists cannot produce different
ngspice verdicts. NO paid API is called; the frontier-model baselines are read
from committed results only, never re-run.

Usage:
    ./venv/bin/python scripts/measure_determinism.py                # 5 runs
    ./venv/bin/python scripts/measure_determinism.py --runs 3 --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXCLUDE_TOP_KEYS = {"image", "out_dir"}   # input path / output path only


# ------------------------------------------------------------ digesting ---
def _canon(obj, path="", depth=0):
    """Canonical, order-stable byte encoding of an arbitrary result value."""
    if isinstance(obj, np.ndarray):
        return b"|ndarray:" + str(obj.shape).encode() + str(obj.dtype).encode() \
            + hashlib.sha256(np.ascontiguousarray(obj).tobytes()).digest()
    if isinstance(obj, (np.generic,)):
        return b"|np:" + repr(obj.item()).encode()
    if isinstance(obj, dict):
        return b"{" + b",".join(
            repr(k).encode() + b":" + _canon(v, f"{path}.{k}", depth + 1)
            for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(_canon(v, f"{path}[{i}]", depth + 1)
                                for i, v in enumerate(obj)) + b"]"
    if isinstance(obj, (set, frozenset)):
        return b"S[" + b",".join(sorted(_canon(v) for v in obj)) + b"]"
    if is_dataclass(obj) and not isinstance(obj, type):
        return b"D:" + type(obj).__name__.encode() + _canon(vars(obj), path,
                                                            depth + 1)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return b"O:" + type(obj).__name__.encode() + _canon(vars(obj), path,
                                                            depth + 1)
    return repr(obj).encode()


def digest(obj) -> str:
    return hashlib.sha256(_canon(obj)).hexdigest()


def deep_diff(a, b, path="result", out=None, limit=40):
    """Structural diff of two result dicts; returns a list of differing paths."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        elif a.shape != b.shape:
            out.append(f"{path}: shape {a.shape} vs {b.shape}")
        elif not np.array_equal(a, b):
            n = int((a != b).sum())
            out.append(f"{path}: {n} of {a.size} cells differ")
        return out
    if is_dataclass(a) and is_dataclass(b):
        return deep_diff(vars(a), vars(b), path, out, limit)
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=repr):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present in only one run")
            else:
                deep_diff(a[k], b[k], f"{path}.{k}", out, limit)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", out, limit)
        return out
    try:
        same = bool(a == b)
    except Exception:
        same = repr(a) == repr(b)
    if not same:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


# ---------------------------------------------------------- signatures ----
def topology_signature(components: list[dict]) -> list:
    """Naming-invariant partition of component terminals into nets.

    A run that renumbers n3 -> n7 everywhere has not changed the recovered
    circuit and must not be counted as a change; a run that moves one terminal
    onto a different net has, and is. Unsnapped terminals form singletons, so
    losing or gaining a connection registers.
    """
    groups: dict[str, list] = {}
    for c in components:
        for t, net in enumerate(c.get("node_names", [])):
            key = f"net:{net}" if net is not None else f"open:{c['id']}:{t}"
            groups.setdefault(key, []).append([int(c["id"]), int(t)])
    return sorted([sorted(g) for g in groups.values()])


def label_signature(components: list[dict]) -> list:
    return sorted([int(c["id"]), str(c["class"])] for c in components)


# ------------------------------------------------------------- one run ----
def do_run(args) -> dict:
    """One full pass over the split in this (fresh) interpreter."""
    # BEFORE anything can touch it: determinism.set_global_seed assigns
    # os.environ["PYTHONHASHSEED"] = str(cfg seed), which cannot change the
    # running interpreter's already-fixed hash seed but does destroy the
    # record of what it actually was. Capture it first, and probe the live
    # hash so the seed is not merely asserted -- a differing probe across
    # runs is proof that set/dict ordering really was perturbed.
    hashseed_env = os.environ.get("PYTHONHASHSEED")
    hash_probe = hash("schematic2netlist")

    from schematic2netlist.benchmark import aggregate, score_prediction
    from schematic2netlist.config import config_hash, load_config
    from schematic2netlist.detect import load_cached_detections
    from schematic2netlist.determinism import set_global_seed
    from schematic2netlist.frames import resolve_and_check
    from schematic2netlist.gt import gt_to_components, load_gt
    from schematic2netlist.netlist import export_spice_netlist
    from schematic2netlist.pipeline import run_pipeline

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    gt_dir = Path(args.gt_dir or cfg["benchmark"]["gt_dir"])
    det_dir = Path(cfg["detect"]["cache_dir"])
    names = (Path(args.splits_dir) / f"{args.split}.txt").read_text().split()
    if args.limit:
        names = names[: args.limit]
    images_dir = resolve_and_check(args.images_dir, names, cfg)
    ph = cfg["netlist"]["placeholders"]

    def netlist_bytes(comps, extra=None) -> tuple[bytes, dict]:
        with tempfile.NamedTemporaryFile("w", suffix=".sp", delete=False) as fh:
            p = fh.name
        info = export_spice_netlist(comps, p, placeholders=ph, extra_lines=extra)
        data = Path(p).read_bytes()
        Path(p).unlink(missing_ok=True)
        return data, info

    records: dict[str, dict] = {}
    rows: list[dict] = []
    full_dump: dict[str, dict] = {}
    sample = set(names[:: max(1, len(names) // max(1, args.full_sample))][
        : args.full_sample])

    t_start = time.perf_counter()
    for idx, name in enumerate(names, 1):
        stem = Path(name).stem
        gt_path, det_path = gt_dir / f"{stem}.json", det_dir / f"{stem}.json"
        if not gt_path.exists() or not det_path.exists():
            continue
        gt = load_gt(gt_path)
        if not gt.get("verified") and not args.include_unverified:
            continue

        rec: dict = {}
        try:
            detections = load_cached_detections(
                det_path, min_confidence=cfg["detect"].get("confidence"))
            result = run_pipeline(images_dir / name, cfg, detections=detections)
        except Exception as e:                       # invalid output: crash
            rec["error"] = f"{type(e).__name__}: {e}"
            records[name] = rec
            continue

        comps = result["components"]
        rep = result.get("repair")
        base, base_info = netlist_bytes(comps)
        repaired, _ = netlist_bytes(
            comps, rep.extra_lines if rep is not None else None)

        rec["netlist_base_sha"] = hashlib.sha256(base).hexdigest()
        rec["netlist_repaired_sha"] = hashlib.sha256(repaired).hexdigest()
        rec["netlist_base_bytes"] = len(base)
        rec["topology_sha"] = digest(topology_signature(comps))
        rec["labels_sha"] = digest(label_signature(comps))
        rec["full_result_sha"] = digest(
            {k: v for k, v in result.items() if k not in EXCLUDE_TOP_KEYS})
        rec["n_components"] = len(comps)
        rec["n_wire_nodes"] = int(result["num_wire_nodes"])
        rec["wrote_any"] = bool(base_info.get("wrote_any"))
        rec["n_skipped_elements"] = len(base_info.get("skipped", []))
        # invalid output = crashed, or a netlist carrying no element at all
        rec["invalid"] = (not rec["wrote_any"])

        # per-image metrics, scored exactly as scripts/benchmark.py does
        dets = result["detections"]
        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                          dets[c["id"]]["width"], dets[c["id"]]["height"]]}
                for c in comps]
        gt_comps = gt_to_components(gt)
        by_id = {c["id"]: c for c in gt["components"]}
        for c in gt_comps:
            c["bbox"] = by_id[c["id"]]["bbox"]
        row = score_prediction(pred, gt_comps, iou_threshold=args.iou_threshold)
        rec["metrics"] = {k: (float(v) if not isinstance(v, bool) else int(v))
                          for k, v in row.items()}
        rows.append(row)
        records[name] = rec

        if name in sample and args.full_dump:
            full_dump[name] = {k: v for k, v in result.items()
                               if k not in EXCLUDE_TOP_KEYS}
        if idx % 25 == 0:
            print(f"  run {args.run_index}: {idx}/{len(names)}", flush=True)

    agg = aggregate(rows, seed=seed) if rows else {}
    out = {
        "run_index": args.run_index,
        "pythonhashseed": hashseed_env,
        "str_hash_probe": hash_probe,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "pid": os.getpid(),
        "config_hash": config_hash(cfg),
        "seed": seed,
        "split": args.split,
        "n_scored": len(rows),
        "wall_s": round(time.perf_counter() - t_start, 2),
        "aggregate": {k: (v["mean"] if isinstance(v, dict) else v)
                      for k, v in agg.items()},
        "records": records,
    }
    Path(args.record_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record_out).write_text(json.dumps(out, indent=1, sort_keys=True))
    if args.full_dump:
        with open(args.full_dump, "wb") as fh:
            pickle.dump(full_dump, fh)
    print(f"  run {args.run_index} done: {len(rows)} scored in "
          f"{out['wall_s']}s -> {args.record_out}", flush=True)
    return out


# -------------------------------------------------------------- driver ----
def spawn(args, run_index: int, record_out: Path, full_dump: Path | None):
    env = dict(os.environ)
    # different hash seed per run: reproducible, and a strictly harder test
    env["PYTHONHASHSEED"] = str(run_index)
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", "--run-index", str(run_index),
           "--record-out", str(record_out),
           "--split", args.split, "--splits-dir", args.splits_dir,
           "--iou-threshold", str(args.iou_threshold),
           "--full-sample", str(args.full_sample)]
    if args.config:
        cmd += ["--config", args.config]
    if args.images_dir:
        cmd += ["--images-dir", args.images_dir]
    if args.gt_dir:
        cmd += ["--gt-dir", args.gt_dir]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.include_unverified:
        cmd += ["--include-unverified"]
    if full_dump:
        cmd += ["--full-dump", str(full_dump)]
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"run {run_index} failed with code {r.returncode}")


def spread(vals: list[float]) -> dict:
    return {
        "values": [round(v, 12) for v in vals],
        "mean": statistics.mean(vals),
        "stdev": statistics.pstdev(vals),
        "variance": statistics.pvariance(vals),
        "min": min(vals), "max": max(vals), "range": max(vals) - min(vals),
        "all_identical": len(set(vals)) == 1,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out-dir", default="results/determinism")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--full-sample", type=int, default=8,
                    help="images whose FULL result dict is dumped and diffed")
    ap.add_argument("--dump-dir", default=None,
                    help="where full-result pickles go (default: a temp dir; "
                         "they are large and are not results)")
    ap.add_argument("--spice", action="store_true",
                    help="also run ngspice (off by default: byte-identical "
                         "netlists cannot yield different ngspice verdicts)")
    # worker plumbing
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-index", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--record-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--full-dump", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        do_run(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_dir or tempfile.mkdtemp(prefix="determinism_"))
    dump_dir.mkdir(parents=True, exist_ok=True)

    load_start = list(os.getloadavg())
    t0 = time.perf_counter()
    runs = []
    for k in range(args.runs):
        rec_path = out_dir / "runs" / f"run{k}.json"
        # full result dicts are dumped for the first two runs only; that is
        # all a pairwise structural diff needs and they are ~100 MB each
        dump = dump_dir / f"full_run{k}.pkl" if k < 2 else None
        print(f"[{k + 1}/{args.runs}] fresh interpreter, PYTHONHASHSEED={k}",
              flush=True)
        spawn(args, k, rec_path, dump)
        runs.append(json.loads(rec_path.read_text()))
    load_end = list(os.getloadavg())

    images = sorted(set().union(*(set(r["records"]) for r in runs)))
    per_image, agree = {}, {
        "netlist_base": 0, "netlist_repaired": 0, "topology": 0,
        "labels": 0, "full_result": 0,
    }
    topo_changed, netlist_changed, errored, invalid_any = [], [], [], []
    invalid_counts = []

    for img in images:
        recs = [r["records"].get(img) for r in runs]
        if any(rc is None for rc in recs):
            errored.append(img)
            continue
        if any("error" in rc for rc in recs):
            errored.append(img)
            continue
        same = {}
        for field, key in (("netlist_base", "netlist_base_sha"),
                           ("netlist_repaired", "netlist_repaired_sha"),
                           ("topology", "topology_sha"),
                           ("labels", "labels_sha"),
                           ("full_result", "full_result_sha")):
            vals = {rc[key] for rc in recs}
            same[field] = len(vals) == 1
            agree[field] += int(same[field])
        if not same["topology"]:
            topo_changed.append(img)
        if not (same["netlist_base"] and same["netlist_repaired"]):
            netlist_changed.append(img)
        n_invalid = sum(int(rc["invalid"]) for rc in recs)
        invalid_counts.append(n_invalid)
        if n_invalid:
            invalid_any.append(img)
        per_image[img] = same

    n = len(per_image)
    metric_keys = ["strict_success", "net_f1", "terminal_pair_f1", "nged",
                   "per_component_connected_acc", "per_component_recall_acc"]
    metric_spread = {k: spread([r["aggregate"][k] for r in runs])
                     for k in metric_keys if k in runs[0]["aggregate"]}

    # ---- the stricter check: structural diff of the FULL result dicts ----
    full_diff = {"performed": False}
    d0, d1 = dump_dir / "full_run0.pkl", dump_dir / "full_run1.pkl"
    if d0.exists() and d1.exists():
        a = pickle.loads(d0.read_bytes())
        b = pickle.loads(d1.read_bytes())
        shared = sorted(set(a) & set(b))
        diffs = {}
        for img in shared:
            d = deep_diff(a[img], b[img], path=f"result[{img}]")
            if d:
                diffs[img] = d
        full_diff = {
            "performed": True,
            "runs_compared": [0, 1],
            "pythonhashseeds": [runs[0]["pythonhashseed"],
                                runs[1]["pythonhashseed"]],
            "n_images": len(shared),
            "images": shared,
            "keys_compared": sorted(set(a[shared[0]])) if shared else [],
            "excluded_keys": sorted(EXCLUDE_TOP_KEYS),
            "n_images_with_any_difference": len(diffs),
            "differences": diffs,
        }

    summary = {
        "what_this_is": "determinism of the schematic2netlist pipeline, "
                        "measured over independent fresh-interpreter runs",
        "runs": args.runs,
        "split": args.split,
        "n_circuits": n,
        "n_errored_circuits": len(errored),
        "process_isolation": {
            "separate_processes": True,
            "pythonhashseed_per_run": [r["pythonhashseed"] for r in runs],
            "str_hash_probe_per_run": [r.get("str_hash_probe") for r in runs],
            "distinct_str_hash_probes": len(
                {r.get("str_hash_probe") for r in runs}),
            "pids": [r["pid"] for r in runs],
            "why": "string-hash seed is fixed at interpreter startup, so "
                   "set/dict ordering effects are invisible to repeats inside "
                   "one process; each run here is a fresh interpreter with a "
                   "different hash seed",
        },
        "config_hash_all_runs": sorted({r["config_hash"] for r in runs}),
        "exact_output_agreement": {
            "netlist_base_byte_identical_fraction": (agree["netlist_base"] / n
                                                     if n else None),
            "netlist_repaired_byte_identical_fraction": (
                agree["netlist_repaired"] / n if n else None),
            "n_circuits_netlist_changed": len(netlist_changed),
            "circuits_netlist_changed": netlist_changed[:50],
            "note": "real export_spice_netlist output, hashed byte for byte",
        },
        "topology_changes": {
            "n_circuits_topology_changed": len(topo_changed),
            "circuits": topo_changed[:50],
            "fraction_stable": (agree["topology"] / n) if n else None,
            "comparison": "naming-invariant partition of component terminals "
                          "into nets",
        },
        "label_changes": {
            "n_circuits_class_labels_changed": n - agree["labels"],
            "fraction_stable": (agree["labels"] / n) if n else None,
        },
        "full_result_agreement": {
            "fraction_all_runs_identical": (agree["full_result"] / n
                                            if n else None),
            "digest_covers": "every key of the run_pipeline dict except "
                             + ", ".join(sorted(EXCLUDE_TOP_KEYS)),
        },
        "full_result_structural_diff": full_diff,
        "metric_spread_across_runs": metric_spread,
        "invalid_outputs": {
            "definition": "pipeline raised, or the exported netlist contains "
                          "no element line at all (wrote_any false)",
            "n_circuits_invalid_in_any_run": len(invalid_any),
            "circuits": invalid_any[:50],
            "total_invalid_circuit_runs": int(sum(invalid_counts)),
            "total_circuit_runs": n * args.runs,
            "invalid_frequency": (sum(invalid_counts) / (n * args.runs)
                                  if n else None),
            "n_errored_circuits": len(errored),
            "spice_checked": bool(args.spice),
            "spice_note": "not run: byte-identical netlists cannot produce "
                          "different ngspice verdicts, so SPICE validity and "
                          "solvability inherit the byte comparison above",
        },
        "wall_s_per_run": [r["wall_s"] for r in runs],
        "wall_s_total": round(time.perf_counter() - t0, 2),
        "concurrency_caveat": "Other agents were running on this machine "
                              "throughout. This affects the WALL TIMES only; "
                              "determinism results are exact comparisons and "
                              "are unaffected by load.",
        "load_average_1_5_15": {"start": load_start, "end": load_end},
        "machine": platform.platform(),
        "python": sys.version.split()[0],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DETERMINISM ===")
    print(f"circuits: {n}   runs: {args.runs}   "
          f"hash seeds: {[r['pythonhashseed'] for r in runs]}   "
          f"distinct live str-hashes: "
          f"{summary['process_isolation']['distinct_str_hash_probes']}")
    e = summary["exact_output_agreement"]
    print(f"  netlist byte-identical across all runs: "
          f"{e['netlist_base_byte_identical_fraction']:.4f} (base) / "
          f"{e['netlist_repaired_byte_identical_fraction']:.4f} (repaired)")
    print(f"  circuits whose TOPOLOGY changed: "
          f"{summary['topology_changes']['n_circuits_topology_changed']}")
    print(f"  full run_pipeline dict identical: "
          f"{summary['full_result_agreement']['fraction_all_runs_identical']:.4f}")
    if full_diff["performed"]:
        print(f"  structural diff over {full_diff['n_images']} images: "
              f"{full_diff['n_images_with_any_difference']} differ")
    print(f"  invalid outputs: "
          f"{summary['invalid_outputs']['invalid_frequency']:.4f} "
          f"({summary['invalid_outputs']['total_invalid_circuit_runs']} of "
          f"{summary['invalid_outputs']['total_circuit_runs']} circuit-runs)")
    for k, v in metric_spread.items():
        print(f"  {k:28s} mean {v['mean']:.6f}  stdev {v['stdev']:.2e}  "
              f"range {v['range']:.2e}  identical={v['all_identical']}")
    print(f"\nwrote {out_dir}/summary.json + runs/run*.json")


if __name__ == "__main__":
    main()
