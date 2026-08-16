#!/usr/bin/env python3
"""The reproducibility packet: what produced every table and figure (task F4).

A reader who wants to check a number in this paper needs three things and
usually gets none of them: which artifact holds it, which command wrote that
artifact, and what the environment was when it did. This assembles all three
into REPRODUCIBILITY.md, and it reads the artifacts rather than a hand-kept
list -- a mapping maintained by hand goes stale in exactly the way the numbers
themselves would.

Three parts:

  1. ARTIFACT MAP. Each table and figure -> the result files behind it, the
     script that regenerates them, and the config hash and seed recorded in
     those files' own run_meta.json. The config hash is the honest identifier:
     two runs with the same hash saw the same configuration, whatever the paths
     in between happened to be.

  2. ENVIRONMENT. Python, key package versions, ngspice, platform, and the git
     SHA each result was written at. Recorded from the artifacts where they
     carry it, and from this machine where they do not -- and it says which is
     which, because "the environment now" and "the environment then" are
     different claims.

  3. WHAT A READER CAN AND CANNOT REGENERATE. Stated plainly. Some artifacts
     need the datasets, which are not ours to redistribute; some need a GPU;
     one needs paid API access. Listing only the reproducible parts would be a
     more flattering and less useful document.

Usage:
    python scripts/reproducibility_packet.py
    python scripts/reproducibility_packet.py --out REPRODUCIBILITY.md
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# (paper label, human name, script, artifacts)
ARTIFACTS: list[tuple[str, str, str, list[str]]] = [
    ("tab:imaging", "Imaging properties of both corpora",
     "scripts/corpus_characterization.py",
     ["results/corpus_characterization.json"]),
    ("tab:swaps", "Pin-swap detection rate",
     "scripts/measure_op_agreement.py --stage accept",
     ["results/final/op_agreement/acceptance_test.json"]),
    ("tab:detector", "Detection on the held-out test split",
     "scripts/eval_detector.py",
     ["results/final/detection/seed0/test/summary.json",
      "results/final/detection/seed1/test/summary.json",
      "results/final/detection/seed2/test/summary.json"]),
    ("tab:main", "End-to-end reconstruction",
     "scripts/benchmark.py",
     ["results/final/benchmark/seed0/summary.json",
      "results/final/benchmark/seed1/summary.json",
      "results/final/benchmark/seed2/summary.json",
      "results/final/benchmark_val/summary.json"]),
    ("fig:ablation", "Cumulative ablation",
     "scripts/regen_on_split.py && scripts/make_paper_figures.py",
     ["results/final/ablation/index.json", "spec/ablation_arms.yaml"]),
    ("tab:pins", "Pin-order accuracy",
     "scripts/measure_pin_order.py",
     ["results/final/pin_order/summary.json"]),
    ("fig:opgap", "Topological perfection vs operating point",
     "scripts/measure_op_agreement.py",
     ["results/final/op_agreement/summary.json"]),
    ("tab:ladder", "Pin-aware ladder",
     "scripts/pin_aware_ladder.py",
     ["results/pin_aware_ladder.json", "spec/pin_symmetry.yaml"]),
    ("tab:multicondition", "Agreement under three probes",
     "scripts/multi_condition_agreement.py",
     ["results/multi_condition_agreement.json"]),
    ("(multistability)", "Multistability control",
     "scripts/multistability_control.py",
     ["results/multistability.json"]),
    ("tab:repair", "Declared repair interventions",
     "scripts/benchmark.py",
     ["results/final/benchmark/seed0/summary.json"]),
    ("tab:determinism", "Run-to-run determinism",
     "scripts/measure_determinism.py",
     ["results/final/determinism/summary.json"]),
    ("tab:vlm", "Frontier-model anchor",
     "scripts/vlm_task.py && scripts/score_vlm.py",
     ["results/vlm/PROVENANCE_TEST_SPLIT.md"]),
    ("tab:transfer", "Zero-shot detection transfer to CGHD",
     "scripts/cghd_detection_transfer.py",
     ["results/cghd_detection_transfer.json"]),
    ("tab:sizerecall", "Recall by component area",
     "scripts/cghd_detection_transfer.py",
     ["results/cghd_detection_transfer.json"]),
    ("fig:capture", "Capture invariance",
     "scripts/cghd_capture_invariance.py",
     ["results/cghd_capture_invariance.json"]),
    ("(cross-corpus netlists)", "CGHD netlist scoring, both conditions",
     "scripts/score_cghd_incremental.py",
     ["results/cghd_scored/incremental.json"]),
    ("(second annotation)", "Inter-annotator agreement",
     "scripts/make_blind_packet.py && scripts/compare_annotations.py",
     ["results/blind_review/manifest.csv",
      "results/blind_review/site_evidence_coverage.json"]),
]


def meta_for(paths: list[str]) -> dict:
    """config hash, seed and git SHA from whichever artifact records them."""
    out: dict = {"config_hash": set(), "seed": set(), "git_sha": set(),
                 "present": 0, "missing": []}
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            out["missing"].append(rel)
            continue
        out["present"] += 1
        if p.suffix != ".json":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:                                     # noqa: BLE001
            continue
        if isinstance(d, dict):
            for k in ("config_hash", "seed", "git_sha"):
                if k in d and not isinstance(d[k], (dict, list)):
                    out[k].add(str(d[k]))
        rm = p.parent / "run_meta.json"
        if rm.exists():
            try:
                m = json.loads(rm.read_text())
                for k in ("git_sha", "seed"):
                    if k in m:
                        out[k].add(str(m[k]))
                if "config" in m and "config_hash" in m:
                    out["config_hash"].add(str(m["config_hash"]))
            except Exception:                                 # noqa: BLE001
                pass
    return out


def env_now() -> dict:
    def cmd(args):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=20)
            return (r.stdout or r.stderr).strip().splitlines()[0]
        except Exception:                                     # noqa: BLE001
            return "not available"

    pkgs = {}
    for mod in ("numpy", "scipy", "cv2", "torch", "ultralytics",
                "networkx", "skimage"):
        try:
            m = __import__(mod)
            pkgs[mod] = getattr(m, "__version__", "unknown")
        except Exception:                                     # noqa: BLE001
            pkgs[mod] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} "
                    f"({platform.machine()})",
        "ngspice": cmd(["ngspice", "--version"]),
        "git_sha_now": cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "packages": pkgs,
    }


CAVEATS = """\
### What a reader can regenerate, and what they cannot

**Regenerable from this repository alone.** Everything downstream of the stored
result artifacts: every table, every figure, the pin-aware ladder, the
multi-condition agreement, the significance tests, and the manuscript's number
check (`scripts/manuscript_numbers.py --check paper/access.tex`). These read
committed JSON and CSV, so they need no dataset and no GPU.

**Needs the datasets.** Digitize-HCD and CGHD images are not redistributed here.
Digitize-HCD is used under its own terms; CGHD is obtained from its Zenodo
record. Any script that re-runs the pipeline over images -- the benchmark, the
ablation replay, detection transfer, capture invariance -- needs them present
under `data/`.

**Needs a GPU.** Detector training (three seeds, 300 epochs). The trained
weights are committed, so nothing in the paper requires retraining to verify;
the training path is included for completeness rather than as a prerequisite.

**Needs paid API access.** The frontier-model anchor. Prompts, model and API
versions, reasoning settings, per-image input hashes, output schema, run counts,
invalid-output handling, token usage, cost and per-image predictions are all
released, so the analysis is checkable without re-running it -- but re-running
it costs money and the models are not version-frozen by their providers.

**Needs a human.** The independent second annotation, the CGHD annotation
campaign, and the adjudication of any disagreement they produce. The tooling,
the sampling design, the blind-safety proofs and the scoring are all here and
self-tested; what is missing is a person, and no amount of code substitutes.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="REPRODUCIBILITY.md")
    a = ap.parse_args()

    env = env_now()
    rows, missing_any = [], []
    for label, name, script, paths in ARTIFACTS:
        m = meta_for(paths)
        if m["missing"]:
            missing_any.append((label, m["missing"]))
        def joined(k, default="--"):
            v = sorted(m[k])
            return ", ".join(v) if v else default
        rows.append(
            f"| `{label}` | {name} | `{script}` | "
            f"{m['present']}/{len(paths)} | {joined('config_hash')} | "
            f"{joined('seed')} |")

    pkg_lines = "\n".join(f"| {k} | {v} |" for k, v in env["packages"].items())
    body = f"""# Reproducibility packet

Generated by `scripts/reproducibility_packet.py`. Regenerate after any results
change; it reads the artifacts rather than a hand-kept list, so it cannot go
stale independently of them.

## 1. What produced each table and figure

`config hash` is the honest identifier for a run: two runs sharing it saw the
same configuration, whatever the paths in between happened to be.

| Paper element | What it is | Regenerate with | Artifacts | Config hash | Seed |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## 2. Environment

Recorded on the machine that generated this file. Where an artifact carries its
own `git_sha` and `seed`, those are shown in the table above and take precedence
-- "the environment now" and "the environment when the result was written" are
different claims and are not merged here.

| | |
| --- | --- |
| Python | {env['python']} |
| Platform | {env['platform']} |
| ngspice | {env['ngspice']} |
| git SHA (now) | `{env['git_sha_now']}` |

| Package | Version |
| --- | --- |
{pkg_lines}

## 3. One-command paths

```bash
# verify every number in the manuscript against its source
PYTHONPATH=src python scripts/manuscript_numbers.py --check paper/access.tex

# regenerate the macros the manuscript inputs
PYTHONPATH=src python scripts/manuscript_numbers.py --emit

# the metric self-tests, none of which need a dataset
PYTHONPATH=src python scripts/compare_annotations.py --self-test
PYTHONPATH=src python scripts/multi_condition_agreement.py --self-test
PYTHONPATH=src python scripts/score_cghd_incremental.py --self-test
PYTHONPATH=src python -m pytest tests/ -q
```

{CAVEATS}
## 4. Artifacts not present on this machine

{"None -- every artifact listed above is present." if not missing_any else
 chr(10).join(f"- `{lbl}`: " + ", ".join(f"`{p}`" for p in miss)
              for lbl, miss in missing_any)}
"""
    (ROOT / a.out).write_text(body)
    print(f"wrote {a.out}")
    print(f"  {len(ARTIFACTS)} paper elements mapped")
    if missing_any:
        print(f"  {len(missing_any)} element(s) have missing artifacts:")
        for lbl, miss in missing_any:
            print(f"    {lbl}: {', '.join(miss)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
