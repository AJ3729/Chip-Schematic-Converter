# Frozen configuration — Tier 1

**The pipeline is frozen at this configuration.** Every Tier 1 result references
the hash below. Nothing downstream tunes, selects, thresholds, or early-stops
against any evaluation corpus, and CGHD in particular is a test corpus only.

Tag: `tier1-freeze`

---

## 1. Identity

| | |
| --- | --- |
| `configs/default.yaml` sha256 | `2f6c51081383d905f7881547e6423af81f052237d9f826f60caa1e0410f0ec22` |
| config hash recorded in run metadata | `54069dba36fa` |
| git SHA at freeze | `306d0f55fad3fb0e0ac459e9879fe1fa0211a19c` |
| global seed | `0` |

The two hashes measure different things and both are recorded on purpose. The
sha256 is of the file on disk. The `config_hash` field is computed over the
*parsed* configuration and appears in every `run_meta.json`, so a result can be
matched to its configuration even if the file was reformatted.

## 2. What the configuration points at

| key | value |
| --- | --- |
| `detect.weights` | `experiments/train_valstop/runs/yolov8s_640_seed0/weights/best.pt` |
| `detect.cache_dir` | `data/detections_valstop` |
| `preprocess.images_dir` | `data/cleaned_1024` (1024×1024 frames) |
| `benchmark.gt_dir` | `data/gt_test_1024` |

Detector seeds 1 and 2 live beside seed 0 under `experiments/train_valstop/`
and are used only for the three-seed variance columns; seed 0 is the shipped
configuration.

**The detector early-stops on `val`, not on the reported split.** This was not
always true and the defect is documented in Section VI of the manuscript. It is
settled and closed; no Tier 1 task reopens it.

## 3. Environment

```
Python      3.11.9        macOS-15.5-arm64-arm-64bit
numpy       2.3.5         scipy        1.16.3
opencv      4.10.0        networkx     3.6.1
torch       2.13.0        ultralytics  8.4.101
PyYAML      6.0.3
```

`ngspice` must be on PATH for every simulation metric. Its banner does not
parse a version string cleanly on this host, so the version is recorded by the
package manager rather than by `ngspice -v`; a reproducer should record their
own and note any difference, because solver defaults affect DC convergence.

Dependency pinning is via `pyproject.toml` plus the project virtualenv at
`./venv`. Every command in `REPRODUCE.md` uses `./venv/bin/python` explicitly
rather than an ambient interpreter.

## 4. Reproduction check (task A2)

The frozen pipeline was re-run over the 190 validation images and every metric
diffed against the published validation column.

| metric | published | reproduced | delta |
| --- | --- | --- | --- |
| terminal-pair F1 | 0.7996 | 0.7996 | +0.000010 |
| net F1 | 0.8766 | 0.8766 | −0.000026 |
| per-component connected | 0.6061 | 0.6061 | +0.000003 |
| nGED | 0.2155 | 0.2155 | −0.000016 |
| strict success | 0.4474 | 0.4474 | −0.000032 |
| SPICE valid | 1.0000 | 1.0000 | +0.000000 |
| DC solvable, pre-repair | 0.5105 | 0.5105 | +0.000026 |
| DC solvable, post-repair | 0.7579 | 0.7579 | −0.000005 |

n = 190 scored. **PASS** — every delta is below 5×10⁻⁵, which is the rounding
of the published four-decimal figures rather than any behavioural difference.

This is a genuine re-execution of the pipeline, not a re-read of a cached
summary: the run wrote a fresh `summary.json` to a scratch directory and the
comparison was made against the committed one.

## 5. What "frozen" forbids

From this tag until the manuscript is submitted:

* No threshold, kernel size, gate, or model weight changes.
* No parameter is selected using any evaluation corpus — not the Digitize-HCD
  test split, and not CGHD at any point for any purpose.
* CGHD is scored zero-shot. If a task appears to require adapting the pipeline
  to CGHD, that task is wrong and is reported rather than performed.
* A new computation that contradicts a published number is reported with both
  values and the reason. It never silently overwrites the published one.

Changes that are permitted because they cannot move a pipeline output:
measurement code, metric definitions applied to stored artifacts, analysis,
figures, and documentation.
