# What is in `results/`, and which artifact backs which claim

This directory is the evidence, not a working area: `summary.json`,
`per_image.csv` and `run_meta.json` are committed for every run because they
*are* the reported numbers. Bulky regenerable products (per-image ledgers,
exported netlists, review crops) are gitignored.

There are ~76 run directories here and only a dozen back the paper. This file
says which, so a reviewer does not have to guess.

## Read this first: which split a run used

The two evaluation splits **exchanged names on 2026-08-03** (`data/README.md` →
"the 2026-08-03 role swap"). Every parameter had been tuned on the 190-image
split, so anything reported there is in-sample; the untouched 192-image split
became `test`.

> **Every artifact dated before 2026-08-03 was computed on the 190 images and is
> a validation number — whatever its directory name or its `run_meta.json`
> `"split"` field says.** That includes `benchmark_1024_final/`, the oracles,
> the ablations, the sweeps and the VLM anchor.

Run `run_meta.json` records `timestamp_utc`, `split` and `gt_dir` for every run,
so the split a number came from is always checkable.

## The reported set

Everything the manuscript cites comes from these, all on the 192-image test
split. `scripts/regen_on_split.py --split test --fill-caches` rebuilds them all.

| directory | what it is | feeds |
| --- | --- | --- |
| `paper_test/seeds/seed{0,1,2}` | primary configuration, three detector seeds | Table V, and every headline macro |
| `paper_test/ablation/v1..v12` | the cumulative connectivity ablation, each arm replayed from its own frozen config | Table IV, Fig. 4 |
| `ablations_test192/wire_method.csv` | those twelve arms consolidated | Table IV, Fig. 4 |
| `detection_test192/test` | detector on test, per-class AP + 3-seed stats | Table III, Fig. 8 |
| `detection_test192/val` | the same weights on the split the detector never saw — the unbiased detection estimate | Table III, and the early-stopping caveat |
| `oracle_test192` | GT-substitution stage attribution, modes A–D | Table VII, Fig. 5 |
| `repair_test192` | solvability lift, topology-preservation proof, ground gauge | Table X |
| `stratified_test192` | performance by drawing characteristic, and the precision buckets | Table VIII, Fig. 3 |
| `split_swap/val_vs_test.json` | identical configuration on both splits, with the difficulty profile | Table VI |
| `gt_verification/stats.json` | ground-truth decision counts, derived from the released decision records | Table II |
| `ports` | port-template localization (scale-invariant, not split-specific) | Table XI |

## Supporting, cited but not headline

| directory | what it is |
| --- | --- |
| `vlm/` | external anchor: two frontier vision-language models on the same images as the pipeline. **On the validation split** — the three-way comparison is internally consistent because all three ran the same 190 circuits, and the paper says so. |
| `cghd_zero_shot/` | cross-dataset zero-shot detection on CGHD (100 images, 25 drafters) |
| `runtime_1024/`, `runtime/` | per-stage latency |
| `comparisons/` | paired per-image bootstrap comparisons between configurations |

## Everything else

The remaining directories are the experimental history — superseded resolutions,
abandoned mechanisms, diagnostics and oracles that produced negative results.
They are kept because a negative result nobody can check is not a result, and
several of the paper's design decisions rest on them. Notable groups:

| group | what it holds |
| --- | --- |
| `sweeps_bench/`, `sweeps/` | every parameter sweep. `sweeps_bench/README.md` maps each run to the split it actually read — including 15 that read images now in the test split, and why no shipped parameter derives from them. |
| `benchmark_1024*`, `benchmark_2048`, `v2..v5_*` | the configuration progression, at 512, 1024 and 2048 px. The 512 and 2048 sets are superseded and must never be mixed with 1024 numbers in one table. |
| `oracle*`, `oracles_phase0` | ceiling measurements taken *before* spending effort on a fix. Two of them redirected the project by showing a planned fix could not pay. |
| `weld_adjudication/`, `weld_*` | the over-merge investigation: where nets fuse, whether any single cut separates them (`manifest.csv` holds the adjudicated verdicts). |
| `crossing_transfer*`, `real_crossings/` | the wire-crossing classifier and its transfer behaviour. |
| `trace/`, `*_diag/` | per-stage debug visualizations. |

## Conventions

- `summary.json` — aggregate metrics with bootstrap 95% CIs.
- `per_image.csv` — one row per circuit; this is what the paired bootstrap and
  every stratification read.
- `run_meta.json` — the complete config snapshot, git SHA, seed, environment
  versions, split and GT directory. This is what makes a run replayable years
  later, and what `scripts/regen_on_split.py` reads to reconstruct an arm.
