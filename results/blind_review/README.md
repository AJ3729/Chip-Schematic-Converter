# Independent second annotation — working directory

State: **packet built, annotator pending.** Nothing here is a real
inter-annotator result yet.

## What is here

| Path | What |
| --- | --- |
| `packet/` | what the second annotator receives: 58 raw photographs, a README, and `circuits.txt`. Nothing else — see below. |
| `manifest.csv` | the sample: stem, stratum, seed, source path and sha256. **Deliberately outside `packet/`** — the stratum label tells a reader which circuits are expected to be hard, which would bias exactly the circuits the strata exist to test. |
| `sampling_meta.json` | seed, pool sizes, the hard-core stratum's definition and its fallback, and the blind-safety assertion that was run. |
| `selftest/` | output of `compare_annotations.py --self-test`. Annotation B there is a **perturbed copy of A**, not anyone's annotation; the files carry a `_SYNTHETIC` banner saying so. Kept because it shows the exact output format a reviewer will get. |
| `comparison.json` | **absent on purpose.** It appears when a real second annotation is compared. There is no annotator yet, and a placeholder at that path would be mistaken for a result. |

## Sample composition (seed 20260804)

| Stratum | n | Drawn from |
| --- | --- | --- |
| `uniform` | 20 | all 192 test images — the only stratum that can carry an unbiased agreement estimate, so it is drawn first and from the whole split |
| `multi_terminal` | 20 | the 58 test images holding a 3+-terminal device, minus the above |
| `hard_core` | 18 | pipeline strict failures, minus the above — see the caveat |

**Hard-core caveat.** The intended definition was "the pipeline *and* both
frontier VLMs disagree with the ground truth". It could not be used: the VLM
per-image scores under `results/vlm/*_b/scored/` were computed before the
2026-08-03 role swap and share **zero** images with the current test split
(`data/README.md`). The stratum fell back to pipeline-only failures, which is a
weaker signal — one system failing is ordinary. The fallback is recorded in
`sampling_meta.json` under `pools.hard_core.definition`.

## Rebuilding / running

```sh
python scripts/make_blind_packet.py                       # rebuild the packet
python scripts/compare_annotations.py --self-test         # validate the differ
python scripts/compare_annotations.py --gt-b <B's dir> \
    --stems results/blind_review/manifest.csv             # the real comparison
```

The packet builder refuses to finish unless every image it copied is
byte-identical (sha256) to an untouched photograph under `data/raw`, is
hash-disjoint from all 1,216 render files in the repo, and the packet contains
no JSON. It exits non-zero rather than shipping a doubtful packet.
