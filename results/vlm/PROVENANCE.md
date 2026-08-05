# VLM external anchor — experimental provenance

Full record of the two frontier-model runs held under `results/vlm/`, written
for a reviewer who needs to judge the comparison without re-running it.

Gaps are marked **MISSING** and left visible. A documented gap is worth more
than a clean-looking record, and several of the items below are gaps.

---

## 0. The one-line summary, and what it does not mean

Two frontier vision-language models were run over the same images, through the
same metric cascade, as the pipeline. Strict success:

| system | strict success | terminal-pair F1 | net F1 | nGED |
| --- | --- | --- | --- | --- |
| pipeline (`benchmark_1024_final/seed0`) | 0.4421 (84/190) | 0.7889 | 0.8714 | 0.2274 |
| `claude-opus-5` (variant B) | 0.4579 (87/190) | 0.8282 | 0.8868 | 0.1827 |
| `gpt-5.5-2026-04-23` (variant B) | 0.4789 (91/190) | 0.8562 | 0.9082 | 0.1700 |
| union of all three (oracle over systems) | **0.5737** (109/190) | — | — | — |

The reading this supports: the task is hard, and a small specialised pipeline
is competitive with frontier general models on it.

The readings it does **not** support, and which nothing in this directory
licenses:

* it is **not** evidence that the ground truth is correct. No system here was
  checked against a human re-read; agreement and disagreement with the
  annotation are both unaudited.
* it is **not** evidence that the dataset is uncompromised. It is a
  measurement of three systems, not of the data.
* the 81 images all three systems miss are **not** thereby proven to be
  information-limited. They are a queue for a human, and
  `analysis/hardcore_review_queue.csv` is that queue.

---

## 1. Which images — and the split-rename trap

**The VLM runs cover the 190 images that are TODAY called `val`.**

Verified by intersecting the 190 stems in `results/vlm/claude_b/rep0/*.json`
against the current manifests:

| manifest | images | overlap with the VLM runs |
| --- | --- | --- |
| `data/splits/val.txt` | 190 | **190 / 190** |
| `data/splits/test.txt` | 192 | **0 / 192** |

`results/vlm/openai_b/rep0` holds the identical stem set.

The runs were submitted on 2026-08-01. The **2026-08-03 role swap**
(`data/README.md`) then exchanged the two split names: the 190 images formerly
called `test` became `val`, and the 192 formerly called `val` became `test`.
No image moved and no annotation changed — only the labels. At submission time
`scripts/vlm_baseline.py` defaulted to `--split test`, which then selected
these 190 images; commit `f1481ebd` changed that default to `--split val`, so
the current default still selects the same set.

**Consequence: every VLM number in this directory, and every three-way
analysis under `analysis/`, is a VALIDATION-split result.** That includes the
0.4579 / 0.4789 strict-success figures above, the 81/63/12/25 group counts in
`analysis/twelve_summary.json`, and the 0.5737 union ceiling.

The 190-image split is the split every parameter in `configs/default.yaml` was
selected on, so the pipeline's 0.4421 in the table above is in-sample and a
reviewer is entitled to discount it. The VLMs did not see that split during
any selection, so their numbers are not in-sample in the same way — but they
are still not test-split numbers, and the *comparison* inherits the weaker of
the two positions.

**Re-running the anchor on the 192-image test split would cost money and has
not been approved. It is not done here.** Section 8 gives the exact commands
and a token-based estimate for whoever approves it.

The 192-image test-split GT was corrected on 2026-08-04 (`circuit_513`, one
net merged so a short-circuited voltage source is recorded as drawn).
`circuit_513` is in `test`, **not** in `val`, so that correction does not
touch anything in this directory.

### Blast radius of the mislabelling

`results/benchmark_1024_final/seed0/summary.json` still carries
`"split": "test"`. That field is stale — the run is on the 190 images now
called `val`. `data/README.md` already warns that every `results/` artifact
committed before 2026-08-03 is a validation number whatever its metadata says.

Checked: the VLM anchor is **not cited anywhere in `paper/`**. The
mislabelling has not reached the manuscript.

---

## 2. Prompts

`configs/vlm_prompts.json`, committed in `47f4173d` on **2026-08-01 15:39**,
before the first batch was submitted at ~17:44 the same day. The file's own
header states the reason: prompt design is a confound, and a reviewer cannot
judge the comparison without the exact text.

Contents: a shared `system` prompt, `variant_b_user` (the variant that was
run), and `variant_a_user` (built but **not run** — no `results/vlm/*_a/`
directory exists).

Both providers import the prompt from `scripts/vlm_task.py`, which is the only
module they share. Neither runner builds a prompt of its own, so the two
models received byte-identical text, image and output schema. This is
structural, not a claim: `vlm_task.build_task()` is called by both.

**Variant B** — the variant that was run — hands the model the frame with the
pipeline's own detected component boxes drawn and numbered, plus the component
list, and asks only for the net of each terminal. Component alignment is
therefore the identity map and cannot confound the comparison; the models are
handed detection and snapping for free and are tested on wire tracing alone.

---

## 3. Model versions

| | recorded id | dated snapshot? |
| --- | --- | --- |
| OpenAI | `gpt-5.5-2026-04-23`, in every one of the 190 per-image `_model` fields **and** in `openai_b/batches.json` | yes |
| Anthropic | `claude-opus-5`, in all 190 `_model` fields | **MISSING** |

**GAP.** The Anthropic side records only the undated alias `claude-opus-5`.
There is no dated snapshot anywhere in the artifacts, and
`scripts/vlm_baseline.py` hardcodes `MODEL = "claude-opus-5"` at module level
with **no CLI override**, so the alias cannot even be pinned without editing
the file. If the alias is repointed at a new snapshot, a re-run silently
measures a different model and nothing in this directory would reveal it.
`vlm_openai.py` does the opposite and refuses to guess (`--model` is
required), which is why the OpenAI side has a dated id.

`openai_b/batches.json` also records the batch id
`batch_6a6e6bd3a7588190a741ec619d900aca` and input file
`file-D6Texeqr2DBG3FPqrcdA6X`. `claude_b/batches.json` records two batch ids
for repeat 0 (see §6).

---

## 4. Reasoning settings

**MISSING — not recoverable from the artifacts.** No per-image record stores
effort, thinking mode, or the token cap. What can be stated:

*Defaults in the code as committed* (`vlm_baseline.py:params_for`, argparse):

| setting | Anthropic default | OpenAI default (`vlm_openai.py:body_for`) |
| --- | --- | --- |
| thinking / reasoning | `{"type": "disabled"}` (`--thinking` off) | `reasoning_effort` omitted unless `--effort` given |
| effort | `low` | `None` |
| token cap | `max_tokens = 16000` | `max_completion_tokens = 32000` |
| temperature | not set | not set, deliberately |
| structured output | `json_schema`, `SCHEMA_B` | `json_schema`, `strict: true`, `SCHEMA_B` |

*Evidence the defaults were probably not what ran, on the Anthropic side.*
`vlm_baseline.py` carries measured constants for this exact task:
`MEASURED_OUT[("disabled","low")] = 317` and
`MEASURED_OUT[("adaptive","low")] = 2368` output tokens per image. The observed
mean output in `claude_b/rep0` is **1708** tokens per image — 5.4x the
thinking-disabled figure and well short of the adaptive-low figure. Something
between the two was in force, or the constants are stale. **Either way the
setting that produced these 190 responses is not determinable from what is
stored.** On the OpenAI side the gap is narrower but real: the code's
`MEASURED_OUT = 1681` at `reasoning_effort=low` against an observed mean of
**1272**.

Neither provider was given a seed or a temperature, so the runs are not
deterministic even holding every recorded setting fixed.

**Fix for any future run:** persist the resolved request parameters (model,
effort, thinking, max_tokens) next to `batches.json` at submission time. One
JSON write closes this gap permanently.

---

## 5. Output schema

Strict JSON schema, defined once in `scripts/vlm_task.py` and sent to both
providers.

`SCHEMA_B` (the one that ran) — `{"components": [{"id": int, "terminals":
[string]}]}`, `additionalProperties: false`, both fields required. `SCHEMA_A`
additionally requires `class` and `bbox` and was never run.

Enforcement differs slightly by provider and this is not controlled for:
Anthropic receives it as `output_config.format.type = "json_schema"`, OpenAI as
`response_format.json_schema` with `strict: true`.

Post-hoc handling in `scripts/score_vlm.py:pred_from_response`: terminal lists
are padded with `None` or trimmed to the class's terminal count, so a
short answer is scored as a wrong answer rather than dropped. Detections whose
class carries zero terminals (`Wire Crossover`) are excluded from scoring but
are still drawn in the image, because the pipeline uses those boxes.

---

## 6. Number of runs

**One run per model. `rep0` only.**

Both runners default to `--repeat 3` and both docstrings say a single pass is
not a measurement, but only `rep0/` exists under `claude_b/` and `openai_b/`,
and `scored/summary.json` reports `n_repeats: 1` for both.

**Consequences, stated plainly:**

* cross-run determinism is **unmeasured**. With no temperature or seed set,
  the run-to-run spread of these models on this task is unknown.
* the `mean +/- SD` block that `score_vlm.py` prints reports **SD = 0.0000**
  for every metric. That zero is an artefact of `statistics.stdev` over one
  sample, not a finding, and must not be quoted as one.
* the strict-success gaps between the three systems (0.4421 / 0.4579 / 0.4789
  — 84, 87 and 91 images out of 190) are **within a range that a single extra
  repeat could plausibly move**, and no repeat exists to check.
* every group count in §7 rests on one sample per model.

### The truncation incident

**Not written down in any artifact as prose — recorded here for the first
time.** An earlier Anthropic attempt used `max_tokens = 4096`. Thinking tokens
count against that budget, so **19 of the 190 responses were clipped
mid-answer** (`stop_reason = max_tokens`, unparseable JSON). The cap was raised
to 16000 and the 19 were resubmitted.

The incident is *traceable* in the artifacts, if you know to look:

* `claude_b/batches.json` holds **two** batch ids for repeat 0
  (`msgbatch_01CpaMBFoqXkuFfXG4ocQGUr`, `msgbatch_01Ngvtakp5T89LsoLce7DN3G`).
  The second is the retry. `openai_b/batches.json` holds one, and needed one.
* file mtimes in `claude_b/rep0/` split **171 files at 17:44** and exactly
  **19 files at 17:51** — the retry batch, harvested seven minutes later.
* the incident is described in source comments at `vlm_baseline.py:80-83` and
  `:168-171`, and again in `vlm_openai.py:254-257`.

A second-order consequence that *is* a real gap: the 19 retried images were
answered by a **second, later API call** than the other 171. If the alias
`claude-opus-5` moved between 17:44 and 17:51 — nothing recorded would show it
— those 19 came from a different model than the rest. The 19 are identifiable
by mtime and are listed nowhere else.

Recovery behaviour, for the record: `is_done()` treats a cached error as
not-done so reruns retry it, and `harvest()` refuses to let an older batch's
failure overwrite a good result from a retry. Both were written in response to
this incident.

---

## 7. Invalid-output handling

* **Cached errors are not final.** `is_done()` returns false for any cached
  response containing an `error` key, so a rerun retries it.
* **Unusable responses are scored, not dropped.** `score_vlm.py` scores an
  empty prediction when `pred_from_response` returns `None`, keeping the
  denominator honest.
* **Final state: zero errors.** All 190 files in each `rep0/` parse, carry
  `_usage` and `_model`, and `scored/summary.json` reports `unusable: 0` for
  both models. The 190 images yield the same 13.6 mean components for both.

---

## 8. Token usage and cost

Aggregated from the per-image `_usage {input, output}` fields, 190 images per
model, one repeat.

| | input total | output total | grand total | input mean / median / p90 | output mean / median / p90 |
| --- | --- | --- | --- | --- | --- |
| `claude-opus-5` | 500,365 | 324,507 | 824,872 | 2634 / 2646 / 2949 | 1708 / 1578 / 3771 |
| `gpt-5.5-2026-04-23` | 369,943 | 241,597 | 611,540 | 1947 / 1955 / 2174 | 1272 / 1131 / 2276 |

Output ranges are wide (Anthropic 89–5553, OpenAI 91–2910), consistent with
reasoning tokens billed as output and scaling with circuit complexity.

### Cost — an ESTIMATE, derived from token counts

**These are estimates, not invoices.** The true billed amount must be read from
the provider consoles. The rates below are the values **committed in this
repository's own runner code**, which is the only rate provenance the artifacts
carry:

| rate | value | source | dated |
| --- | --- | --- | --- |
| Anthropic input / output | $5.00 / $25.00 per Mtok | `vlm_baseline.py:61` `PRICE_IN, PRICE_OUT` | commit `47f4173d`, 2026-08-01 |
| OpenAI input / output | $10.00 / $30.00 per Mtok | `vlm_openai.py:173-177` argparse defaults | commit `0f29a90e`, 2026-08-01 |
| batch discount | 50% | `BATCH_DISCOUNT = 0.5`; both runs used the Batches API | — |

The OpenAI rate is self-described in the source as *"a deliberately pessimistic
flagship rate — set your real one"*, so the OpenAI figure below is an **upper
bound**, not a best estimate. **MISSING:** no invoice, usage export or console
screenshot is stored for either provider, so no figure here is confirmed
against what was actually charged.

| model | estimated batch cost | (if it had been run sync, 2x) |
| --- | --- | --- |
| `claude-opus-5` | **~$5.31** | ~$10.61 |
| `gpt-5.5-2026-04-23` | **~$5.47** (upper bound) | ~$10.95 |
| **both, as run** | **~$10.78** | ~$21.56 |

Not included: the 19 truncated responses from the abandoned `max_tokens=4096`
attempt were billed for their output tokens and are not in any `_usage` field,
because the truncated files were overwritten by the retry. The true Anthropic
spend is therefore **higher than $5.31 by an unrecorded amount**.

---

## 9. Per-image predictions

* `results/vlm/claude_b/rep0/*.json` — 190 files
* `results/vlm/openai_b/rep0/*.json` — 190 files

Shape: `{"components": [{"id": <detection index>, "terminals": ["n1", ...]}],
"_usage": {"input": int, "output": int}, "_model": str}`. Both providers write
this identical shape, so `score_vlm.py` has no per-provider branch and neither
model gets a scoring advantage. Results are keyed by `custom_id`, never by
position, because both Batch APIs return results in arbitrary order.

---

## 10. Scoring

`scripts/score_vlm.py`. It calls `benchmark.score_prediction` and
`benchmark.aggregate` — the same functions `scripts/benchmark.py` uses — at the
same IoU threshold (0.3), against the same GT. No metric is reimplemented.

Metrics: `terminal_pair_f1`, `net_f1`, `per_component_connected_acc`, `nged`,
`strict_success`. `strict_success` requires `unmatched_gt == 0` **and**
terminal-pair F1 == 1.0 **and** net F1 == 1.0.

**Why the pipeline-vs-VLM comparison is internally consistent.** All three
systems ran the same 190 images, at the same resolution
(`data/cleaned_1024`), from the same detection cache
(`data/detections_1024`), with the same class head applied
(`vlm_task.load_detections` calls `class_head_reclassify`, matching what
`run_pipeline` does internally — without it, 6 of 40 self-test images scored
differently from the benchmark on identical predictions). They are scored by
the same functions against the same GT at the same threshold. In variant B the
models return the pipeline's own detection ids, so component matching is the
identity map. Independently verified while building §11: re-running the
pipeline and re-scoring both models reproduces the committed per-image
`terminal_pair_f1` and `net_f1` **exactly**, for all three systems.

A consequence visible in the queue: all three systems have `unmatched_gt > 0`
on the **same 27** of the 81 hard-core circuits. Detection misses are shared
infrastructure, not a per-system failure.

### GAP — the scorer no longer reproduces itself

`score_vlm.py` reads GT from `cfg["benchmark"]["gt_dir"]`, which since the role
swap is `data/gt_test_1024` — the **192-image test split**. The VLM outputs are
val images, so every stem misses its GT file and is dropped by
`if not gp.exists(): continue`.

Verified by running it (no API calls involved). Today it reports:

```
rep0: n=0 unusable=0  terminal_pair_f1=0.0000  net_f1=0.0000  strict_success=0.0000
```

It does **not** crash. It writes a complete, plausible-looking, entirely zero
summary. Anyone re-running the documented command and trusting the output would
silently get nothing. `score_vlm.py` has no `--gt-dir` flag and `load_config`
does no key merging, so the only workaround today is a full alternative config
file with `benchmark.gt_dir: data/gt_val_1024`. **Adding `--gt-dir`, and making
a zero-scored run a hard error, is the fix.**

`scripts/vlm_hardcore.py` (§11) takes `--gt-dir` explicitly, defaults it to
`data/gt_val_1024`, treats a missing GT file as a fatal error, and asserts that
the scored stems are exactly the `--split` manifest.

### GAP — the inputs are not versioned

`data/*` is gitignored except `splits/`, `README.md` and `gt_test_1024/`. So
the images the models saw (`data/cleaned_1024`), the detections drawn on them
(`data/detections_1024`), the class-head weights, and the **val GT the anchor
is scored against** (`data/gt_val_1024`, 0 files tracked) are all outside git.
The anchor cannot be re-derived from the repository alone; it needs the data
artifact transferred alongside.

---

## 11. Three-way analysis, recomputed on the current split

`analysis/twelve_summary.json` was computed before the rename and is
unlabelled. Recomputed from the committed per-image CSVs against the **current
`data/splits/val.txt`**, it reproduces **exactly**:

| group | n | meaning |
| --- | --- | --- |
| all three succeed | 63 | |
| pipeline fails, **both** VLMs succeed | 12 | the "recoverable 12" |
| pipeline fails, **at least one** VLM succeeds | 25 | |
| **all three fail — the hard core** | **81** | the review queue |
| union (any system succeeds) | 109 | ceiling **0.5737** |

So the earlier numbers were *arithmetically* right and *provenance*-wrong: they
are validation-split figures. All outputs written by `scripts/vlm_hardcore.py`
now carry `split`, `gt_dir` and the manifest path in every row and in
`analysis/hardcore_review_queue.meta.json`.

### The hard-core review queue

`scripts/vlm_hardcore.py` → `analysis/hardcore_review_queue.csv`, 81 rows, 45
columns. Per circuit: GT size, all three systems' `terminal_pair_f1`,
`net_f1`, `nged`, `unmatched_gt`, missing/extra terminal-pair counts, net-count
delta, and a split/weld failure mode; the pairwise inter-system F1s; and the
disputed terminal pairs rendered as `c3.t0~c7.t1`.

Ranked by `gt_outlier_margin` = (mean pairwise terminal-pair F1 **among the
three predictions**) − (mean terminal-pair F1 of each prediction **against
GT**). High margin = the three systems agree with each other more than any of
them agrees with the annotation, which is the configuration in which the
annotation is worth re-reading first.

**This ranks a hypothesis; it does not settle it.** A three-way disagreement is
equally consistent with an ambiguous drawing, with a bias the three systems
share (the same bare-crossing convention, the same tolerance for a faint
conductor — their agreement is then not independent evidence at all), or with
three independent failures. Nothing in this queue validates the ground truth,
and it must not be cited as an audit that found the annotations sound.

Headline shape of the queue:

* **42 of 81** circuits have a positive margin.
* **11 of 81** have inter-system F1 of **exactly 1.000** — all three systems
  produce the *identical* terminal-pair set, and all three contradict the GT.
  These are the highest-value human re-reads: `circuit_575`, `circuit_1121`,
  `circuit_1018`, `circuit_178`, `circuit_1239`, `circuit_153`, `circuit_611`,
  `circuit_833`, `circuit_312`, `circuit_427`, `circuit_645`.
* **11 of 81** have `consensus_delta == 0` — no disagreement shared by all
  three. Those look like three independent failures, and rank low.
* median consensus delta 10 pairs.
* failure modes differ by system on these 81: the pipeline is weld-dominant on
  44 and split-dominant on 31; OpenAI is weld-dominant on 58 of 81.

---

## 12. REQUIRING OWNER APPROVAL — the 3-repeat configuration

**Nothing below has been run. No API call was made in producing this
document.** Every command here spends money.

### Why a fresh output directory, not `--repeat 3` in place

Both runners skip images already cached, so pointing `--repeat 3` at the
existing directories would keep `rep0` and add reps 1–2. That would be wrong:
**the settings that produced `rep0` are unknown** (§4), so reps 1–2 would be
run under today's defaults and mixed with a rep0 of a different configuration.
The resulting spread would measure a settings change, not run-to-run variance.
Use a new `--out-dir` so all three repeats share one configuration.

### Commands — validation split (comparable to everything above)

```bash
# Anthropic — 3 repeats, fresh directory.
# NOTE: MODEL is hardcoded in the script; to pin a dated snapshot you must
# edit vlm_baseline.py:60. Do that before running, or §3's gap recurs.
./venv/bin/python scripts/vlm_baseline.py \
    --variant b --split val --repeat 3 \
    --effort low --max-spend 20 \
    --out-dir results/vlm/claude_b_r3

# OpenAI — 3 repeats, model pinned explicitly, real rates substituted.
./venv/bin/python scripts/vlm_openai.py \
    --variant b --model gpt-5.5-2026-04-23 --split val --repeat 3 \
    --effort low --price-in <REAL> --price-out <REAL> --max-spend 20 \
    --out-dir results/vlm/openai_b_r3

# Scoring. score_vlm.py has no --gt-dir and the config's benchmark.gt_dir is
# the TEST split, so a plain invocation scores n=0 SILENTLY (§10). Pass a full
# config whose benchmark.gt_dir is data/gt_val_1024:
./venv/bin/python scripts/score_vlm.py --run-dir results/vlm/claude_b_r3 \
    --variant b --config configs/default_valgt.yaml
./venv/bin/python scripts/score_vlm.py --run-dir results/vlm/openai_b_r3 \
    --variant b --config configs/default_valgt.yaml

./venv/bin/python scripts/vlm_hardcore.py --split val \
    --claude-dir results/vlm/claude_b_r3 --openai-dir results/vlm/openai_b_r3
```

### Commands — test split (a reportable number; separately approvable)

192 unseen images, none of which any model has been shown. `benchmark.gt_dir`
is already correct for these, so no config override is needed. Compare against
`results/paper_test/seeds/seed0/per_image.csv`.

```bash
./venv/bin/python scripts/vlm_baseline.py --variant b --split test --repeat 3 \
    --effort low --max-spend 20 --out-dir results/vlm/claude_b_test
./venv/bin/python scripts/vlm_openai.py --variant b --model gpt-5.5-2026-04-23 \
    --split test --repeat 3 --effort low --price-in <REAL> --price-out <REAL> \
    --max-spend 20 --out-dir results/vlm/openai_b_test
./venv/bin/python scripts/score_vlm.py --run-dir results/vlm/claude_b_test --variant b
./venv/bin/python scripts/score_vlm.py --run-dir results/vlm/openai_b_test --variant b
./venv/bin/python scripts/vlm_hardcore.py --split test \
    --gt-dir data/gt_test_1024 \
    --pipeline-csv results/paper_test/seeds/seed0/per_image.csv \
    --claude-dir results/vlm/claude_b_test --openai-dir results/vlm/openai_b_test
```

Always confirm with `--dry-run` first: it builds one request, writes
`dryrun_prompt.txt` and `dryrun_image.png`, and makes **no API call**.

### Cost estimate — token-based, batched, at the §8 rates

Scaled from the measured per-image usage of `rep0` (2634 in / 1708 out for
Anthropic; 1947 in / 1272 out for OpenAI).

| scenario | images x reps x models | estimated batch cost |
| --- | --- | --- |
| already spent (rep0, both models) | 190 x 1 x 2 | ~$10.78 |
| **3 repeats, val, fresh dir, both models** | 190 x 3 x 2 | **~$32.34** |
| 3 repeats, test, both models | 192 x 3 x 2 | **~$32.68** |
| both of the above | | **~$65.02** |
| 2 extra reps appended to the existing dirs (**not recommended**, §12 opener) | 190 x 2 x 2 | ~$21.56 |

Caveats on these figures, all of which push the true cost **up**:

* the OpenAI rate is the source's self-described pessimistic placeholder, so
  that half is an upper bound — but the Anthropic half is not.
* per-image output tokens scale with reasoning depth. Raising `--effort` above
  `low` moves this by up to **67x** on the Anthropic side, by the code's own
  measured constants. Do not change `--effort` without re-pricing.
* neither runner charges for retries separately; a repeat of the §6 truncation
  incident adds unbilled-to-us-but-billed-by-them output tokens.
* both scripts refuse to submit above `--max-spend` and print the projection
  before submitting. Read the projection.

---

## 13. Gap register

| # | gap | severity |
| --- | --- | --- |
| 1 | Anthropic model recorded as the undated alias `claude-opus-5`; no CLI override to pin a snapshot | high |
| 2 | Reasoning settings (effort / thinking / token cap) not stored; observed token counts contradict the code defaults | high |
| 3 | One repeat per model, not the three planned; cross-run determinism unmeasured and reported SD is a one-sample artefact | high |
| 4 | `score_vlm.py` silently scores n=0 today and writes an all-zero summary — the anchor does not reproduce from the documented command | high |
| 5 | The 19 retried images came from a later API call than the other 171; if the alias moved in between, nothing would show it | medium |
| 6 | Images, detections, class-head weights and val GT are all outside git; the anchor is not re-derivable from the repository alone | medium |
| 7 | No invoice or usage export; all costs are token-derived estimates and the Anthropic figure omits the abandoned attempt | medium |
| 8 | No temperature or seed set for either provider | medium |
| 9 | Variant A was built and never run; only variant B has results | low |
| 10 | Schema enforcement mechanism differs slightly between providers and is not controlled for | low |

---

*Written 2026-08-04 from committed artifacts only. No API call was made. The
recomputation in §11 reproduces every committed per-image metric exactly.*
