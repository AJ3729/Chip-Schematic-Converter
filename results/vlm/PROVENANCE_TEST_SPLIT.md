# VLM anchor — full experimental provenance (TEST split, variants A and B)

Complete record of the four frontier-model runs held under `results/vlm/*_test/`,
written so a reviewer can judge or repeat the comparison without re-running it.

**This supersedes nothing.** `PROVENANCE.md` documents the earlier runs on the
190 images now called `val`; those remain valid as validation-split results and
their section 4 records reasoning settings as unrecoverable. Everything below is
on the **192-image held-out test split**, and every field that was MISSING for
the earlier run is recorded here because both runners now persist the resolved
request parameters *before* any request leaves the machine.

Written 2026-08-06. Repository at git `943764f2`.

---

## 0. The four runs at a glance

| run | variant | model | repeats | requests | directory |
| --- | --- | --- | --- | --- | --- |
| A/Claude | A | `claude-opus-5` | 1 | 192 | `results/vlm/claude_a_test/` |
| B/Claude | B | `claude-opus-5` | **3** | 576 | `results/vlm/claude_b_test/` |
| A/GPT | A | `gpt-5.5-2026-04-23` | 1 | 192 | `results/vlm/openai_a_test/` |
| B/GPT | B | `gpt-5.5-2026-04-23` | 1 | 192 | `results/vlm/openai_b_test/` |

**What the two variants are, and why both exist.**

* **Variant A — unaided.** The model receives the raw photograph and the class
  list, and must find every component, classify it, produce a bounding box, and
  assign nets. This is the end-to-end task the pipeline performs.
* **Variant B — assisted.** The model receives the frame with **our detected
  component boxes drawn on it**, plus a list of component ids, classes and
  terminal names. It returns only the net of each terminal. Detection and
  classification are handed to it for free.

Variant B therefore measures the **connectivity stage in isolation**; variant A
measures the whole task. Reporting B alone would overstate what a general model
does with a schematic, and reporting A alone would hide which stage it fails at.

---

## 1. Exact prompts

`configs/vlm_prompts.json`, `schema_version: 1`,
sha256 `e1d49fa5c968e6b4dafeb6c37e273fc40e90eace119735ef41f7bc5395feaa8f`.

Both providers import the prompt from `scripts/vlm_task.py`, the only module
they share. Neither runner builds a prompt of its own, so **the two models
received byte-identical text, image and output schema**. Prompt design is a
confound and a reviewer cannot judge the comparison without the exact text, so
it is reproduced in full.

### 1.1 System prompt (shared by both variants, both models; 456 chars)

```
You are an expert at reading hand-drawn circuit schematics. You trace conductors
carefully and you know the standard notations: a junction dot or a plain
T-intersection means two wires CONNECT; a semicircular hop (one wire bulging
over another) means they do NOT connect. Different people draw differently —
some mark every non-connection with a hop, others draw bare crossings and rely
on the reader's circuit knowledge. Read what is actually on the page.
```

### 1.2 Variant A user prompt (951 chars, before substitution)

```
This is a hand-drawn circuit schematic. Identify every electrical component and
determine the complete circuit connectivity.

For each component report:
- its class, from exactly this list: {class_list}
- its bounding box in image pixel coordinates as [center_x, center_y, width,
  height]. The image is {width} by {height} pixels, origin at the TOP-LEFT.
- the net of each of its terminals, in the terminal order given below

Terminal counts by class:
{terminal_counts}

An electrical NET is a maximal set of terminals joined by conductor. A crossing
WITHOUT a junction — especially one drawn as a hop — does NOT join nets.

Rules:
- Net names are arbitrary labels ('n1', 'n2', ...). Only the GROUPING matters.
- The net a GND symbol touches MUST be named exactly "0".
- Do NOT report wire crossings, junction dots, or text labels as components —
  only the electrical components in the class list.

Trace each conductor from end to end before answering.
```

`{class_list}` and `{terminal_counts}` are filled from the canonical class table
and are identical for every image. `{width}`/`{height}` are 1024/1024.

### 1.3 Variant B user prompt (1129 chars, before substitution)

```
This is a hand-drawn circuit schematic. Every component has already been located
for you and is outlined in the image with its ID number. Your ONLY job is to
determine the electrical connectivity.

An electrical NET is a maximal set of terminals joined by conductor. Two
terminals joined by any unbroken conductive path share one net. A crossing
WITHOUT a junction — especially one drawn as a hop — does NOT join nets.

Components in this image:
{component_list}

For each component, give the net of each of its terminals, in the listed
terminal order.

Rules:
- Net names are arbitrary labels ('n1', 'n2', ...). Only the GROUPING matters,
  not the names.
- The net that a GND symbol touches MUST be named exactly "0". If there is no
  GND symbol, no net is named "0".
- Two terminals get the same net name if and only if a conductor joins them.
- Return exactly the terminal count listed for each component, in that order.
- Every component in the list must appear in your answer, even if you are unsure.

Trace each conductor from end to end before answering. Where wires cross, decide
deliberately whether the drawing joins them.
```

`{component_list}` is per-image and is generated from **our detections**, one
line per component, e.g.

```
  id 0: Resistor — 2 terminal(s), in order: terminal 0, terminal 1
  id 4: BJT-NPN — 3 terminal(s), in order: Collector, Base, Emitter
```

The fully rendered first-image prompt for every run is stored verbatim at
`results/vlm/<run>/request_provenance.json` → `prompts.user_template_rendered_for_first_image`.

---

## 2. Model and API versions

| | A/Claude | B/Claude | A/GPT | B/GPT |
| --- | --- | --- | --- | --- |
| model requested | `claude-opus-5` | `claude-opus-5` | `gpt-5.5-2026-04-23` | `gpt-5.5-2026-04-23` |
| model id **returned by the API** | `claude-opus-5` | `claude-opus-5` | `gpt-5.5-2026-04-23` | `gpt-5.5-2026-04-23` |
| API surface | Messages + Batches | Messages + Batches | Chat Completions + Batches | Chat Completions + Batches |
| SDK | `anthropic` 0.120.2 | 0.120.2 | `openai` 2.52.0 | 2.52.0 |
| submitted (UTC) | 2026-08-06 02:59:33 | 2026-08-06 02:05:31 | 2026-08-06 03:00:03 | 2026-08-06 02:20:31 |

Python 3.11.9, macOS-15.5-arm64. Repository git SHA `943764f2` for all four.

The returned id is recorded **per image** in the `_model` field of every cached
response, so a silent model substitution by either provider would be visible in
the artifacts rather than assumed away.

Chat Completions rather than the Responses API on the OpenAI side because the
Batch API supports it most broadly; `max_completion_tokens` rather than
`max_tokens` because current reasoning models reject the latter.

---

## 3. Reasoning settings

**Recorded at submission time, not reconstructed.** This is the field the
earlier val-split run could not supply.

| | Anthropic (both variants) | OpenAI (both variants) |
| --- | --- | --- |
| thinking / reasoning | `{"type": "adaptive"}` (`--thinking` on) | `reasoning_effort` **omitted** (provider default) |
| effort | `low` | — |
| token cap | `max_tokens` = 16000 | `max_completion_tokens` = 32000 |
| temperature | not set (provider default) | not set — several reasoning models reject it |
| seed | not supported / not set | not set |
| structured output | `json_schema` | `json_schema`, `strict: true` |

**Neither provider was given a seed or a temperature, so the runs are not
deterministic even holding every recorded setting fixed.** Section 10 measures
exactly how non-deterministic.

Anthropic effort was set to `low` deliberately: thinking tokens are invisible but
billed as output and dominate cost, and at `high` this task costs roughly 67× as
much per image. `low` with adaptive thinking is the setting whose measured output
(2,368 tok/image) brackets the earlier val run's observed 1,708 from above,
making it the closest defensible reconstruction of that run.

---

## 4. Input images

| | variant A | variant B |
| --- | --- | --- |
| images | 192 | 192 |
| media type | `image/jpeg` | `image/png` |
| mean bytes | 77,591 | 181,428 |
| total payload | 14.9 MB | 34.8 MB |

Source frames: `data/cleaned_1024`, **1024 × 1024 px**, the deskewed,
shadow-normalised frames the pipeline itself reads.

* **Variant A** sends the frame's own JPEG bytes unmodified. Re-encoding to PNG
  would inflate the payload ~4× for pixels that are already lossy.
* **Variant B** must draw on the frame, so it re-encodes to PNG — box outlines
  and id glyphs are exactly the kind of hard edge JPEG smears.

### 4.1 Byte-level provenance, and why it matters

`results/vlm/<run>/request_provenance.json` → `inputs.images` (OpenAI) and
`results/vlm/<run>/request_manifest_rep<N>.json` (Anthropic) record, **for every
one of the 192 images**: sha256 of the exact bytes transmitted, media type, byte
length, number of detections, number of components, and sha256 of the rendered
user text.

This is not bookkeeping for its own sake. **Variant B renders our detector's
output into the image**, so those hashes pin which detector state the anchor was
measured against — here `data/detections_valstop`, produced by the retrained
detector that early-stops on `val` rather than on the reported test split.

Example, `circuit_1013`:

```
variant A  image_sha256 23d2afeeb1018a7f790d93e68d7dcfc7...  image/jpeg
variant B  image_sha256 c569b1c18d7d749af7866d59bdb28a7d...  image/png
```

Detections directory: `data/detections_valstop`.
Images directory: `data/cleaned_1024`.

---

## 5. Output schema

Strict JSON schema, defined once in `scripts/vlm_task.py` and sent to both
providers. `additionalProperties: false` throughout; all fields required.

**SCHEMA_A** (variant A):

```json
{"type":"object","properties":{"components":{"type":"array","items":{
  "type":"object","properties":{
    "class":{"type":"string"},
    "bbox":{"type":"array","items":{"type":"number"}},
    "terminals":{"type":"array","items":{"type":"string"}}},
  "required":["class","bbox","terminals"],"additionalProperties":false}}},
 "required":["components"],"additionalProperties":false}
```

**SCHEMA_B** (variant B):

```json
{"type":"object","properties":{"components":{"type":"array","items":{
  "type":"object","properties":{
    "id":{"type":"integer"},
    "terminals":{"type":"array","items":{"type":"string"}}},
  "required":["id","terminals"],"additionalProperties":false}}},
 "required":["components"],"additionalProperties":false}
```

Variant A's `bbox` is `[center_x, center_y, width, height]` in pixels, origin
top-left, as stated in the prompt.

---

## 6. Number of runs

| run | repeats | why |
| --- | --- | --- |
| B/Claude | **3** | a single pass cannot measure run-to-run stability; three is the minimum that yields a variance and a topology-change count |
| A/Claude, A/GPT, B/GPT | 1 | anchor point estimates only |

**Consequence, stated plainly: determinism is measured on Claude/variant B
only.** GPT's numbers and both variant A numbers are single samples of a
stochastic process and carry no run-to-run error bar. Any claim about GPT's
reproducibility would be unsupported by these artifacts.

Batch mode throughout (50% discount). One batch per repeat, batch ids
checkpointed in `batches.json` so an interrupted poll resumes, results keyed by
`custom_id` because both Batch APIs return results in arbitrary order.

---

## 7. Invalid-output handling

### 7.1 How an unusable response is recorded

| condition | recorded as |
| --- | --- |
| model refusal | `{"error": "refusal", "category": ...}` |
| no text block | `{"error": "no_text", "stop_reason": ...}` |
| unparseable JSON | `{"error": "bad_json", "message": ...}` |
| batch-level failure | `{"error": <result type>, "message": ...}` |

A cached **error does not count as done** (`is_done()`), so a rerun retries it
rather than baking in a transient failure. An older batch's failure never
clobbers a good result already harvested from a retry.

### 7.2 How the scorer treats one

`scripts/score_vlm.py` counts an unusable response as a **failure with an empty
prediction** — never as a missing sample. The denominator stays at 192 for every
run, so a model cannot improve its score by failing to answer.

### 7.3 What actually happened

| run | requests | usable | errors |
| --- | --- | --- | --- |
| A/Claude | 192 | 190 | 2 (batch-level) |
| B/Claude | 576 | **576** | none |
| A/GPT | 192 | 184 | **8 × `http_429`** |
| B/GPT | 192 | 192 | none |

The 8 GPT variant-A failures are `http_429` from an exhausted credit balance,
not model refusals. They are scored as failures. **Even if all 8 had succeeded
perfectly, A/GPT's ceiling would be 0.167 strict success against the pipeline's
0.531, so the conclusion is unchanged** — but the handling is stated rather than
hidden.

A separate earlier B/GPT submission lost 137 of 192 requests to the same credit
exhaustion; those were retried in a second batch and the 55 already-successful
responses were reused. The final B/GPT set is complete at 192.

---

## 8. Token usage

Aggregated from the per-image `_usage` fields.

| run | requests | input mean | output mean | output p90 | input total | output total |
| --- | --- | --- | --- | --- | --- | --- |
| A/Claude | 190 | 2,622 | 1,558 | 3,519 | 498,180 | 296,020 |
| B/Claude | 576 | 2,626 | 1,708 | 4,029 | 1,512,690 | 983,651 |
| A/GPT | 184 | 1,906 | 3,604 | 7,284 | 350,704 | 663,136 |
| B/GPT | 192 | 1,942 | 3,872 | 8,998 | 372,864 | 743,424 |

Two observations worth carrying into the discussion:

* **GPT spends roughly 2.2× the output tokens Claude does on the same task**
  (3,872 vs 1,708 on variant B), consistent with reasoning tokens billed as
  output. It is also the stronger performer on variant B. The extra computation
  is visible in the bill.
* Output ranges are wide (p90 is 2–2.3× the mean for every run), scaling with
  circuit complexity.

---

## 9. Cost

**Estimates derived from token counts, not invoices.** Rates are the values
committed in this repository's runner code, which is the only rate provenance
the artifacts carry. No invoice, usage export or console screenshot is stored,
so no figure here is confirmed against what was actually charged.

| rate | value | source |
| --- | --- | --- |
| Anthropic input / output | $5.00 / $25.00 per Mtok | `vlm_baseline.py` `PRICE_IN, PRICE_OUT` |
| OpenAI input / output | $10.00 / $30.00 per Mtok | `vlm_openai.py` argparse defaults |
| batch discount | 50% | both runs used the Batches API |

The OpenAI rate is self-described in the source as *"a deliberately pessimistic
flagship rate"*, so OpenAI figures are an **upper bound**.

| run | estimated cost |
| --- | --- |
| A/Claude | $4.94 |
| B/Claude (3 repeats) | $16.08 |
| A/GPT | $11.70 |
| B/GPT | $13.02 |
| **total** | **$45.74** |

Per-image marginal cost, variant B: **$0.028** (Claude), **$0.068** (GPT).
The pipeline's per-image marginal cost is **$0.00**.

**A caution for anyone re-running this.** Both runners project cost before
submitting and abort above `--max-spend`, but the projection uses constants
measured on variant B at particular settings. Observed GPT output was 2.3× its
projection constant. The gate protects against catastrophic overspend, not
against a 2× surprise — set the ceiling from observed tokens, not from the
projection.

---

## 10. Determinism (B/Claude, 3 repeats)

The only run with repeats. 192 images, identical settings, three independent
batches.

| measure | value |
| --- | --- |
| exact-output agreement (byte-identical JSON, all 3 runs) | 45/192 = **0.2344** |
| topology agreement (naming-invariant partition, all 3) | 78/192 = **0.4062** |
| **circuits whose predicted topology CHANGED** | **114/192 = 0.5938** |
| invalid / unusable outputs | 0/576 |

Pairwise topology agreement: rep0–rep1 0.5052, rep0–rep2 0.4740, rep1–rep2
0.5156. Not one rogue run — every pair disagrees on about half the circuits.

Metric variance across the three repeats: strict success **±0.0167**,
terminal-pair F1 ±0.0038, net F1 ±0.0041, nGED ±0.0042.

**Topology agreement is computed naming-invariantly** — the partition of
`(component_id, terminal_index)` pairs into nets — so relabelling `n1`→`n2` does
not count as a change. The 114 are real connectivity differences.

For contrast, the pipeline over 5 fresh interpreters with distinct
`PYTHONHASHSEED` values (`results/determinism/summary.json`): **192/192
byte-identical, 0 topology changes, 0.0 variance on all four headline metrics.**

---

## 11. Per-image predictions

Every response is cached, one JSON file per image per repeat:

```
results/vlm/claude_a_test/rep0/<stem>.json
results/vlm/claude_b_test/rep{0,1,2}/<stem>.json
results/vlm/openai_a_test/rep0/<stem>.json
results/vlm/openai_b_test/rep0/<stem>.json
```

Example — `results/vlm/claude_b_test/rep0/circuit_1013.json`:

```json
{"components": [{"id": 0, "terminals": ["n1", "0"]},
                {"id": 1, "terminals": ["0"]},
                {"id": 2, "terminals": ["n3", "0"]},
                {"id": 3, "terminals": ["n2", "n3"]},
                {"id": 4, "terminals": ["n1", "n2"]},
                {"id": 5, "terminals": ["n2", "0"]}],
 "_usage": {"input": 2383, "output": 233},
 "_model": "claude-opus-5"}
```

Per-image **scores** (not just predictions) are at
`results/vlm/<run>/scored/per_image.csv`, one row per image per repeat, with the
same metric columns `scripts/benchmark.py` emits for the pipeline.

---

## 12. Scoring — identical machinery, no per-provider branch

`scripts/score_vlm.py` calls `benchmark.score_prediction` and
`benchmark.aggregate` — the exact functions `scripts/benchmark.py` uses for the
pipeline — against the same verified GT at the same IoU threshold (0.3). No
metric is reimplemented and there is no per-provider branch, so neither model
gets a scoring advantage.

GT: `data/gt_test_1024`, 192 verified as-drawn files.
Detections for variant B alignment: `data/detections_valstop`, with the class
head applied exactly as `run_pipeline` applies it.

Two guards were added before these runs and both are exercised here:

* `--gt-dir` is explicit, and a run that scores **zero** images is a hard error
  rather than a summary full of 0.0. Scoring a val-era run against test GT now
  fails loudly; previously it wrote a clean-looking zero.
* `--split` asserts the scored stems are exactly the split manifest.

**Alignment differs by variant, and this is a real confound.** Variant B returns
our detection ids, so component matching is the identity map and cannot confound
the comparison. Variant A requires Hungarian matching at IoU 0.3 against boxes
the model invented, so detection quality confounds connectivity — part of a low
variant-A score is bad boxes rather than bad tracing. That is inherent to the
unaided task, not an artifact of the scorer, but it must be stated.

---

## 13. Results

All on the 192-image held-out test split. Pipeline row from
`results/final/benchmark/seed0/summary.json` (retrained detector).

### 13.1 Variant A — unaided, the end-to-end task

| system | strict success | tp F1 | net F1 | per-comp | nGED ↓ |
| --- | --- | --- | --- | --- | --- |
| **pipeline** | **0.5313** | **0.8172** | **0.8837** | **0.6570** | **0.1730** |
| claude-opus-5 | 0.1250 | 0.5561 | 0.7277 | 0.2791 | 0.2049 |
| gpt-5.5 | 0.1250 | 0.5047 | 0.6712 | 0.2528 | 0.2313 |

Both models land on exactly 24/192. The pipeline solves **84 circuits Claude
cannot** (6 the other way) and **81 GPT cannot** (3 the other way).

### 13.2 Variant B — connectivity only, models given our detections

| system | strict success | tp F1 | net F1 | per-comp | nGED ↓ |
| --- | --- | --- | --- | --- | --- |
| pipeline | 0.5313 | 0.8172 | 0.8837 | 0.6570 | **0.1730** |
| claude-opus-5 (mean of 3) | 0.5295 ±0.0167 | 0.8598 | 0.9065 | 0.6835 | 0.1745 |
| **gpt-5.5** | **0.6823** | **0.9316** | **0.9577** | **0.8211** | **0.1404** |

### 13.3 The finding

Per-component connected accuracy collapses from 0.68/0.82 (assisted) to
0.28/0.25 (unaided). **The models' failure is localised to detection and
classification, not to wire tracing** — given components, GPT traces
connectivity better than the pipeline does.

Reporting variant B alone would say a general model beats this pipeline.
Reporting variant A alone would say general models cannot do this task. Both are
misleading; the decomposition is the result.

### 13.4 Three-way hard core (strict success, majority over repeats)

| | circuits |
| --- | --- |
| pipeline | 102 |
| claude-opus-5 (variant B) | 106 |
| gpt-5.5 (variant B) | 131 |
| union (oracle over systems) | 145 |
| **all three fail** | **47/192 = 0.2448** |

The 47 are the priority human-review queue: three independent methods agreeing
on failure is evidence about the drawing or the annotation, not about one
method. They are **not** thereby proven information-limited — that requires the
human read.

---

## 14. What this record does NOT establish

* **Not** evidence that the ground truth is correct. No system here was checked
  against an independent human re-read; agreement and disagreement with the
  annotation are both unaudited.
* **Not** evidence the dataset is uncompromised. It measures three systems, not
  the data.
* **Not** a determinism result for GPT or for either variant A — one run each.
* **Not** a latency comparison. Both runs used Batch APIs whose turnaround is
  asynchronous and advertised in hours; hosted response time is a property of
  the host, not the model.
* **Not** an invoice. Section 9 costs are token-derived estimates.
