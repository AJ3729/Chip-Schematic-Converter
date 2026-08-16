# CGHD annotation sampling design (task B8)

**For review before annotation begins.**

Queue: `data/cghd/annotation_queue.json`, seed 20260815, 50 drawings from a pool of 119.

## The rule

Selection is **blind to whether the pipeline gets a circuit right**.
It uses drafter, component count and capture index only. Component
count comes from the *annotation*, never from detector output —
using the detector would leak a pipeline product into the sample and
bias the evaluation in our favour.

Drawings, not images: each drawing carries up to four photographs, so
one annotated netlist scores several images through the capture
grouping.

## Balance at every prefix

| prefix | drawings | images scored | drafters | complexity deciles |
| --- | --- | --- | --- | --- |
| first 10 | 10 | 40 | 10 | 5 |
| first 20 | 20 | 80 | 18 | 8 |
| first 40 | 40 | 160 | 18 | 9 |
| full | 50 | 200 | 18 | 9 |

Stopping after any of these prefixes still leaves a stratified
sample. That is the point of the ordering.

## Double annotation

Every 6th drawing (15%) is
re-queued after a delay for self-agreement measurement (task E4).

## Caveat the reviewer should weigh

Cross-corpus detection transfers at mAP@0.5 0.3445, and the pipeline
localises 0.486 of components. Netlist-level scores on this sample
will therefore be dominated by missing components rather than by
wire-tracing errors. The sample is still worth annotating — it is the
only way to put a number on cross-corpus reconstruction — but it will
measure the detector more than the tracer, and the effort should be
budgeted with that in mind.
