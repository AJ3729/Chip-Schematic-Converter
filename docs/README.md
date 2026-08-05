# Documentation

| document | what it is |
| --- | --- |
| [GT_VAL_VERIFICATION_REPORT.md](GT_VAL_VERIFICATION_REPORT.md) | **The account of how the test-split ground truth was built and checked.** Method, the volume of judgement recorded, the automated self-consistency re-derivation (an AI assistant, not an independent human reader — genuine second annotation is pending), every judgement call a reader should know about, and the residual risk stated plainly. This is the document to read if you want to decide whether to trust the benchmark. |
| [GT_VERIFICATION_GUIDE.md](GT_VERIFICATION_GUIDE.md) | The working protocol an annotator follows — what to look at, in what order, and what counts as evidence for a junction against a crossing. |
| [examples/](examples/) | Worked artifacts. |
| [development/](development/) | Engineering log and operational runbooks. Not part of the release; kept because several design decisions are only explicable from them. |

Top-level documents live at the repository root: [README.md](../README.md)
(orientation and headline result), [REPRODUCE.md](../REPRODUCE.md) (the exact
command behind every reported number), [data/README.md](../data/README.md)
(dataset provenance, splits, ground-truth format) and
[results/README.md](../results/README.md) (which artifact backs which table).
