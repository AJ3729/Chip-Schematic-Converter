# CGHD blind-set annotation packet

**Status: PREPARED, NOT ANNOTATED.** Every terminal in `gt/*.json` is
`"net": null`. No pipeline, detector or wire-tracer output was used to build
anything in this directory. Read `../READINESS.md` first -- it states what
this set can and cannot establish, and it contains the freeze-then-evaluate
protocol you must follow.

## What is here

| path | what |
| --- | --- |
| `gt/<stem>.json` | component inventory ONLY: class + bbox from CGHD's published Pascal-VOC annotations, projected to the 1024 frame. **All nets null.** |
| `decisions/<stem>.json` | empty decision record, same schema as `data/gt_test_1024/decisions/`. Fill it in; do not edit `gt/` by hand. |
| `aux/<stem>.json` | CGHD's own junction / crossover / text / terminal boxes, and the per-component mapping calls you must confirm. Evidence, not answers. |
| `../../../data/cghd_blind_1024/images/<stem>.jpg` | the 1024 frame to annotate against. |

## Which guide is authoritative

**Rules: `docs/ANNOTATION_GUIDE.md`.** It is the current guide and it carries
the *annotate as drawn* rule. The older `scripts/gt_val_tools/BRIEF.md` told
annotators that "electrical impossibility wins" at an ambiguous crossing; that
rule is **withdrawn**, and following it here would produce ground truth that
silently repairs the drawing.

**Tooling and commands: `scripts/gt_val_tools/BRIEF.md`.** The decisions-file
schema is identical in both, so nothing else diverges.

## Differences from the Digitize-HCD pass you should know about

1. **Classes need confirming, not just accepting.** CGHD does not split
   NPN/PNP or N/P MOSFET, so every `transistor.bjt` and `transistor.fet`
   arrives as `BJT-NPN` / `MOSFET-N` by default. Read the arrow and set the
   real class through the `classes` key. `aux/<stem>.json` lists exactly
   which component ids this affects.
2. **`vss` is not necessarily ground.** It is mapped to `GND` by
   `data/cghd/class_mapping.yaml`, which would force it onto net `"0"`. If
   the drawing means a supply rail, change the class to `V-DC (one port)`.
3. **CGHD annotates junctions and crossovers itself.** `aux` carries those
   boxes. They are the dataset authors' reading of the drawing and are the
   single biggest saving relative to the Digitize-HCD pass -- but they mark
   a location, not a partition of wires, so you still make the call.
4. **The images are photographs from a different corpus.** Expect different
   paper, pens, lighting and framing.

## Procedure

Annotate to `docs/ANNOTATION_GUIDE.md` -- the same guide, the same rules and
the same schema as the Digitize-HCD test split, so the two sets stay
comparable. Build the per-image review package (overlay, site crops,
component crops) at annotation time:

```
python scripts/gt_val_tools/batch.py <val-root> <pkg-out>
```

where `<val-root>` holds `img1024/<stem>.jpg` and `gt/<stem>.json`. Running
the tracer *then* is correct and is what the Digitize-HCD pass did: it
proposes intersection sites for a human to adjudicate. It was deliberately
not run *here*, so that nothing in the committed packet originates from the
system under test.

Finish each sheet with `finalize.py`, which writes the ERC-checked GT from
your decisions file. Set `verified: true` and `annotator` yourself -- that
sign-off is a human action by design.
