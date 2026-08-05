# Blind re-annotation packet -- connectivity ground truth

You are the independent second annotator. Everything you need is in `images/`:
58 photographs of hand-drawn circuit schematics, exactly as they came off the
camera.

## What to produce

For every image, one JSON file recording, for each component, which electrical
**net** each of its terminals connects to, and -- at every place where wire ink
meets -- whether the wires are **joined** or one **passes over** the other.

The file format, the net naming rule, the terminal-numbering convention for
3-terminal parts, and the worked examples are all in:

    docs/ANNOTATION_GUIDE.md

Read it before you start. If anything in the guide contradicts this README, the
guide wins.

## Ground rules

1. **Work only from the photographs.** Do not look at any other directory in
   this repository, and in particular do not open anything under `data/gt_*`,
   `results/`, or any file named `*netlist*`, `*decision*` or `*render*`. An
   existing annotation of these same images exists; the entire value of your
   pass is that you have not seen it.
2. **Do not run the pipeline** or any VLM on these images, and do not consult
   any automatic tool's opinion about connectivity. If you want a magnified
   view, zoom the photograph.
3. **Record judgement calls, don't hide them.** Where a crossing has neither a
   solder dot nor a hop, both readings are defensible; say which you chose, at
   which coordinates, and what made you choose it. Disagreements with a reason
   attached are useful data. Silent ones are noise.
4. **Do not skip a circuit because it looks ambiguous.** Ambiguous circuits are
   the ones this exercise is measuring. Annotate it and flag it.
5. **Do not consult the other annotator** (or anyone who has seen the first
   pass) about a specific circuit until your pass is complete and delivered.

## What you are NOT told, on purpose

These circuits were sampled in strata -- some at random, some because they hold
a 3-terminal device, some because several automatic systems disagreed with the
existing annotation. You are not told which is which, because knowing that a
particular drawing is "one of the hard ones" changes how long you look at it and
what you expect to find. Treat every image as equally likely to be routine.

## Delivering

Write one `<stem>.json` per image into a single output directory and hand back
that directory. Comparison against the existing annotation is automatic:

    python scripts/compare_annotations.py --gt-b <your output directory>

`circuits.txt` lists the stems in this packet, in no meaningful order.
