# Blind re-annotation packet -- connectivity ground truth

You are the independent second annotator. The packet holds 58 hand-drawn circuit
schematics, in two forms of the same drawings:

| directory | what it is | use it for |
| --- | --- | --- |
| `images/` | the photographs, exactly as they came off the camera (~2000 px) | **looking**: it has the most detail, so zoom into it to read faint pencil |
| `frames_1024/` | the same drawings normalised to a 1024 px frame | **coordinates**: every `[x, y]` you write down must be in this frame |

Both are the drawing and nothing else -- no boxes, no nets, no calls. The
1024 frame exists because the annotation schema records positions, and a
position is only meaningful once both passes agree on the frame it is in. Read
faint ink in `images/`, write coordinates from `frames_1024/`.

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

Hand back one directory containing, per image:

    <stem>.json             the netlist: your components, and the net each
                            terminal sits on
    decisions/<stem>.json   your call at each wire-ink intersection, plus notes

### Recording intersections by coordinate

Give each call the position you saw it at, **in the `frames_1024/` frame**:

```json
{
  "sites_xy": [
    {"xy": [434, 869], "call": "crossing"},
    {"xy": [612, 240], "call": "junction"}
  ],
  "notes": "S(434,869): plain X, no dot and no hop -- read as a crossing because ..."
}
```

`call` is one of `junction`, `crossing`, `none`, or an explicit edge grouping.

**Why coordinates and not index numbers.** The existing annotation numbers its
intersections, but that numbering is derived from where *that* annotator drew the
component boxes -- so the numbers are a fact about their pass, not about the
drawing, and you cannot be given them without being given part of their answer. A
position in a shared frame is the one thing both passes can name independently.

A coordinate is matched to an intersection within 12 px. If two intersections are
that close together, or two of your coordinates land on the same one, the call is
reported as unresolved rather than guessed -- so put the coordinate on the ink you
mean, and don't worry about hitting it exactly.

### Scoring

Comparison against the existing annotation is automatic:

    python scripts/compare_annotations.py --gt-b <your output directory>

`circuits.txt` lists the stems in this packet, in no meaningful order.

**Three of these circuits will not support the per-site comparison**, because
the first pass's own intersection numbering has drifted relative to the tracer
and its calls there can no longer be trusted to name the ink they once named.
They are not identified here: knowing which three would tell you where the
existing annotation is already suspect, and that is exactly the kind of hint
this packet exists to withhold. Annotate every circuit the same way. Their nets,
pin order and components are compared normally; only the per-site agreement
excludes them, and which three they are is recorded outside the packet.
