# Ground-truth re-check brief

A first annotator has already produced a finished GT file for each image. You
are the **second reader**. Your job is not to rubber-stamp it: it is to re-derive
the netlist from the drawing yourself and then see whether the two agree.

Read `/home/claude/tools/BRIEF.md` first — it defines the task, the conventions,
the tools and the decisions-file schema. Everything there applies here.

## What is different for you

* The first pass left a decisions file at `/home/claude/dec/<stem>.json` and a
  finished GT at `/home/claude/out/gt/<stem>.json`. The GT's `notes` field
  records the reasoning, site by site.
* **Do not read the notes until you have formed your own reading of the
  circuit.** Read the drawing, work out the nets, and only then open the notes
  and compare. If you read the reasoning first you will simply agree with it.
* Where you disagree, settle it in the pixels (`inkmap.py`), not by argument.
* If you agree, say so and change nothing.

## Sibling evidence

These sheets are hand drawings of *generated* circuits, and the same generated
circuit was often drawn more than once. `/home/claude/cal/gt/` holds 190
**human-verified** annotations of the test-split drawings, with images in
`/home/claude/cal/img1024/`. To find sheets with the same component inventory:

```
python3 /home/claude/tools/siblings.py <stem>
```

It prints each sibling's net count and per-component net assignment. A sibling
is evidence, not proof — the same parts list can be wired differently, and in
the verified set only about two thirds of sibling pairs share a net count. Use
it to *generate a hypothesis* about what the intended topology is, then go and
check that hypothesis against the ink of your own drawing. Never copy a
sibling's answer.

## If you change something

Edit `/home/claude/dec/<stem>.json`, re-run

```
python3 /home/claude/tools/finalize.py <stem> /home/claude/dec/<stem>.json /home/claude/val
python3 /home/claude/tools/pkg.py <stem> /home/claude/val /home/claude/pkg/<stem> /home/claude/dec/<stem>.json
```

and look at the regenerated overview. Extend `notes` with what you changed and
why — keep the first pass's reasoning, add yours; the note is the audit trail.

## Report

Per image: AGREE or CHANGED. If CHANGED, say exactly which terminals moved
between nets and what pixel evidence settled it. If you agree but the file still
carries an ERC error or contradicts its siblings, explain why the drawing really
is that way — an unexplained ERC error is not an acceptable end state.
