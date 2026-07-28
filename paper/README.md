# IEEE Access manuscript

**Status: scaffold + section drafts for Ammaar's revision. Nothing here
is final — the scholarly voice belongs to the author.**

## Layout

- `main.tex` — root; targets the official IEEE Access class
  (`ieeeaccess.cls`) and falls back to `IEEEtran` so it compiles
  anywhere. To submit: drop this directory into the official IEEE
  Access Overleaf template.
- `sections/*.tex` — one file per section. `introduction`,
  `related_work`, `dataset`, `method`, `experimental_setup` are
  **drafted**; `results`, `discussion`, `conclusion` are **skeletons**
  that fill as experiments land.
- `figures/pipeline_figure.tex` — TikZ vector pipeline figure.
- `generated/numbers.tex`, `tables/*.tex` — **auto-generated** by
  `python scripts/make_paper_tables.py` from `results/` artifacts.
  Never edit these by hand; never type a number into the prose.
  This is how the project enforces its no-hand-typed-numbers rule.
- **No bibliography yet, by choice.** References are a Week-4 task. The
  `\cite{}` keys in the text mark where citations belong and which work
  is meant; they render as `[?]` until a `.bib` is added, which does
  not affect drafting. The commented-out `\bibliographystyle` /
  `\bibliography` lines in `main.tex` are where it plugs back in.

## Draft conventions

- `\todoa{...}` — red: decisions/inputs only Ammaar can provide
  (affiliation, ORCID, mentor status, final scoping).
- `\draftnote{...}` — orange: notes from the drafting session about
  what a passage still needs. Strip both before submission
  (`grep -rn 'todoa\|draftnote' paper/sections` must come back empty).

## Regenerating numbers

```bash
python scripts/make_paper_tables.py
```

Re-run after any change under `results/`. The Week-4 numbers audit
cross-checks every macro against the CSVs.
