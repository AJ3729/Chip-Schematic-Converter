# IEEE Access manuscript

**Status: draft under revision.** The prose is the author's; every number and
every table is machine-generated from `results/`.

## Layout

```
main.tex              root; targets ieeeaccess.cls, falls back to IEEEtran
sections/*.tex        one file per section
generated/numbers.tex \newcommand macros — AUTO-GENERATED, never edit
tables/*.tex          booktabs bodies — AUTO-GENERATED, never edit
figures/*.pdf         data figures — AUTO-GENERATED, never edit
figures/pipeline_figure.tex   the one hand-authored figure (TikZ)
```

To submit: drop this directory into the official IEEE Access Overleaf template.

## The no-hand-typed-numbers rule

Every numeric value in the prose is a macro from `generated/numbers.tex`, and
every table body is generated. Nothing is transcribed by hand at any point:

```bash
python scripts/make_paper_tables.py       # macros + tables
python scripts/make_paper_figures.py      # the five data figures
python scripts/make_paper_qualitative.py  # the failure gallery
python scripts/audit_paper_numbers.py     # FAILS if a literal result value appears
```

The audit is the enforcement, not the convention. It has caught real drift more
than once — including two literals typed straight into figure captions.

Regenerate after any change under `results/`. `make_paper_tables.py --variant`
selects the result set: `test` (default, the reported held-out split), `1024`
(the same pipeline on the split every parameter was tuned on, i.e. validation
numbers), `512` (superseded). **Never mix two variants in one table.**

## Figures

| file | shows |
| --- | --- |
| `fig_precision_cliff` | strict success by terminal-pair precision, both splits |
| `fig_ablation_waterfall` | cumulative strict success, v1 → v12 |
| `fig_oracle_waterfall` | stage attribution by ground-truth substitution |
| `fig_per_class_ap` | per-class detection against class support |
| `fig_size_scatter` | per-circuit accuracy against circuit size |
| `fig_failure_gallery` | one circuit per failure mode, GT against prediction |

The gallery selects its circuits by per-image metric — largest precision/recall
asymmetry, most unmatched components, largest strict success — never by eye, so
it cannot become a curated best case.

## Draft conventions

- `\todoa{...}` — red: decisions only the author can supply (affiliation, ORCID,
  mentor status, repository URL, final scoping).
- `\draftnote{...}` — orange: what a passage still needs.

Both must be stripped before submission:

```bash
grep -rn 'todoa\|draftnote' paper/sections paper/main.tex
```

## Known gaps

- **No bibliography.** The `\cite{}` keys mark where citations belong and which
  work is meant; they render as `[?]` until a `.bib` is added. The commented
  `\bibliographystyle` / `\bibliography` lines in `main.tex` are where it plugs
  back in.
- **No graphical abstract.** IEEE Access requires one (660×295 px, ≤45 KB).
- **Never compiled.** There is no LaTeX toolchain on the development machine, so
  page count, float placement and overfull boxes are unverified.
