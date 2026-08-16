# CGHD netlist annotator

Local, keyboard-driven tool for tracing connectivity from CGHD photographs.
Task C1. Runs on localhost against the local filesystem; nothing leaves the
machine.

## Start

```bash
./venv/bin/python tools/annotator/server.py --tutorial   # calibrate first
./venv/bin/python tools/annotator/server.py              # the real queue
```

Then open <http://127.0.0.1:8765>.

**Do the tutorial first.** It serves three Digitize-HCD circuits whose ground
truth already exists. Annotate one yourself, then press *reveal known answer*
and compare. It takes ten minutes and it calibrates you against a right answer
before any CGHD effort is spent.

## Keys

| key | |
| --- | --- |
| `t` / `i` | terminal mode / intersection mode |
| click | place a terminal (or an intersection, in `i` mode) |
| `n` | next unused net name |
| `0` | select ground — the net a GND symbol touches must be `0` |
| `j` `k` `e` `o` | set the last intersection to junction / crossing / edge group / none |
| `x` | undo the last placement |
| `c` | focus the class selector |
| `s` | save draft |
| `[` `]` | previous / next drawing |
| `Enter` | submit |
| `Esc` | reset zoom |
| wheel | zoom · shift-drag | pan |

Terminals are placed **in port order**. For a BJT that is collector, base,
emitter; the panel shows the expected order for the selected class and flags a
component whose terminal count is wrong.

## The rules this tool enforces

**It never shows you pipeline output.** There is no endpoint that serves
detections or predicted nets, so it cannot pre-fill or suggest one even by
accident. Circular evaluation is prevented structurally, not by discipline.

**Record what is drawn.** If a ground is missing, leave it missing. If a node
floats, leave it floating. Repairs you *would* apply go in the interventions
box and are stored in a separate field — never folded into topology.

## Output

Submissions are written to `data/cghd/annotations/incoming/`. Run

```bash
./venv/bin/python scripts/sync_board.py
```

to validate and file them. Valid records move to `accepted/`; invalid ones move
to `rejected/` with a `.errors.txt` naming the exact problem. The validator
flags, it never auto-corrects — an annotation that disagrees with the detector
is a finding, not an error.

Progress, stratification coverage, self-agreement and a completion projection
are regenerated into `reports/annotation_progress.md` on every sync.

Drafts autosave every 10 seconds and on navigation, so a circuit can be left
half-finished and resumed.
