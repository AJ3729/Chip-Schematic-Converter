# Ground-Truth Verification Guide (the [HUMAN] pass)

> **SUPERSEDED 2026-08-04 — do not follow this document for new annotation.**
> Use [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md) instead. This one is kept
> because the shipped corpus was produced under it, so it records how those
> decisions were actually made. Three things in it are now known to be wrong:
> it tells you to decide sloppy crossings "from circuit sense", which the
> as-drawn ruling withdraws; it gives a geometric pin-order heuristic
> ("drain/collector usually toward the supply") that the verification report
> blames for the observed pin-order error rates; and its paths and counts are
> stale (190 images, `data/gt_netlists/`, `data/cleaned/` — the canonical set
> is now 192 files in `data/gt_test_1024/` on `data/cleaned_1024/`).


This is the one task in the whole publication plan that must be done by
you, by hand. It is also the highest-leverage thing you personally
contribute: **every benchmark number in the paper is computed against
these files**, and the paper's claim that ground truth was "manually
created and verified" becomes true only when you finish this pass.
Reviewers formally score "do the data support the conclusions" — bad GT
poisons every downstream table.

**What it is**: for each of the 190 test-split images, confirm (or fix)
a JSON file that records which components exist and which electrical
net each terminal connects to. Then mark it verified with your name.

---

## 1. Prerequisites — what you need to know

**Circuit reading (must have).** You need to identify by sight: the 16
electrical symbol classes — Resistor, Capacitor, Inductor, Diode,
Zener Diode, GND (ground), V-DC / V-AC (two-terminal voltage sources),
V-DC (one port) (a labeled supply rail touching the circuit at a single
point), I-DC / I-AC (current sources), MOSFET-N / MOSFET-P, BJT-NPN /
BJT-PNP, Op-Amp — and be able to trace wires. If two symbols look
ambiguous in someone's handwriting (e.g. MOSFET-N vs MOSFET-P), consult
the published reference crops in
`data/digitize_hcd/extracted/Digitize-HCD Dataset/Component Port
Location Data/<Class>/Input Images/` — thousands of examples per class.

**The net concept (must have).** An electrical *net* is a maximal set
of terminals connected by wire. Two terminals joined by any conductive
path share one net; a crossing without a junction dot does NOT join
nets. The net touching a GND symbol is always named `"0"`. Other names
are arbitrary labels (`n1`, `n2`, …) — only the *grouping* matters to
the metrics, not the names.

**JSON editing (basic).** You will edit values in a text editor. Any
editor works; VS Code is convenient because it flags syntax errors
(a missing comma etc.) inline.

**Terminal counts and order (reference, keep open while working):**

| Class | Terminals | Order (index 0, 1, 2) |
| --- | --- | --- |
| Resistor, Capacitor, Inductor, Diode, Zener Diode, V-DC, V-AC, I-DC, I-AC | 2 | left/right or top/bottom as drawn |
| GND | 1 | the wire it grounds |
| V-DC (one port) | 1 | the wire it feeds |
| MOSFET-N, MOSFET-P | 3 | drain, gate, source |
| BJT-NPN, BJT-PNP | 3 | collector, base, emitter |
| Op-Amp | 3 | in+, in−, out |
| Wire Crossover | — | never appears in GT (drawing annotation, not a component) |

For two-terminal passives the terminal order does not affect any
metric; for sources/diodes/transistors use the drawn orientation as
listed. The validator (`--check`) enforces these counts.

---

## 2. Where everything lives

| Path | What it is |
| --- | --- |
| `data/gt_netlists/circuit_<n>.json` | the GT files you edit (190 test images) |
| `data/gt_netlists/renders/circuit_<n>.png` | machine-drawn overlay of the current GT file — your visual check |
| `data/cleaned/circuit_<n>.jpg` | the preprocessed image the overlay is drawn on |
| `data/raw/circuit_<n>.jpg` | original photo — consult when the cleaned image is unclear |
| `data/splits/test.txt` | the authoritative list of the 190 images |
| `docs/GT_VERIFICATION_GUIDE.md` | this guide |

The GT files were bootstrapped by machine
(`source: "coco+pipeline_bootstrap"`): the **component list comes from
the published Digitize-HCD annotations** (complete and correctly
classified — you will rarely need to add or re-class a component) and
the **net assignments come from the current heuristic pipeline**, which
is right roughly 3 times out of 4. **Your job is mostly to check and
fix nets.** Terminals the pipeline could not resolve are `"net": null`.

## 3. One-time setup (≈ 5 min)

```bash
cd ~/Documents/Chip-Schematic-Converter
source venv/bin/activate
python scripts/annotate_topology.py --check     # baseline: all load, 0 verified
python scripts/annotate_topology.py --render    # regenerate overlays
```

Open `data/gt_netlists/renders/` in Finder (gallery view) so you can
flick through overlays, and keep a terminal + editor side by side.

## 4. Per-image workflow

Work from `data/splits/test.txt` top to bottom so nothing is missed.
For each image `circuit_<n>`:

1. **Open** `renders/circuit_<n>.png` (and `data/cleaned/circuit_<n>.jpg`
   if you need an unannotated view). Each component shows its id +
   class; each terminal is a colored dot labeled with its net name —
   same color = same net.
2. **Sweep components (≈15 s).** Every drawn symbol boxed? Every class
   right? (Rarely wrong — the inventory is from the published
   annotations.) Fix a wrong `"class"` by typing the canonical name
   from the table above. If something is genuinely missing, append a
   new component object (copy an existing one; next unused `"id"`;
   `"bbox"` is `[center_x, center_y, width, height]` in cleaned-image
   pixels — eyeball it; the bbox is a visual aid only, metrics never
   read it).
3. **Trace nets (the real work, 1–3 min).** Follow each wire run in the
   drawing and check that every terminal it touches carries the same
   net name, and that separate wire runs carry different names:
   - Merge two nets the pipeline wrongly split: rename all terminals of
     one to the other's name.
   - Split a net the pipeline wrongly merged (usually a crossover it
     treated as a junction): give one group a fresh unused name.
   - Fill every `"net": null` with the right net name.
   - The net touching any GND symbol must be renamed `"0"`
     (all GND symbols in one image usually share `"0"`; if the drawing
     truly has two separate grounds, they are still all `"0"` —
     electrically one reference).
   - A terminal that genuinely connects to nothing in the drawing:
     leave `null` and add `"unconnected": true` to that component.
4. **Re-render and eyeball**:
   ```bash
   python scripts/annotate_topology.py --render
   ```
   (a few seconds for all files; re-open the PNG — colors now reflect
   your edit; wrong groupings jump out as same-color dots on
   unconnected wires).
5. **Sign off**: set `"verified": true` and `"annotator": "Ammaar
   Junaid"` in the JSON. Do this only when you are actually confident —
   an unverified file is honest; a wrongly verified one is a landmine.
6. **Validate periodically** (every ~10 images):
   ```bash
   python scripts/annotate_topology.py --check
   ```
   Verified files are checked strictly: every terminal must have a net
   (or `"unconnected": true`), terminal counts must match the class,
   GND must sit on net `"0"`, and any net touching only one terminal is
   flagged as suspicious. Fix everything it prints.

### Judgment calls you will hit

- **Crossing vs junction**: a dot at the crossing (or a T-shape) =
  junction = same net. A plain X crossing with no dot = different nets.
  When the drafter was sloppy, decide from circuit sense (would the
  circuit be nonsense otherwise?) and note your reasoning in `"notes"`.
- **Terminal order for transistors**: use the drawn orientation
  (drain/collector usually toward the supply, source/emitter toward
  ground; gate/base is the odd pin out). If truly ambiguous, pick the
  physically sensible reading and note it.
- **Op-Amp in+/in−**: read the +/− marks in the drawing; index 0 = in+,
  1 = in−, 2 = out.
- **Component drawn but connected to nothing**: keep it, mark
  `"unconnected": true`.
- **Never delete a component because the pipeline missed it** — GT
  describes the drawing, not the pipeline's abilities.

## 5. Realistic time budget

| Item | Estimate |
| --- | --- |
| Read this guide + first 3 images slowly | 30–45 min |
| Typical image (nets mostly right, small fixes) | 1.5–2.5 min |
| Messy image (dense, crossings, transistors) | 5–8 min |
| **Total for 190 images** | **6–8 hours of focused work** |

Do it in 5–6 sessions of 30–40 images (~75–90 min each); accuracy
drops sharply with fatigue, and this data must be right. After ~20
images you will be much faster — do NOT go back and rubber-stamp the
first 20; re-check them with your trained eye.

## 6. Done means

```bash
python scripts/annotate_topology.py --check
# [SUMMARY] 191 GT file(s): 190 verified, 1 unverified, 0 with validation issues
```

(The 1 unverified file is `circuit_1199.json` — a demo image not in the
test split; verify it too if you like, it just isn't required.) Then
commit: I (Claude) will handle committing whenever you say the word, or
commit yourself:

```bash
git add data/gt_netlists   # note: data/ is gitignored — this needs a force-add
```

Actually, **tell me when you're done instead** — the GT files need a
deliberate decision about whether they are published inside the repo or
as a Zenodo artifact (they are part of benchmark contribution C1), and
I'll wire that up properly with the numbers audit.
