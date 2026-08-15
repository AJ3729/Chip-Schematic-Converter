# CGHD inventory (task B1)

Contents of the copy on disk, `data/cghd/cghd-zenodo-12.zip` (**version 12**,
2023-10-31). Version audit and licensing: `data/cghd/PROVENANCE.md`. A newer
release (v16, 2025-10-25) exists and the choice between them is open — see §6
of that file.

Counts below are of **distinct stems**, not file rows: 25 images appear under
two extension spellings (`.jpg` and `.JPG`), so a naive file count reports
3,366 where there are 3,341 distinct images.

---

## 1. Headline

| | |
| --- | --- |
| drafters | 25 |
| **drafters usable (fully annotated)** | **24** |
| images, all drafters | 3,341 |
| bounding-box annotations | 2,424 |
| **evaluable pool** | **2,304 images** |
| **distinct physical drawings in pool** | **576** |
| classes shipped | 53 |

## 2. The corpus has a perfectly regular structure

Filenames are `C<circuit>_D<drawing>_P<picture>`, and for drafters 1–24 the
design is exactly balanced:

```
24 drafters  x  12 circuits  x  2 drawings  x  4 photographs  =  2,304 images
```

Verified rather than assumed: the pictures-per-drawing histogram over the
evaluable pool is `{4: 576}` — **every one of the 576 drawings has exactly four
photographs, and none has any other count.**

This matters more than a dataset statistic normally would. It means the
capture-invariance experiment (B7) is not a scavenged subset but the whole
pool: 576 independent drawings, each photographed four times under different
camera positions and illuminations. One ground-truth netlist per drawing yields
four scored images, and the invariance measure itself needs no ground truth at
all.

Circuit numbering is global (`C1`–`C288`), so a circuit id identifies both the
drafter and which of that drafter's twelve circuits it is.

## 3. Per-drafter composition

Drafters 1–24 are identical in shape: **97 images, 97 annotations** each, a
clean 1:1. `drafter_0` is the exception and is excluded — see §4.

| subset | drafters | images | annotations |
| --- | --- | --- | --- |
| `drafter_0` | 1 | 1,038 | 121 |
| `drafter_1` … `drafter_24` | 24 | 2,304 | 2,304 |

Auxiliary annotation, present only for a minority of images and not required by
this project's pipeline:

| type | count | format |
| --- | --- | --- |
| binary segmentation maps | 284 | JPEG, stroke vs background |
| instance polygons | 257 | labelme JSON |
| LTspice schematics | 13 | `.asc`, all under `drafter_1/spice/` |

## 4. Why `drafter_0` is excluded

`drafter_0` holds 1,038 images but only 121 annotations — **917 images with no
bounding-box annotation, and every unannotated image in the corpus is one of
them.** The README states the opposite ("For every Raw image in the dataset,
there is an accompanying bounding box annotation file"), so this is a
documented-versus-actual discrepancy, recorded rather than worked around.

Excluding `drafter_0` costs nothing this project needs: detection transfer (B5)
requires annotations to score against, and the capture-invariance grouping
requires the regular `C_D_P` structure that only drafters 1–24 have.

## 5. Annotation formats and coordinate conventions

* **Bounding boxes** — Pascal VOC XML, one file per image. Absolute pixel
  coordinates in the *native* image resolution, `xmin/ymin/xmax/ymax`
  (corner form). The pipeline uses centre form in a 1024 frame, so the adapter
  (B3) must convert both the parameterisation and the coordinate space, and
  must prove the round trip closes to within one pixel.
* **Instance polygons** — labelme JSON, native pixel coordinates. Deliberately
  coarse; intended to be used together with the binary segmentation maps.
* **Segmentation maps** — same resolution as the source image.
* **Images** — RGB, `.jpg` / `.jpeg` / `.png`, mixed-case extensions. Native
  resolution varies; this is the photographic variation the corpus exists to
  supply, and B4 measures it.

## 6. Classes

CGHD ships **53** classes against this pipeline's 17. The extras are of three
kinds: finer-grained electrical distinctions, non-component annotation classes
that the pipeline handles structurally rather than as components (`text`,
`junction`, `crossover`, `terminal`), and components with no counterpart in the
17-class vocabulary.

Mapping every CGHD class to one of {a Digitize-HCD class, `OUT_OF_VOCABULARY`,
`AMBIGUOUS`} is task B2. Circuits containing out-of-vocabulary components
cannot be fairly scored by a 17-class pipeline and are excluded from the
evaluable pool, with the exclusion counted and reported.

`data/cghd/class_mapping.yaml` predates this plan and is superseded by B2's
output; it is not used.

## 7. The `.asc` files are intent, not ground truth

All 13 LTspice files sit under `drafter_1/spice/` and are named `C1`–`C12` —
one per *circuit type*, not one per drawing. `C1.asc` is titled "LED pulse".
They describe the circuit each drafter was **asked** to draw.

They are therefore not a shortcut around annotation. This project's ground truth
records the topology visibly on the page; a drafter may have drawn the intended
circuit incorrectly, and that error is part of what the corpus tests. The `.asc`
files are recorded as a cross-check — a drawing whose annotation diverges wildly
from its template is worth a second look — and for nothing else.

## 8. What is already deliverable from this inventory

Before a single netlist is annotated:

* **B5, detection transfer** — 2,304 images with bounding-box ground truth,
  24 drafters, scoreable immediately against the frozen detector.
* **B7, capture invariance** — 576 drawings × 4 captures, needing no ground
  truth for the invariance measure itself.
* **B4, imaging characterization** — native resolutions, aspect ratios and
  capture variation, measurable on both corpora now.

## 9. Verification against the dataset paper

The manuscript's reference 11 (Thoma et al., ICDAR 2021) describes a
substantially smaller and earlier release than v12. Counts in this report
should not be compared against that paper's; the Zenodo record and version are
the citable source for corpus size, and both must appear in the manuscript.
