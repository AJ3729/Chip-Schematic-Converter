# CGHD — dataset provenance

Task B1. Records exactly which release of CGHD this project uses, where it came
from, and what its license permits.

---

## 1. Version audit

The plan cites Zenodo record `14042961`. That is **not** the current release,
and the copy already on disk is older still. All three are recorded so the
choice is visible rather than implicit.

| | version | published | DOI | license | size |
| --- | --- | --- | --- | --- | --- |
| **on disk here** | **12** | 2023-10-31 | (record `10056817`) | **CC BY-SA 3.0** | 3.43 GB |
| cited by the plan | 14 | 2024-11-08 | `10.5281/zenodo.14042961` | CC BY 4.0 | 4.4 GB |
| **current latest** | **16** | **2025-10-25** | **`10.5281/zenodo.17469897`** | **CC BY 4.0** | 4.88 GB |

Concept DOI (always resolves to latest): `10.5281/zenodo.6385813`
Code repository: <https://github.com/DFKI/cghd> — code and documentation only;
the versioned data lives on Zenodo.

Resolved by querying the Zenodo API rather than by reading the landing page,
because the landing page for v14 says only that "a newer version exists"
without naming it.

## 2. License, and why the version choice is not cosmetic

**This is the STOP-condition check on licensing. It passes — but only for the
newer releases.**

* **v16 / v14 — CC BY 4.0.** Permits research use and redistribution of derived
  annotations, requiring attribution only. Our CGHD netlist annotations could
  be released under whatever license this project chooses.
* **v12 (on disk) — CC BY-SA 3.0**, per the `README.md` inside the archive.
  Attribution *plus* share-alike. Under a share-alike license, annotations
  derived from the images plausibly inherit the copyleft obligation, which
  would constrain how this project's own ground truth may be released.

So the two candidate versions differ in what we are allowed to do with the
annotations the author is about to spend weeks producing. That is a substantive
reason to prefer v16 independent of its extra content.

Neither license prohibits the intended use. **No stop condition is triggered.**

## 3. Citation

The dataset paper, which is reference 11 in the manuscript:

```bibtex
@inproceedings{thoma2021public,
  title     = {A Public Ground-Truth Dataset for Handwritten Circuit Diagram Images},
  author    = {Thoma, Felix and Bayer, Johannes and Li, Yakun and Dengel, Andreas},
  booktitle = {International Conference on Document Analysis and Recognition},
  pages     = {20--27},
  year      = {2021},
  organization = {Springer}
}
```

The Zenodo record must be cited **in addition**, with its version and DOI,
because the paper describes an earlier and much smaller release than any
version considered here.

## 4. Contents of the copy on disk (v12)

Full inventory: `reports/cghd_inventory.md`. Summary:

| | |
| --- | --- |
| drafters | 25 (`drafter_0` … `drafter_24`) |
| image files | 3,366 (3,341 distinct stems; 25 stems appear under two extensions) |
| bounding-box annotations | 2,449 (Pascal VOC XML) |
| binary segmentation maps | 284 |
| instance polygons | 257 (labelme format) |
| LTspice `.asc` files | 13, all under `drafter_1/spice/` |
| classes | 53 |

Naming is `C<circuit>_D<drawing>_P<picture>`: 12 circuits per drafter, 2
drawings per circuit, up to 4 photographs per drawing.

## 5. Documented-versus-actual discrepancies

The plan asks for these to be recorded rather than smoothed over.

1. **Annotation coverage is not universal.** The README states "For every Raw
   image in the dataset, there is an accompanying bounding box annotation
   file." In fact 917 images have no annotation — **all 917 belong to
   `drafter_0`**, which holds 1,038 images against 121 annotations. Drafters
   1–24 are each exactly 97 images / 97 annotations, a clean 1:1.

   Consequence: the usable annotated corpus is drafters 1–24, i.e. **2,328
   images across 24 drafters**, not 3,341 across 25.

2. **The `.asc` files are circuit templates, not per-drawing netlists.** All 13
   sit under `drafter_1/spice/` and are named `C1`–`C12` — one per *circuit
   type*, not one per drawing. `C1.asc` is titled "LED pulse". These describe
   the circuit each drafter was asked to draw.

   They are **intent, not as-drawn topology**, so they cannot substitute for
   annotation: a drafter may have drawn the circuit incorrectly, and the
   project's as-drawn rule requires recording what is on the page. They are
   useful as a cross-check and are recorded for that purpose only.

3. **Known labelling issues shipped by the dataset authors** (README): `C25_D1_P4`
   and `C27` cut off text, `C29_D1_P1` has an extra text, `C31_D2_P4`,
   `C33_D1_P4` have one text fewer, `C46_D2_P2` cuts off text. These affect the
   text class, which this pipeline suppresses rather than scores.

## 6. Open decision — which version to evaluate on

**Not yet resolved; recorded here so it is not decided by default.**

Using v12 because it happens to be on disk would mean evaluating on a release
that is two versions and two years stale, under a share-alike license that
constrains release of our own annotations. Using v16 costs a 4.88 GB download
and a re-run of this inventory, and changes the drafter and image counts that
the sampling design (B8) will be built on.

Recommendation: **v16**. The licensing difference alone justifies it, and a
reviewer can check the current Zenodo version in seconds.

Until this is decided, every downstream CGHD task states which version it used.
