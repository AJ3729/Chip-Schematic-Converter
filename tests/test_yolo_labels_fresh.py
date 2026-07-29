"""The YOLO detection labels must match the CURRENT preprocessing.

`data/cleaned` was regenerated on 2026-07-27 (the preprocessing fix) and
`data/transforms.json` with it, but `data/yolo_cleaned/labels/` was left
at its 2026-07-23 projection. Nothing noticed. The committed detection
mAP of 0.9725 then failed to reproduce — re-running the same command
returned 0.051 — because every label carried a systematic ~0.04
normalized y-offset against the images it was paired with.

The failure is invisible from either side on its own: the labels parse,
the images load, the counts agree, and `eval_detector.py` reports a
number rather than an error. Only the value is wrong. So this test
re-derives labels from the published COCO annotations through the
transforms on disk and compares them to what is committed — if
preprocessing moves again, this fails instead of a headline number
quietly going stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COCO = REPO / ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
               "Component Symbol and Text Label Data/component_annotations.json")

# (label dir, frames' transforms) pairs that must stay in step
DATASETS = [
    ("data/yolo_cleaned_rebuilt", "data/transforms.json"),
    ("data/yolo_1024", "data/transforms_1024.json"),
]

TOL = 1e-4          # labels are written with 6 decimals
SAMPLE = 12         # images per dataset; the defect was systematic, not sparse


def _expected(stem: str, coco: dict, transforms: dict) -> list[tuple]:
    from schematic2netlist.preprocess import project_bbox

    by_name = {i["file_name"]: i["id"] for i in coco["images"]}
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_index = {c["id"]: i for i, c in enumerate(cats)}
    iid = by_name[f"{stem}.jpg"]
    meta = transforms[stem]
    W = H = meta["target_size"]
    out = []
    for a in coco["annotations"]:
        if a["image_id"] != iid:
            continue
        x, y, w, h = a["bbox"]
        cx, cy, bw, bh = project_bbox(meta, x, y, w, h)
        if cx < 0 or cy < 0 or cx > W or cy > H or bw <= 0 or bh <= 0:
            continue
        out.append((class_index[a["category_id"]], cx / W, cy / H, bw / W, bh / H))
    return out


@pytest.mark.parametrize("ds,tf", DATASETS)
def test_labels_match_current_transforms(ds, tf):
    lbl_dir = REPO / ds / "labels" / "test"
    tf_path = REPO / tf
    if not lbl_dir.exists() or not tf_path.exists() or not COCO.exists():
        pytest.skip(f"{ds} or its inputs not present (data is gitignored)")

    coco = json.loads(COCO.read_text())
    transforms = json.loads(tf_path.read_text())
    files = sorted(lbl_dir.glob("*.txt"))[:SAMPLE]
    assert files, f"{lbl_dir} has no labels"

    stale = []
    for f in files:
        got = [tuple([int(p[0])] + [float(v) for v in p[1:]])
               for p in (l.split() for l in f.read_text().split("\n") if l.strip())]
        want = _expected(f.stem, coco, transforms)
        if len(got) != len(want):
            stale.append(f"{f.stem}: {len(got)} labels on disk vs {len(want)} expected")
            continue
        for (gc, *g), (wc, *w) in zip(got, want):
            if gc != wc or any(abs(a - b) > TOL for a, b in zip(g, w)):
                stale.append(
                    f"{f.stem}: label {g} (class {gc}) != expected "
                    f"{[round(v, 6) for v in w]} (class {wc})")
                break

    assert not stale, (
        f"{ds} labels do not match {tf} — regenerate with "
        f"scripts/make_yolo_dataset.py, then re-run scripts/eval_detector.py:\n"
        + "\n".join(stale[:5]))


def test_known_stale_dataset_is_detected():
    """The original data/yolo_cleaned is the artifact that caused this.
    If it is still present and still stale, say so explicitly rather
    than letting it sit next to the rebuilt one looking equivalent."""
    lbl_dir = REPO / "data/yolo_cleaned/labels/test"
    tf_path = REPO / "data/transforms.json"
    if not lbl_dir.exists() or not tf_path.exists() or not COCO.exists():
        pytest.skip("data/yolo_cleaned not present")

    coco = json.loads(COCO.read_text())
    transforms = json.loads(tf_path.read_text())
    f = sorted(lbl_dir.glob("*.txt"))[0]
    got = [[float(v) for v in l.split()[1:5]]
           for l in f.read_text().split("\n") if l.strip()]
    want = [list(w[1:]) for w in _expected(f.stem, coco, transforms)]
    if len(got) == len(want) and all(
            abs(a - b) <= TOL for g, w in zip(got, want) for a, b in zip(g, w)):
        pytest.skip("data/yolo_cleaned has been regenerated — nothing to warn about")
    pytest.xfail(
        "data/yolo_cleaned is STALE (superseded by data/yolo_cleaned_rebuilt); "
        "it must not be used for eval_detector.py")
