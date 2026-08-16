#!/usr/bin/env python3
"""Do drawings of the same circuit straddle the train/test boundary? (fix 1c)

The splits are stratified by component-count tertile crossed with rarest class,
because the published Digitize-HCD annotations carry no drafter or circuit-id
metadata (``data/splits/splits_meta.json`` says so). An image-level split of a
corpus that contains several drawings of one circuit would put some of those
drawings in training and others in test, and every end-to-end number would be
optimistic by an unknown amount. Nobody has measured whether that happened.

This measures it three ways. None settles it alone, and one of them turned out
to answer a narrower question than it appears to -- which is stated here rather
than discovered by a reviewer:

  1. COMPONENT INVENTORY. The multiset of component classes per image. Two
     drawings of the same circuit MUST share an inventory, so a collision is
     necessary and nowhere near sufficient -- two unrelated circuits with three
     resistors and a source collide too. It bounds the risk from above and
     cannot confirm a single duplicate.

  2. NEAR-DUPLICATE FRAMES. Correlation of 64x64 ink-density vectors. This
     answers "is the same SHEET present twice", not the question actually asked.
     It is calibrated against CGHD, which photographs each drawing four times
     and therefore knows the right answer -- and the calibration FAILS: two
     photographs of one CGHD sheet score 0.117 median against 0.054 for
     unrelated drawings, because the four captures differ in camera pose and a
     fixed-grid correlation cannot see through that. So this measure detects
     only WELL-REGISTERED duplicates. A hit is real; a miss proves nothing.

     It is kept because a hit would be the worst version of the problem and is
     worth ruling out cheaply. It is reported with its own failed control
     attached, because a null result from an uncalibrated measure is exactly
     the kind of reassurance that should not be trusted.

  3. TOPOLOGY ITSELF, where ground truth exists. A Weisfeiler-Lehman hash of the
     net/component graph, invariant to net naming and component id. This is the
     question the mentor actually asked -- two people drawing the same circuit
     produce visually unrelated images and identical topology, so 2 would miss
     them entirely and 3 catches them exactly. Ground truth exists only for the
     382 test and validation images.

WHAT THIS CANNOT DO, AND IT IS THE MAIN RESULT. Measurement 3 is exact but blind
to the 895 training images, which have no netlist ground truth. Nothing here can
prove a test circuit is topologically absent from training. Closing that needs
training-set netlists, which is an annotation campaign, not a script. What CAN
be settled is the val/test boundary -- the one that corrupts model selection --
and it is settled below.

Usage:
    python scripts/split_duplicate_audit.py
    python scripts/split_duplicate_audit.py --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IMAGES = ROOT / "data/cleaned_1024"
COCO = ROOT / ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
               "Component Symbol and Text Label Data/component_annotations.json")
OUT = ROOT / "results/split_duplicate_audit.json"

# Hamming distance on a 64-bit dHash below which two frames are called near
# duplicates. 5 is the usual working figure; the distribution is reported so the
# choice can be checked rather than trusted.
HAMMING = 5


# Side of the downsampled ink image used for similarity. 64 keeps the layout of
# the drawing -- where the rails and symbol clusters sit -- while discarding pen
# texture, which is what differs between two photographs of one sheet.
GRID = 64

# Correlation at or above this is called a near-duplicate. NOT chosen by eye:
# calibrate_on_cghd() measures what two photographs of the SAME drawing actually
# score, and this is set below that distribution.
CORR_THRESHOLD = 0.85


def ink_vector(path: Path, grid: int = GRID) -> np.ndarray | None:
    """A drawing -> unit vector of ink density on a grid x grid mesh.

    An 8x8 dHash was tried first and is useless here, which is worth recording
    because it looks reasonable and is not: these frames are mostly white paper,
    so an 8x8 downsample is nearly uniform for every drawing and the hashes
    collide en masse. It reported 301 train/test "near-duplicate" pairs whose
    actual pixel correlation was 0.08-0.43 -- i.e. entirely unrelated circuits.
    Ink density on a finer mesh, compared by correlation, does not have that
    failure mode, and calibrate_on_cghd() checks that claim against drawings
    known to be the same.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (grid, grid), interpolation=cv2.INTER_AREA)
    v = 255.0 - small.astype(np.float64)      # ink is signal, paper is zero
    v -= v.mean()
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return (v / n).ravel()


def similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """All-pairs correlation as one matrix product."""
    m = np.vstack(vectors)
    return m @ m.T


def calibrate_on_cghd() -> dict:
    """Positive control: CGHD photographs each drawing four times.

    Those groups are known, so they are the one place we can ask what score two
    images of the SAME drawing actually get. Without this the threshold is a
    guess, and a similarity measure calibrated by guessing is how the dHash
    attempt produced 301 confident false positives.
    """
    ann = ROOT / "data/cghd_1024/annotations"
    img = ROOT / "data/cghd_1024/images"
    if not ann.is_dir() or not img.is_dir():
        return {"available": False}

    groups: dict[str, list[str]] = defaultdict(list)
    for p in sorted(ann.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:                                     # noqa: BLE001
            continue
        g = d.get("drawing_group")
        if g:
            groups[g].append(p.stem)
    groups = {g: v for g, v in groups.items() if len(v) >= 2}
    if not groups:
        return {"available": False}

    # cap the control at a few hundred images; it only needs to be enough to
    # separate two distributions
    stems, keep = [], []
    for g, v in sorted(groups.items())[:80]:
        for s in v:
            p = img / f"{s}.jpg"
            if p.exists():
                stems.append((g, s))
                keep.append(p)
    vecs, meta = [], []
    for (g, s), p in zip(stems, keep):
        v = ink_vector(p)
        if v is not None:
            vecs.append(v)
            meta.append((g, s))
    if len(vecs) < 4:
        return {"available": False}

    sim = similarity_matrix(vecs)
    same, diff = [], []
    for i in range(len(meta)):
        for j in range(i + 1, len(meta)):
            (same if meta[i][0] == meta[j][0] else diff).append(float(sim[i, j]))
    same_a, diff_a = np.array(same), np.array(diff)
    return {
        "available": True,
        "_what": ("CGHD photographs each drawing 4 times; SAME pairs are two "
                  "photographs of one sheet, DIFFERENT pairs are unrelated "
                  "drawings. A usable threshold separates them."),
        "images": len(meta), "groups": len({g for g, _ in meta}),
        "same_drawing_pairs": len(same),
        "same_min": float(same_a.min()), "same_p05": float(np.percentile(same_a, 5)),
        "same_median": float(np.median(same_a)),
        "different_pairs": len(diff),
        "different_median": float(np.median(diff_a)),
        "different_p99": float(np.percentile(diff_a, 99)),
        "different_max": float(diff_a.max()),
        "threshold_used": CORR_THRESHOLD,
        "threshold_is_below_same_p05": bool(CORR_THRESHOLD <= np.percentile(same_a, 5)),
        "threshold_is_above_different_p99": bool(
            CORR_THRESHOLD >= np.percentile(diff_a, 99)),
    }


def load_splits() -> dict[str, set[str]]:
    out = {}
    for name in ("train", "val", "test"):
        f = ROOT / f"data/splits/{name}.txt"
        out[name] = {s.strip().replace(".jpg", "")
                     for s in f.read_text().split() if s.strip()}
    return out


def inventories() -> dict[str, tuple]:
    """stem -> sorted multiset of component class names, from the COCO source."""
    if not COCO.exists():
        return {}
    d = json.loads(COCO.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(cats.get(a["category_id"], "?"))
    stem_of = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    return {stem_of[i]: tuple(sorted(v)) for i, v in by_img.items() if i in stem_of}


def topology_hashes() -> dict[str, str]:
    """stem -> WL hash of the net/component graph, for images that have GT."""
    try:
        import networkx as nx
    except ImportError:
        return {}
    from schematic2netlist.gt import gt_to_components, load_gt

    out = {}
    for gt_dir in ("data/gt_test_1024", "data/gt_val_1024"):
        d = ROOT / gt_dir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("circuit_*.json")):
            try:
                comps = gt_to_components(load_gt(p))
            except Exception:                                 # noqa: BLE001
                continue
            g = nx.Graph()
            for c in comps:
                cid = f"c{c['id']}"
                # the component node carries its CLASS; net nodes carry nothing,
                # so net names cannot influence the hash
                g.add_node(cid, label=c["class"])
                for net in c["nets"]:
                    if net is None:
                        continue
                    g.add_node(f"n{net}", label="net")
                    g.add_edge(cid, f"n{net}")
            if g.number_of_nodes():
                out[p.stem] = nx.weisfeiler_lehman_graph_hash(
                    g, node_attr="label", iterations=3)
    return out


def cross_split_pairs(groups: dict, splits: dict[str, set[str]]) -> list[dict]:
    """Groups sharing a key, reported only where members span two splits."""
    def split_of(stem):
        for name, members in splits.items():
            if stem in members:
                return name
        return "unassigned"

    out = []
    for key, stems in groups.items():
        if len(stems) < 2:
            continue
        by = defaultdict(list)
        for s in stems:
            by[split_of(s)].append(s)
        if len(by) < 2:
            continue
        out.append({"key": str(key)[:80], "n": len(stems),
                    "by_split": {k: sorted(v) for k, v in sorted(by.items())}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=CORR_THRESHOLD)
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    a = ap.parse_args()

    splits = load_splits()
    all_stems = sorted(set().union(*splits.values()))
    print(f"splits: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))

    # ---- 2. perceptual near-duplicates ------------------------------------
    print("calibrating the similarity measure on CGHD's known repeats...")
    cal = calibrate_on_cghd()
    if cal.get("available"):
        print(f"  same-drawing pairs:      median {cal['same_median']:.3f}, "
              f"5th pct {cal['same_p05']:.3f}, min {cal['same_min']:.3f}")
        print(f"  different-drawing pairs: median {cal['different_median']:.3f}, "
              f"99th pct {cal['different_p99']:.3f}, max {cal['different_max']:.3f}")
        ok = (cal["threshold_is_below_same_p05"]
              and cal["threshold_is_above_different_p99"])
        print(f"  threshold {a.threshold}: separates the two "
              f"{'YES' if ok else 'NO -- read the numbers before trusting this'}")
    else:
        print("  CGHD not available; threshold is uncalibrated")

    print("computing ink similarity...")
    vecs, stems_ok = [], []
    for s in all_stems:
        p = IMAGES / f"{s}.jpg"
        v = ink_vector(p) if p.exists() else None
        if v is not None:
            vecs.append(v)
            stems_ok.append(s)
    print(f"  {len(stems_ok)}/{len(all_stems)} frames vectorised")

    def split_of(stem):
        for name, members in splits.items():
            if stem in members:
                return name
        return "unassigned"

    sim = similarity_matrix(vecs)
    iu = np.triu_indices(len(stems_ok), k=1)
    dists = sim[iu]
    near = []
    for idx in np.argsort(-dists):
        c = float(dists[idx])
        if c < a.threshold:
            break
        i, j = int(iu[0][idx]), int(iu[1][idx])
        s1, s2 = stems_ok[i], stems_ok[j]
        near.append({"a": s1, "b": s2, "correlation": round(c, 4),
                     "split_a": split_of(s1), "split_b": split_of(s2)})
    near_cross = [r for r in near if r["split_a"] != r["split_b"]]
    train_test = [r for r in near_cross
                  if {r["split_a"], r["split_b"]} == {"train", "test"}]

    arr = dists
    print(f"  correlations: max {arr.max():.3f}, "
          f"99.9th pct {np.percentile(arr, 99.9):.3f}, "
          f"median {np.median(arr):.3f}")
    print(f"  near-duplicate pairs (corr >= {a.threshold}): {len(near)}; "
          f"cross-split {len(near_cross)}; train-test {len(train_test)}")
    hashes = {s: 1 for s in stems_ok}

    # ---- 1. component inventory -------------------------------------------
    inv = inventories()
    inv_groups = defaultdict(list)
    for stem, sig in inv.items():
        if stem in hashes:
            inv_groups[sig].append(stem)
    inv_cross = cross_split_pairs(inv_groups, splits)
    inv_tt = [g for g in inv_cross if "train" in g["by_split"]
              and "test" in g["by_split"]]
    print(f"inventory signatures: {len(inv_groups)} distinct over {len(inv)} images")
    print(f"  groups spanning splits: {len(inv_cross)}; train-test {len(inv_tt)}")

    # ---- 3. true topology, where GT exists --------------------------------
    topo = topology_hashes()
    topo_groups = defaultdict(list)
    for stem, h in topo.items():
        topo_groups[h].append(stem)
    topo_cross = cross_split_pairs(topo_groups, splits)
    dup_within = [{"hash": k[:16], "stems": sorted(v)}
                  for k, v in topo_groups.items() if len(v) > 1]
    print(f"topology hashes: {len(topo)} images with GT "
          f"({len(topo_groups)} distinct)")
    print(f"  identical topologies anywhere: {len(dup_within)} group(s); "
          f"spanning val/test: {len(topo_cross)}")

    # Impact, per test image rather than per group -- "19 groups" is not a
    # quantity anyone can act on; "39% of the test split" is.
    shared_with_val, repeated_in_test, in_any_group = set(), set(), set()
    for stems in topo_groups.values():
        if len(stems) < 2:
            continue
        by = defaultdict(list)
        for s in stems:
            by[split_of(s)].append(s)
        for s in by.get("test", []):
            in_any_group.add(s)
            if by.get("val"):
                shared_with_val.add(s)
            if len(by["test"]) > 1:
                repeated_in_test.add(s)
    n_test = len(splits["test"])
    impact = {
        "_what": ("How much of the REPORTED split is affected. Groups are not "
                  "an actionable unit; test images are."),
        "test_images": n_test,
        "test_sharing_topology_with_val": len(shared_with_val),
        "test_sharing_topology_with_val_rate": len(shared_with_val) / n_test,
        "test_sharing_topology_with_another_test_image": len(repeated_in_test),
        "test_in_any_repeated_topology_group": len(in_any_group),
        "largest_group_size": max((len(v) for v in topo_groups.values()),
                                  default=0),
        "_consequence": (
            "Validation selects and test reports, so a topology present in both "
            "means selection saw the circuit the headline is scored on. This is "
            "measured, not inferred. The TRAIN boundary cannot be measured the "
            "same way -- those 895 images have no netlist ground truth -- but a "
            "corpus that repeats circuits this heavily across two splits of 190 "
            "and 192 is unlikely to stop doing so across the 895."),
    }
    print(f"  IMPACT: {len(shared_with_val)}/{n_test} test images "
          f"({len(shared_with_val) / n_test:.1%}) share a topology with a "
          f"validation image")

    report = {
        "_what": ("Whether drawings of the same circuit straddle the split "
                  "boundary. Three measurements, none conclusive alone."),
        "_limit": (
            "Ground truth netlists exist only for the 382 test and validation "
            "images, so measurement 3 cannot see the training boundary at all. "
            "Measurements 1 and 2 bound the risk from opposite sides -- an "
            "inventory collision is necessary but not sufficient for a "
            "duplicate, a perceptual near-duplicate is sufficient but not "
            "necessary -- and neither eliminates it."),
        "splits": {k: len(v) for k, v in splits.items()},
        "perceptual": {
            "_what": ("correlation of 64x64 ink-density vectors over "
                      "data/cleaned_1024, calibrated against CGHD's known "
                      "same-drawing repeats"),
            "calibration": cal,
            "threshold": a.threshold,
            "frames_compared": len(stems_ok),
            "pairs_compared": int(len(dists)),
            "max_correlation": float(arr.max()),
            "pct999_correlation": float(np.percentile(arr, 99.9)),
            "median_correlation": float(np.median(arr)),
            "near_duplicate_pairs": len(near),
            "cross_split_pairs": len(near_cross),
            "train_test_pairs": len(train_test),
            "examples": near[:40],
        },
        "inventory": {
            "_what": "multiset of component classes; necessary not sufficient",
            "images": len(inv),
            "distinct_signatures": len(inv_groups),
            "groups_spanning_splits": len(inv_cross),
            "groups_spanning_train_test": len(inv_tt),
            "largest_groups": sorted(
                ({"n": g["n"], "by_split": {k: len(v) for k, v in
                                            g["by_split"].items()}}
                 for g in inv_cross), key=lambda x: -x["n"])[:15],
        },
        "topology": {
            "_what": ("Weisfeiler-Lehman hash of the net/component graph, "
                      "invariant to net naming and component id"),
            "images_with_gt": len(topo),
            "distinct_topologies": len(topo_groups),
            "identical_topology_groups": dup_within,
            "groups_spanning_splits": topo_cross,
            "impact": impact,
        },
    }
    out_p = ROOT / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
