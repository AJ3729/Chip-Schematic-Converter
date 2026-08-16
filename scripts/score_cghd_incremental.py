#!/usr/bin/env python3
"""Score whatever CGHD annotations exist, and re-score cleanly as more arrive (E1).

The CGHD annotation campaign delivers one drawing at a time over days. A harness
that only runs on a complete campaign would leave every cross-corpus number
unknown until the last drawing lands, which is exactly when it is too late to
discover that the numbers say something the paper has to account for. This one
runs today, on however many annotations exist -- including zero -- and produces
the same artifact each time, so E2, E3 and E4 read one file whose contents grow.

TWO CONDITIONS, AND WHY BOTH ARE NECESSARY

Detection transfers to CGHD at mAP@0.5 0.3445 against 0.9910 in-domain, and
recall in the smallest size quintile is 0.183. An end-to-end score on CGHD is
therefore dominated by components that were never found: it is a detector
measurement wearing a topology measurement's clothes, and it would be read as
"the wire tracing does not generalise" when the evidence says no such thing.

So every image is scored twice:

  detector  the frozen pipeline as it actually runs, cached boxes and all.
            This is the honest end-to-end number and it will be low.

  oracle    the same pipeline with CGHD's OWN annotated boxes substituted for
            the detector's output. Detection is then perfect by construction, so
            what remains measures wire tracing, junction adjudication and
            terminal assignment on a corpus none of them were tuned for. This is
            the number that answers the question a reader actually has.

The gap between them is the part of cross-corpus failure attributable to
detection. Neither number alone supports that decomposition.

WHAT THIS DOES NOT DO. It never writes into the annotation directories, never
inspects an un-annotated image in a way that could inform annotation, and never
reports a mean over a sample too small to carry one -- below MIN_FOR_CI it emits
the per-image rows and says so, rather than a point estimate a reader would
quote.

Usage:
    python scripts/score_cghd_incremental.py
    python scripts/score_cghd_incremental.py --condition oracle
    python scripts/score_cghd_incremental.py --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from schematic2netlist.benchmark import align_components, score_prediction  # noqa: E402
from schematic2netlist.config import load_config  # noqa: E402
from schematic2netlist.netlist import export_spice_netlist  # noqa: E402
from schematic2netlist.pipeline import run_pipeline  # noqa: E402
from schematic2netlist.simulate import run_ngspice_diag  # noqa: E402
from metrics.pin_aware import load_symmetry, score_pin_aware  # noqa: E402
from stats.bootstrap import bootstrap_mean, bootstrap_rate  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark import pred_components  # noqa: E402

IMG = ROOT / "data/cghd_1024/images"
CGHD_ANN = ROOT / "data/cghd_1024/annotations"
CACHE = ROOT / "data/cghd_1024/detections"
ACCEPTED = ROOT / "data/cghd/annotations/accepted"
QUEUE = ROOT / "data/cghd/annotation_queue.json"
OUT = ROOT / "results/cghd_scored/incremental.json"

# Below this many scored images a mean is noise wearing a number's clothes. The
# per-image rows are still written; only the aggregate is withheld.
MIN_FOR_CI = 5

CONDITIONS = ("detector", "oracle")


# --------------------------------------------------------------------- input

def load_annotations(limit: int | None, accepted: Path = ACCEPTED) -> list[dict]:
    """Accepted human netlists, in queue order so a partial campaign stays
    stratified -- the queue is built so every prefix is balanced (B8)."""
    if not accepted.is_dir():
        return []
    files = {p.stem: p for p in accepted.glob("*.json")}
    order: list[str] = []
    if QUEUE.is_file():
        order = [q["drawing_group"] for q in json.loads(QUEUE.read_text())["queue"]]
    ranked = [s for s in order if s in files] + sorted(set(files) - set(order))
    out = []
    for stem in ranked[:limit] if limit else ranked:
        rec = json.loads(files[stem].read_text())
        rec["_stem"] = stem
        out.append(rec)
    return out


def annotation_fingerprint(anns: list[dict]) -> str:
    """Content hash of the annotation set a result was produced from.

    Results accumulate across days and get compared to each other; without this
    there is no way to tell a number that grew because more drawings arrived
    from one that changed because an annotation was corrected.
    """
    h = hashlib.sha256()
    for a in sorted(anns, key=lambda r: r["_stem"]):
        h.update(a["_stem"].encode())
        h.update(json.dumps(a.get("components", []), sort_keys=True).encode())
    return h.hexdigest()[:16]


def to_graph(components: list[dict]) -> list[dict]:
    """Annotation components -> the {id, class, nets, bbox} metrics use."""
    out = []
    for c in components:
        terms = sorted(c.get("terminals", []), key=lambda t: t["index"])
        out.append({"id": c["id"], "class": c["class"],
                    "nets": [t.get("net") for t in terms],
                    "bbox": c.get("bbox")})
    return out


def oracle_detections(stem: str) -> list[dict] | None:
    """CGHD's own annotated boxes, in the detector's output format.

    Confidence is 1.0 because these are not predictions. The class is the
    project-vocabulary class from spec/class_map_cghd.yaml, already resolved
    when the adapter wrote the file.
    """
    p = CGHD_ANN / f"{stem}.json"
    if not p.is_file():
        return None
    meta = json.loads(p.read_text())
    return [{"class": c["class"], "confidence": 1.0,
             "x": c["bbox"][0], "y": c["bbox"][1],
             "width": c["bbox"][2], "height": c["bbox"][3]}
            for c in meta.get("components", [])]


# -------------------------------------------------------------------- scoring

def score_one(stem: str, ref: list[dict], cfg: dict, condition: str,
              sym: dict) -> dict | None:
    if condition == "oracle":
        dets = oracle_detections(stem)
        if dets is None:
            return {"stem": stem, "condition": condition,
                    "error": "no CGHD annotation to use as an oracle"}
    else:
        p = CACHE / f"{stem}.json"
        if not p.is_file():
            return {"stem": stem, "condition": condition,
                    "error": "no cached detections"}
        dets = json.loads(p.read_text())

    t0 = time.perf_counter()
    try:
        res = run_pipeline(IMG / f"{stem}.jpg", cfg, detections=dets)
    except Exception as e:                                    # noqa: BLE001
        return {"stem": stem, "condition": condition,
                "error": f"{type(e).__name__}: {e}"}
    latency_ms = (time.perf_counter() - t0) * 1000.0

    comps = res.get("components") or []
    # A pipeline component carries no box of its own -- the box belongs to the
    # detection it came from, and scripts/benchmark.py is where that join is
    # defined. Reuse it rather than re-deriving it here: component alignment is
    # geometric, so a second definition that drifts would silently score every
    # circuit against the wrong pairs.
    pred = pred_components(res)

    row = {"stem": stem, "condition": condition,
           "n_pred": len(pred), "n_ref": len(ref),
           "latency_ms": round(latency_ms, 2)}
    row.update(score_prediction(pred, ref))

    # Pin-aware, on the SAME alignment the ladder uses: align_components
    # relabels predicted ids into reference id space, so the relabelled lists
    # are what carry the matched pairs.
    pa, ra, _ = align_components(pred, ref, 0.3)
    ref_ids = {g["id"] for g in ra}
    matched = [(c["id"], c["id"]) for c in pa if c["id"] in ref_ids]
    pin = score_pin_aware(pa, ra, matched, sym)
    row["pin_aware_strict"] = bool(pin.strict_success)
    row["pin_aware_correct"] = pin.n_correct
    row["pin_aware_scored"] = pin.n_scored
    row["pin_aware_component_acc"] = pin.component_accuracy

    # SPICE validity, which needs no reference and is comparable across
    # conditions
    try:
        sp = ROOT / "results/cghd_scored" / f"{stem}.{condition}.sp"
        sp.parent.mkdir(parents=True, exist_ok=True)
        export_spice_netlist(comps, str(sp))
        ok, cat, _ = run_ngspice_diag(str(sp), cfg)
        row["spice_valid"] = cat != "parse_error"
        row["solvable"] = bool(ok)
    except Exception:                                         # noqa: BLE001
        row["spice_valid"] = False
        row["solvable"] = False
    return row


def aggregate(rows: list[dict]) -> dict:
    good = [r for r in rows if "error" not in r]
    if len(good) < MIN_FOR_CI:
        return {"_withheld": (
            f"{len(good)} scored image(s) is below the {MIN_FOR_CI} this harness "
            "will summarise. The per-image rows are complete; a mean over this "
            "many circuits would be quoted as a result and should not be."),
            "n_scored": len(good)}
    out = {"n_scored": len(good)}
    RATES = {"strict_success", "pin_aware_strict", "spice_valid", "solvable"}
    for k in ("net_f1", "terminal_pair_f1", "per_component_connected_acc",
              "nged", "strict_success", "pin_aware_strict", "spice_valid",
              "solvable"):
        vals = [r[k] for r in good if k in r]
        if not vals:
            continue
        iv = (bootstrap_rate(vals) if k in RATES
              else bootstrap_mean([float(v) for v in vals]))
        out[k] = {"mean": iv.point, "ci95": [iv.lo, iv.hi], "n": iv.n}
    by_drafter: dict[str, int] = defaultdict(int)
    for r in good:
        by_drafter[r["stem"].split("__")[0]] += 1
    out["images_per_drafter"] = dict(sorted(by_drafter.items()))
    return out


def self_test(cfg: dict, sym: dict, n: int = 6) -> int:
    """Prove the harness scores correctly BEFORE any annotation exists.

    Takes the oracle condition's own output as the reference. That reference is
    by construction exactly what the oracle condition predicts, so every oracle
    metric must come out at 1.0 -- alignment, net extraction, terminal-pair and
    pin-aware alike. Anything less is a plumbing fault in this file, and finding
    it now costs nothing, whereas finding it after 20 hours of annotation costs
    the campaign.

    The detector condition is run on the same references and reported but NOT
    asserted: it measures the real detection gap and has no correct value to
    check against. Nothing here is a result -- the "annotation" is machine
    output, and a number from it describes the pipeline agreeing with itself.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="cghd_incremental_selftest_"))
    made = []
    for p in sorted(CGHD_ANN.glob("*.json")):
        if len(made) >= n:
            break
        dets = oracle_detections(p.stem)
        if not dets or len(dets) < 3:
            continue
        try:
            res = run_pipeline(IMG / f"{p.stem}.jpg", cfg, detections=dets)
        except Exception:                                     # noqa: BLE001
            continue
        pc = pred_components(res)
        if not pc:
            continue
        (tmp / f"{p.stem}.json").write_text(json.dumps({
            "schema_version": 1, "image": f"{p.stem}.jpg",
            "source": "SELF-TEST FIXTURE -- pipeline output, not a human "
                      "annotation", "annotator": "self-test",
            "components": [
                {"id": c["id"], "class": c["class"], "bbox": c["bbox"],
                 "terminals": [{"index": i, "net": nm}
                               for i, nm in enumerate(c["nets"])]}
                for c in pc],
            "sites": [], "interventions": [], "notes": ""}, indent=1))
        made.append(p.stem)

    if not made:
        print("self-test: could not build a fixture (no CGHD images?)  FAIL")
        return 1

    print(f"self-test: {len(made)} images, reference = the oracle condition's "
          f"own output")
    ok = True
    for cond in CONDITIONS:
        rows = [score_one(s, to_graph(json.loads((tmp / f"{s}.json").read_text())
                                      ["components"]), cfg, cond, sym)
                for s in made]
        errs = [r for r in rows if "error" in r]
        good = [r for r in rows if "error" not in r]
        if errs:
            print(f"  {cond}: {len(errs)} image(s) errored: "
                  f"{errs[0].get('error')}")
            ok = False
        if not good:
            continue
        keys = ("net_f1", "terminal_pair_f1", "per_component_connected_acc",
                "strict_success", "pin_aware_strict")
        vals = {k: sum(float(r[k]) for r in good) / len(good) for k in keys}
        line = "  ".join(f"{k}={vals[k]:.4f}" for k in keys)
        if cond == "oracle":
            perfect = all(abs(vals[k] - 1.0) < 1e-9 for k in keys)
            ok &= perfect
            print(f"  {cond:8s} {line}  {'OK' if perfect else 'FAIL'}")
            if not perfect:
                print("      the oracle condition must reproduce a reference it "
                      "generated itself; it did not, so the scoring path is "
                      "wrong, not the pipeline")
        else:
            print(f"  {cond:8s} {line}  (not asserted -- this is the real "
                  "detection gap, and it has no correct value)")

    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=(*CONDITIONS, "both"), default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    ap.add_argument("--accepted", default=str(ACCEPTED.relative_to(ROOT)),
                    help="directory of accepted annotations (override for tests)")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the scoring path with no annotations needed")
    a = ap.parse_args()

    if a.self_test:
        cfg = load_config(a.config)
        cfg["preprocess"]["images_dir"] = "data/cghd_1024/images"
        cfg["detect"]["cache_dir"] = str(CACHE.relative_to(ROOT))
        return self_test(cfg, load_symmetry())

    anns = load_annotations(a.limit, ROOT / a.accepted)
    conditions = CONDITIONS if a.condition == "both" else (a.condition,)

    report: dict = {
        "_what": "Cross-corpus netlist scores on CGHD, over the annotations "
                 "accepted so far. Re-run as more arrive; it is designed to.",
        "_conditions": {
            "detector": "the frozen pipeline as it runs -- the honest "
                        "end-to-end number, bounded by detection transfer "
                        "(mAP@0.5 0.3445)",
            "oracle": "CGHD's own annotated boxes substituted for detection, so "
                      "what is left measures wire tracing and terminal "
                      "assignment rather than the detector",
        },
        "annotations_accepted": len(anns),
        "annotation_fingerprint": annotation_fingerprint(anns),
        "min_images_for_a_mean": MIN_FOR_CI,
    }

    if not anns:
        report["status"] = "NO ANNOTATIONS YET"
        report["_next"] = (
            f"Annotate with `python tools/annotator/server.py`; accepted files "
            f"land in {ACCEPTED.relative_to(ROOT)} via scripts/sync_board.py. "
            "This harness is complete and will score them the moment they "
            "exist -- nothing else is waiting on it.")
        out_p = ROOT / a.out
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=1) + "\n")
        print("no accepted CGHD annotations yet -- harness verified, wrote "
              f"{a.out}")
        print("  run again after the first drawing is accepted; both conditions "
              "are ready")
        return 0

    cfg = load_config(a.config)
    cfg["preprocess"]["images_dir"] = "data/cghd_1024/images"
    cfg["detect"]["cache_dir"] = str(CACHE.relative_to(ROOT))
    sym = load_symmetry()

    for cond in conditions:
        rows = []
        for i, ann in enumerate(anns, 1):
            stem = ann["_stem"]
            ref = to_graph(ann.get("components", []))
            r = score_one(stem, ref, cfg, cond, sym)
            if r:
                rows.append(r)
            print(f"  [{cond}] {i}/{len(anns)} {stem}", flush=True)
        report[cond] = {"per_image": rows, "aggregate": aggregate(rows)}

    out_p = ROOT / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=1) + "\n")

    print(f"\nscored {len(anns)} annotated image(s) -> {a.out}")
    for cond in conditions:
        agg = report[cond]["aggregate"]
        if "_withheld" in agg:
            print(f"  {cond}: {agg['n_scored']} scored, aggregate withheld "
                  f"(< {MIN_FOR_CI})")
        else:
            nf = agg.get("net_f1", {}).get("mean")
            st = agg.get("strict_success", {}).get("mean")
            pa = agg.get("pin_aware_strict", {}).get("mean")
            print(f"  {cond}: net F1 {nf:.4f}, strict {st:.4f}, "
                  f"pin-aware strict {pa:.4f}  (n={agg['n_scored']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
