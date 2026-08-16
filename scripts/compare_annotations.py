#!/usr/bin/env python3
"""Compare two INDEPENDENT connectivity annotations of the same circuits.

Input is two ground-truth directories in the schema of ``src/schematic2netlist/gt.py``
-- ours (A) and an external second annotator's (B) -- plus their ``decisions/``
site records. Output is an inter-annotator agreement report and a per-disagreement
CSV for human adjudication.

WHAT IT MEASURES, AND WHY EACH ONE IS SEPARATE
----------------------------------------------
1. NET-PARTITION AGREEMENT. The Hungarian net-F1 already used by the benchmark
   (``metrics.net_level_metrics``), with annotation B passed in the "prediction"
   slot. Net NAMES are arbitrary and are never compared; only the grouping is.

2. PER-SITE DECISION AGREEMENT. From ``decisions/<stem>.json``, over the four
   calls the annotation records: junction | crossing | none | edge-group. This is
   where the judgement actually happened, so it is where agreement is most
   informative -- and it supports a kappa, which the net metrics do not.

3. PIN ORDER ON 3+-TERMINAL COMPONENTS, SCORED SEPARATELY AND ON PURPOSE.
   Nothing else in this repository can see a pin swap. Net grouping is unchanged
   by one, the six-rule ERC still passes, and -- decisively --
   ``benchmark.canonicalize_terminals`` REORDERS every component's terminals by a
   connectivity signature before scoring, precisely so that arbitrary indexing
   cannot penalise a correct circuit. That is right for prediction-vs-GT and it
   makes the metric structurally blind here: swap a BJT's base and collector and
   the net-F1 stays exactly 1.000. The first pass reported pin order as its
   dominant error mode. So pin order is compared on the RAW terminal order, after
   translating net names through a pin-order-INVARIANT net correspondence, and
   reported as its own number. ``--self-test`` demonstrates the blindness rather
   than asserting it: it injects pin swaps and shows net-F1 unmoved at 1.000.

4. COMPONENT COUNT AND CLASS. Compared through a CLASS-AGNOSTIC geometric
   pairing, deliberately not through ``benchmark.align_components``, which only
   matches within a class -- under that alignment one component both annotators
   drew a box around but labelled differently appears as one missing component
   plus one extra component, i.e. two clerical errors instead of the single class
   disagreement it is.

THE TAXONOMY, AND THE RULE ABOUT GUESSING
-----------------------------------------
Every disagreement is labelled ``clerical``, ``ambiguous-ink``, ``convention``,
``genuine-error-A``, ``genuine-error-B`` or ``unresolved``. Only two categories
are ever proposed automatically, because only two are derivable from the data:

  clerical      a component present in one annotation and absent in the other.
  ambiguous-ink a junction/crossing flip at a site that has NEITHER a solder dot
                NOR a hop -- a plain X, where the annotation guide itself says
                both readings are defensible. The dot and hop are not asserted
                from anyone's prose: they are re-measured from the photograph by
                the same tracer that produced the site indices
                (``scripts/gt_val_tools/trace.py``), using its own thresholds
                (dot_score < 2.30, hop_score < 6.0), and the call is only made at
                degree >= 4. A degree-3 T is excluded: a T has no crossing
                reading, so a flip there is a substantive disagreement, not
                ambiguous ink.

``convention``, ``genuine-error-A`` and ``genuine-error-B`` are NEVER proposed
automatically. Each asserts something the files cannot establish -- which of two
readers is right, or that a difference is stylistic -- and a wrong auto-label is
worse than no label, because it is the label a tired adjudicator accepts. They
exist as values for a human to write into the ``resolution`` column.

AGREEMENT IS REPORTED BEFORE ADJUDICATION. Every headline number counts every
disagreement, including the ones later resolved as clerical. That is the number a
reviewer is asking for; agreement computed after fixing the things you agreed
were wrong is not an agreement measurement.

Usage:
    python scripts/compare_annotations.py --gt-b <second annotator's dir>
    python scripts/compare_annotations.py --gt-b <dir> --stems results/blind_review/manifest.csv
    python scripts/compare_annotations.py --self-test     # no annotator needed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from schematic2netlist.benchmark import (
    align_components,
    canonicalize_terminals,
    iou_center,
)
from schematic2netlist.classes import canonical_class, class_terminals
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import net_level_metrics, terminal_pair_metrics

ROOT = Path(__file__).resolve().parent.parent

SITE_CALLS = ("junction", "crossing", "none", "edge-group")

# The tracer's own decision thresholds, reused so "no dot / no hop" means exactly
# what it meant when the sites were first adjudicated.
DOT_RATIO = 2.30
HOP_THRESH = 6.0

# Radius, in 1024-frame pixels, within which a coordinate an annotator wrote down
# is taken to name a traced site. See resolve_sites_xy for why a match must also
# be UNAMBIGUOUS, and why this number is 12.
SITE_XY_TOL_PX = 12

CATEGORIES = ("clerical", "ambiguous-ink", "convention",
              "genuine-error-A", "genuine-error-B", "unresolved")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def components_of(gt: dict) -> list[dict]:
    """GT dict -> metrics graph format, keeping bbox (which gt_to_components drops).

    Terminal-index -> net ordering comes from ``gt.gt_to_components`` so this file
    does not re-implement it; bbox is needed because component alignment is
    geometric.
    """
    comps = gt_to_components(gt)
    by_id = {c["id"]: c for c in gt["components"]}
    for c in comps:
        c["class"] = canonical_class(c["class"])
        c["bbox"] = by_id[c["id"]].get("bbox")
    return comps


def load_side(gt_dir: Path, stem: str) -> tuple[dict | None, dict | None]:
    gt_path = gt_dir / f"{stem}.json"
    gt = load_gt(gt_path) if gt_path.is_file() else None
    dec_path = gt_dir / "decisions" / f"{stem}.json"
    dec = json.loads(dec_path.read_text()) if dec_path.is_file() else None
    return gt, dec


def normalize_call(value) -> str:
    """A decision value -> one of the four calls. A list is an explicit edge
    grouping, which the schema uses where a site is too complex for a single
    call (see scripts/gt_verification_stats.py)."""
    if isinstance(value, list):
        return "edge-group"
    v = str(value).strip().lower()
    return v if v in SITE_CALLS else f"other:{v}"


def resolve_sites_xy(dec: dict | None, site_evidence: dict | None,
                     tol: float = SITE_XY_TOL_PX,
                     other_calls: dict | None = None) -> tuple[dict, dict]:
    """Coordinate-keyed site record -> the index-keyed one this differ compares.

    WHY THIS EXISTS. A site index is not a property of the drawing, it is a
    property of an annotation: ``gt_val_tools/trace.py`` erases the annotator's
    component boxes before it skeletonises the ink, so which intersections exist
    and what order they are numbered in both follow from where that annotator put
    the boxes. Handing annotation A's indices to an independent second annotator
    would therefore leak A's component enumeration -- which is exactly what
    measure 4 is trying to compare -- and asking the second annotator to invent
    their own indices produces two numberings that do not correspond, so every
    site lands in ``site_coverage`` and the kappa is computed over nothing.

    A pixel coordinate is a fact about the photograph instead of a fact about an
    annotation, so it is the one currency both passes can quote independently.
    Annotator B writes ``{"sites_xy": [{"xy": [434, 869], "call": "crossing"}]}``
    and this resolves each entry to the site A's tracer found there.

    TWO REFUSALS, BOTH DELIBERATE. A coordinate is dropped as unresolved rather
    than guessed when (a) several traced sites sit within ``tol`` of it and the
    choice between them could change the comparison, or (b) two coordinates land
    on the same site. In both cases the call cannot be attached to a specific
    piece of ink, and a mis-attached call is worse than a missing one: it enters
    the kappa as a real disagreement about the wrong intersection.

    THE CLUSTER EXCEPTION. The tracer routinely splits one drawn intersection
    into two or three sites a few pixels apart -- the schema has an entire value
    (``edge-group``) for "a drawn crossing split across two nearby sites". A
    reader looking at the drawing sees one intersection there and writes one
    coordinate, so refusing every cluster would discard exactly the sites the
    drawing is hardest at. When ``other_calls`` is supplied, a cluster is
    resolved to the NEAREST site whenever every site in it that the other
    annotator adjudicated carries the SAME call: the choice is then provably
    unable to change the comparison, so making it is bookkeeping rather than
    interpretation. A cluster whose calls disagree is still refused, because
    there the choice decides the answer.

    ``tol`` is 12 px in the 1024 frame. The tracer merges branch pixels within 6
    px into one site, so anything under ~6 px cannot separate two sites anyway;
    12 px allows for a human pointing at a junction by eye while staying well
    below the spacing at which two intersections are visually distinct.

    Returns ``(sites_by_index, report)``. ``report`` always carries the counts,
    so an unresolved coordinate is visible in the output rather than silently
    absent.
    """
    entries = ((dec or {}).get("sites_xy") or [])
    report = {"given": len(entries), "matched": 0, "unmatched": [],
              "matched_via_cluster": 0, "tolerance_px": tol,
              "evidence_available": site_evidence is not None}
    if not entries:
        return {}, report
    if site_evidence is None:
        report["unmatched"] = [{"xy": e.get("xy"), "why": "no tracer evidence"}
                               for e in entries]
        return {}, report

    resolved: dict[str, object] = {}
    claimed: dict[int, list] = defaultdict(list)
    for e in entries:
        xy = e.get("xy")
        call = e.get("call")
        if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
            report["unmatched"].append({"xy": xy, "why": "malformed xy"})
            continue
        x, y = float(xy[0]), float(xy[1])
        near = sorted(
            ((math.hypot(s["x"] - x, s["y"] - y), i)
             for i, s in site_evidence.items()),
            key=lambda t: (t[0], t[1]))
        within = [(d, i) for d, i in near if d <= tol]
        if not within:
            nd = near[0][0] if near else float("inf")
            report["unmatched"].append(
                {"xy": [x, y], "call": call,
                 "why": f"no traced site within {tol} px (nearest {nd:.1f} px)"})
            continue
        if len(within) > 1:
            # Only the candidates the other annotator actually adjudicated can
            # change the comparison; the rest are uncompared either way.
            adjudicated = {normalize_call((other_calls or {})[str(i)])
                           for _, i in within if str(i) in (other_calls or {})}
            if len(adjudicated) > 1:
                report["unmatched"].append(
                    {"xy": [x, y], "call": call,
                     "why": f"{len(within)} traced sites within {tol} px "
                            f"(indices {[i for _, i in within]}) carry different "
                            f"calls {sorted(adjudicated)} -- the choice between "
                            "them would decide the answer"})
                continue
            report["matched_via_cluster"] += 1
        claimed[within[0][1]].append((x, y, call))

    for idx, hits in claimed.items():
        if len(hits) > 1:
            for x, y, call in hits:
                report["unmatched"].append(
                    {"xy": [x, y], "call": call,
                     "why": f"{len(hits)} coordinates resolve to site {idx} -- "
                            "cannot tell which call belongs to it"})
            continue
        resolved[str(idx)] = hits[0][2]
        report["matched"] += 1
    return resolved, report


# ---------------------------------------------------------------------------
# component pairing
# ---------------------------------------------------------------------------

def geometric_pairs(a: list[dict], b: list[dict], iou_threshold: float = 0.3
                    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Class-AGNOSTIC Hungarian pairing of A boxes to B boxes.

    Separate from ``align_components`` on purpose: that one is class-aware (it
    must be -- it scores a detector), and a class-aware pairing cannot represent
    "same box, different class", which is exactly the disagreement we most want to
    surface. Returns (pairs, unmatched_a_idx, unmatched_b_idx).
    """
    if not a or not b:
        return [], list(range(len(a))), list(range(len(b)))
    cost = np.ones((len(a), len(b)))
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            if ca.get("bbox") and cb.get("bbox"):
                cost[i, j] = 1.0 - iou_center(ca["bbox"], cb["bbox"])
    rows, cols = linear_sum_assignment(cost)
    pairs, used_a, used_b = [], set(), set()
    for r, c in zip(rows, cols):
        iou = 1.0 - cost[r, c]
        if iou >= iou_threshold:
            pairs.append((int(r), int(c), float(iou)))
            used_a.add(int(r))
            used_b.add(int(c))
    return (pairs,
            [i for i in range(len(a)) if i not in used_a],
            [j for j in range(len(b)) if j not in used_b])


def net_correspondence(a: list[dict], b: list[dict], id_of_a: dict, id_of_b: dict
                       ) -> dict[str, str]:
    """Map A's net names onto B's, INVARIANTLY of pin order.

    A net's signature is the multiset of component ids it touches -- component
    ids, never (component, terminal) pairs. Using terminal identity would make the
    correspondence itself move when pin order moves, and the pin-order comparison
    would then be comparing a quantity against a version of itself: a swap could
    cancel out and read as agreement. Hungarian maximises total signature overlap.
    """
    def sig(comps, id_map):
        out: dict[str, Counter] = defaultdict(Counter)
        for c in comps:
            cid = id_map.get(c["id"], ("orphan", c["id"]))
            for net in c["nets"]:
                if net is not None:
                    out[net][cid] += 1
        return out

    sa, sb = sig(a, id_of_a), sig(b, id_of_b)
    if not sa or not sb:
        return {}
    an, bn = sorted(sa), sorted(sb)
    cost = np.zeros((len(an), len(bn)))
    for i, p in enumerate(an):
        for j, q in enumerate(bn):
            cost[i, j] = -sum((sa[p] & sb[q]).values())
    rows, cols = linear_sum_assignment(cost)
    return {an[r]: bn[c] for r, c in zip(rows, cols) if cost[r, c] < 0}


# ---------------------------------------------------------------------------
# per-circuit comparison
# ---------------------------------------------------------------------------

def compare_circuit(stem: str, gt_a: dict, gt_b: dict,
                    dec_a: dict | None, dec_b: dict | None,
                    site_evidence: dict | None) -> tuple[dict, list[dict]]:
    a = components_of(gt_a)
    b = components_of(gt_b)
    rows: list[dict] = []

    def row(kind, detail, category, evidence="", **extra):
        rows.append({"stem": stem, "kind": kind, "detail": detail,
                     "proposed_category": category, "evidence": evidence,
                     "resolution": "", "resolution_note": "", **extra})

    # --- components: count and class ---------------------------------------
    pairs, only_a, only_b = geometric_pairs(a, b)

    # An external annotator working in the ORIGINAL photograph frame (~2000 px)
    # instead of the cleaned_1024 frame the GT bboxes live in produces boxes that
    # overlap nothing, so every component reads as "missing in B" plus "extra in
    # B". The resulting report is catastrophically wrong and looks entirely
    # plausible -- a big pile of clerical errors. Catch it as what it is.
    frame_mismatch = bool(a and b and not pairs)
    if frame_mismatch:
        scale = ""
        if a[0].get("bbox") and b[0].get("bbox"):
            ha = max(c["bbox"][3] for c in a if c.get("bbox"))
            hb = max(c["bbox"][3] for c in b if c.get("bbox"))
            scale = f"; largest box height A {ha:.0f} px vs B {hb:.0f} px"
        row("frame_mismatch_suspected",
            f"{len(a)} A components and {len(b)} B components, but NOT ONE pair "
            f"overlaps at IoU>=0.3{scale}",
            "unresolved",
            "almost certainly a coordinate-frame difference (GT bboxes are in the "
            "cleaned_1024 frame, see data/README.md), not 100% disagreement. "
            "Re-project B before reading any number for this circuit",
            component_a="", component_b="")
    for i in only_a:
        row("component_missing_in_B",
            f"A#{a[i]['id']} {a[i]['class']} at {a[i].get('bbox')} has no counterpart in B",
            "clerical", "no B box overlaps it at IoU>=0.3; a component present in "
            "one annotation and absent in the other is an inventory slip",
            component_a=a[i]["id"], component_b="")
    for j in only_b:
        row("component_extra_in_B",
            f"B#{b[j]['id']} {b[j]['class']} at {b[j].get('bbox')} has no counterpart in A",
            "clerical", "no A box overlaps it at IoU>=0.3; a component present in "
            "one annotation and absent in the other is an inventory slip",
            component_a="", component_b=b[j]["id"])
    class_mismatch = 0
    for i, j, iou in pairs:
        if a[i]["class"] != b[j]["class"]:
            class_mismatch += 1
            row("component_class",
                f"same box (IoU {iou:.2f}): A calls it {a[i]['class']}, "
                f"B calls it {b[j]['class']}",
                "unresolved",
                "which reading of the drawn symbol is right is not derivable "
                "from the annotation files",
                component_a=a[i]["id"], component_b=b[j]["id"])

    # --- net partition: the benchmark's own Hungarian net-F1 ----------------
    # B goes in the "prediction" slot. align_components is class-aware, which is
    # what we want for the METRIC (it is the metric's own definition), even though
    # the class-disagreement report above needed the class-agnostic pairing.
    b_al, a_al, stats_ba = align_components(b, a)
    net_ba = net_level_metrics(canonicalize_terminals(b_al),
                               canonicalize_terminals(a_al))
    tp_ba = terminal_pair_metrics(canonicalize_terminals(b_al),
                                  canonicalize_terminals(a_al))
    # and the same thing with the sides swapped, to expose the asymmetry of a
    # scorer written for prediction-vs-GT (see the report in comparison.json).
    a_al2, b_al2, stats_ab = align_components(a, b)
    net_ab = net_level_metrics(canonicalize_terminals(a_al2),
                               canonicalize_terminals(b_al2))

    # --- pin order on 3+-terminal components --------------------------------
    # Uses the RAW terminal order (never canonicalize_terminals, which would erase
    # exactly what is being measured), with net names translated through the
    # pin-order-invariant correspondence.
    id_of_a = {c["id"]: ("p", k) for k, c in enumerate(a)}
    id_of_b = {}
    for i, j, _ in pairs:
        id_of_b[b[j]["id"]] = id_of_a[a[i]["id"]]
    for j in only_b:
        id_of_b[b[j]["id"]] = ("b_only", b[j]["id"])
    corr = net_correspondence(a, b, id_of_a, id_of_b)

    a_by_id = {c["id"]: c for c in a}
    b_by_id = {c["id"]: c for c in b}
    pin_total = pin_agree = pin_perm = 0
    net_assign_mismatch = 0
    for i, j, _ in pairs:
        ca, cb = a[i], b[j]
        mapped = [corr.get(n) if n is not None else None for n in ca["nets"]]
        multiset_same = (sorted(x or "" for x in mapped)
                         == sorted(x or "" for x in cb["nets"]))
        if not multiset_same:
            net_assign_mismatch += 1
            row("net_assignment",
                f"A#{ca['id']} {ca['class']}: A nets {ca['nets']} -> {mapped} "
                f"vs B nets {cb['nets']} (different set of nets, not a reorder)",
                "unresolved",
                "which grouping the ink supports cannot be settled from the "
                "files; needs the photograph",
                component_a=ca["id"], component_b=cb["id"])
        n_term = max(class_terminals(ca["class"]), class_terminals(cb["class"]))
        if n_term < 3 or len(ca["nets"]) < 3 or len(cb["nets"]) < 3:
            continue
        pin_total += 1
        if mapped == list(cb["nets"]):
            pin_agree += 1
            continue
        pure_perm = multiset_same
        pin_perm += int(pure_perm)
        row("pin_order",
            f"A#{ca['id']} {ca['class']}: terminal order A {ca['nets']} -> {mapped} "
            f"vs B {cb['nets']}"
            + (" (same nets, different pins -- a pure pin swap)" if pure_perm else ""),
            "unresolved",
            "pin identity is read from the arrowhead / +- glyphs / gate bar in the "
            "photograph; no automatic check can settle it, and the net metrics "
            "cannot even see it",
            component_a=ca["id"], component_b=cb["id"])

    # --- site decisions -----------------------------------------------------
    # Either side may record sites by coordinate instead of by index; an index is
    # a fact about an annotation, a coordinate is a fact about the photograph.
    # Resolve first, then merge, so the comparison below is identical either way.
    raw_a = (dec_a or {}).get("sites", {}) or {}
    raw_b = (dec_b or {}).get("sites", {}) or {}
    xy_a, xyrep_a = resolve_sites_xy(dec_a, site_evidence, other_calls=raw_b)
    xy_b, xyrep_b = resolve_sites_xy(dec_b, site_evidence, other_calls=raw_a)
    sites_a = {**raw_a, **xy_a}
    sites_b = {**raw_b, **xy_b}

    for side, rep in (("A", xyrep_a), ("B", xyrep_b)):
        for u in rep["unmatched"]:
            row("site_xy_unresolved",
                f"{side} recorded a call at {u.get('xy')} that names no site: "
                f"{u['why']}",
                "unresolved",
                "the call cannot be attached to a specific intersection, so it is "
                "excluded from the agreement rather than matched to nearby ink; "
                "re-check the coordinate against the 1024 frame",
                site="")
    # A missing decision RECORD is one fact about the delivery, not one
    # disagreement per site: emitting a row per site would let a single
    # undelivered file dominate the CSV and drown the substantive rows.
    missing_record = None
    if dec_a is None or dec_b is None:
        missing_record = "A" if dec_a is None else "B"
        row("decisions_missing", f"no decisions/{stem}.json from {missing_record} "
            f"({len(sites_a) or len(sites_b)} sites recorded by the other side "
            "are therefore uncompared)",
            "unresolved",
            "site-level agreement cannot be measured for this circuit; ask the "
            "annotator for the record rather than adjudicating anything",
            site="")
        sites_a = sites_b = {}
    shared = sorted(set(sites_a) & set(sites_b), key=lambda s: (len(s), s))
    site_pairs, site_agree = [], 0
    for key in shared:
        va, vb = normalize_call(sites_a[key]), normalize_call(sites_b[key])
        site_pairs.append((va, vb))
        if va == vb:
            site_agree += 1
            continue
        ev = (site_evidence or {}).get(int(key)) if str(key).isdigit() else None
        category, why = "unresolved", ""
        if ev is None:
            why = ("no re-measured ink evidence for this site "
                   "(tracer unavailable or site indices did not reconcile)")
        elif {va, vb} == {"junction", "crossing"}:
            if ev["degree"] < 4:
                why = (f"degree {ev['degree']} site (a T, not a crossing): a T has "
                       "no crossing reading, so this is substantive")
            elif ev["dot_score"] >= DOT_RATIO:
                why = (f"a solder dot IS present (dot_score {ev['dot_score']} >= "
                       f"{DOT_RATIO}), so the ink is not ambiguous")
            elif ev["hop_score"] >= HOP_THRESH:
                why = (f"a hop IS present (hop_score {ev['hop_score']} >= "
                       f"{HOP_THRESH}), so the ink is not ambiguous")
            else:
                category = "ambiguous-ink"
                why = (f"plain X at ({ev['x']},{ev['y']}) degree {ev['degree']}: "
                       f"dot_score {ev['dot_score']} < {DOT_RATIO} and hop_score "
                       f"{ev['hop_score']} < {HOP_THRESH} -- neither a dot nor a "
                       "hop, so both readings are defensible")
        else:
            why = (f"{va}/{vb} is not a junction-crossing flip; the ambiguous-ink "
                   "rule does not apply")
        row("site_decision", f"site {key}: A={va} B={vb}", category, why,
            site=key)
    for key in sorted(set(sites_a) ^ set(sites_b), key=lambda s: (len(s), s)):
        side = "A" if key in sites_a else "B"
        row("site_coverage",
            f"site {key} adjudicated only by {side} "
            f"(={normalize_call((sites_a if side == 'A' else sites_b)[key])})",
            "unresolved",
            "the two annotations enumerate different sites; may be a tracer/config "
            "difference rather than a reading difference",
            site=key)

    summary = {
        "stem": stem,
        "components_a": len(a), "components_b": len(b),
        "component_pairs": len(pairs),
        "components_only_in_a": len(only_a), "components_only_in_b": len(only_b),
        "component_count_agrees": len(a) == len(b),
        "class_mismatches": class_mismatch,
        "net_f1": net_ba["f1"],
        "net_precision": net_ba["precision"], "net_recall": net_ba["recall"],
        "net_partition_exact": net_ba["f1"] == 1.0,
        "terminal_pair_f1": tp_ba["f1"],
        "net_f1_reversed": net_ab["f1"],
        "align_matched_b_as_pred": stats_ba["matched"],
        "align_matched_a_as_pred": stats_ab["matched"],
        "pin_order_components": pin_total,
        "pin_order_agree": pin_agree,
        "pin_order_pure_swaps": pin_perm,
        "net_assignment_mismatches": net_assign_mismatch,
        "sites_shared": len(shared),
        "sites_agree": site_agree,
        "sites_only_in_a": len(set(sites_a) - set(sites_b)),
        "sites_only_in_b": len(set(sites_b) - set(sites_a)),
        "sites_xy_given_a": xyrep_a["given"],
        "sites_xy_given_b": xyrep_b["given"],
        "sites_xy_matched_a": xyrep_a["matched"],
        "sites_xy_matched_b": xyrep_b["matched"],
        "sites_xy_matched_via_cluster": (xyrep_a["matched_via_cluster"]
                                         + xyrep_b["matched_via_cluster"]),
        "sites_xy_unresolved": len(xyrep_a["unmatched"]) + len(xyrep_b["unmatched"]),
        "site_decision_record_missing": missing_record,
        "frame_mismatch_suspected": frame_mismatch,
        "site_evidence_available": site_evidence is not None,
        "disagreements": len(rows),
        # net_f1 is computed through align_components, which matches WITHIN a
        # class. So a class disagreement or a count disagreement depresses net_f1
        # even when both annotators grouped the terminals identically. When this
        # flag is true, read the components block before reading net_f1.
        "net_f1_depressed_by_component_differences": bool(
            net_ba["f1"] < 1.0 and net_assign_mismatch == 0
            and (class_mismatch or only_a or only_b)),
    }
    return summary, rows, site_pairs


# ---------------------------------------------------------------------------
# site ink evidence, re-measured from the photograph
# ---------------------------------------------------------------------------

def measure_sites(stem: str, gt_a: dict, images_dir: Path,
                  dec_a: dict | None, reasons: dict | None = None) -> dict | None:
    """Per-site degree / dot_score / hop_score, from the tracer that made the sites.

    Returns None -- meaning "no evidence", which forces every site disagreement to
    ``unresolved`` -- rather than returning something approximate. Two guards:
    the tracer must produce enough sites to cover the recorded indices, and where
    annotation A's notes quote a site coordinate (``S12 (434,869)``) it must
    match. If the enumeration has drifted, indices mean different sites in the two
    files and any evidence attached to them would be evidence about the wrong ink.

    The drift guard is deliberately all-or-nothing. A renumbering that moves two
    sites has moved every site after them, and there is no way to tell from the
    file which of the unquoted indices those are -- so a circuit that fails it
    loses its site comparison entirely rather than keeping the majority that
    happen to still line up. ``reasons``, if given, receives ``stem -> why`` so
    the report can name the guard that fired instead of saying only "no evidence".
    """
    def fail(why: str) -> None:
        if reasons is not None:
            reasons[stem] = why
        return None

    tools = ROOT / "scripts" / "gt_val_tools"
    img = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            img = p
            break
    if img is None:
        return fail(f"no image for {stem} under {images_dir}")
    if not (tools / "trace.py").is_file():
        return fail("scripts/gt_val_tools/trace.py is not present")
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import trace as tracer  # noqa: PLC0415  (gt_val_tools is not a package)
        tr = tracer.trace(str(img), gt_a)
    except Exception as e:
        return fail(f"the tracer raised {type(e).__name__}: {e}")

    sites = tr.get("sites") or []
    recorded = [int(k) for k in ((dec_a or {}).get("sites") or {}) if str(k).isdigit()]
    if recorded and max(recorded) >= len(sites):
        return fail(f"A records site {max(recorded)} but the tracer found only "
                    f"{len(sites)} sites, so the enumerations do not correspond")

    ev = {i: {"x": s["x"], "y": s["y"], "degree": s["degree"],
              "dot_score": s["dot_score"], "hop_score": s["hop_score"],
              "kind": s["kind"]}
          for i, s in enumerate(sites)}

    # coordinate cross-check against A's own prose, where it quotes one
    import re
    quoted = re.findall(r"\bS(\d+)\s*\((\d+)\s*,\s*(\d+)\)", (dec_a or {}).get("notes", ""))
    bad = [int(si) for si, x, y in quoted
           if int(si) in ev and (abs(ev[int(si)]["x"] - int(x)) > 2
                                 or abs(ev[int(si)]["y"] - int(y)) > 2)]
    if quoted and bad:
        return fail(
            f"site numbering has drifted: {len(bad)} of {len(quoted)} coordinates "
            f"A quoted in its own notes (sites {sorted(bad)[:5]}) no longer sit "
            "where the tracer puts those indices, so every index in this circuit "
            "may name different ink than it did")
    return ev


# ---------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------

def cohens_kappa(pairs: list[tuple[str, str]]) -> dict:
    """Cohen's kappa over the pooled site decisions, plus the confusion matrix.

    Reported with the observed agreement beside it, because kappa is unstable
    when one call dominates -- and it does here: 'junction' is ~83% of the
    recorded calls, so a high raw agreement can sit next to a modest kappa and
    both are true. Undefined (pe == 1) is reported as null with a reason, never
    silently as 0 or 1.
    """
    if not pairs:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None,
                "kappa": None, "note": "no site adjudicated by both annotators",
                "confusion": {}}
    labels = sorted({v for pair in pairs for v in pair})
    n = len(pairs)
    obs = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    conf: dict[str, dict[str, int]] = {l: {m: 0 for m in labels} for l in labels}
    for x, y in pairs:
        conf[x][y] += 1
    if abs(1.0 - pe) < 1e-12:
        return {"n": n, "observed_agreement": obs, "expected_agreement": pe,
                "kappa": None,
                "note": ("kappa undefined: both annotators used a single call "
                         "almost exclusively, so chance agreement is 1.0"),
                "confusion": conf}
    return {"n": n, "observed_agreement": obs, "expected_agreement": pe,
            "kappa": (obs - pe) / (1 - pe), "note": "", "confusion": conf}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def read_stem_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return [r["stem"] for r in csv.DictReader(path.open())]
    return [Path(l.strip()).stem for l in path.read_text().splitlines() if l.strip()]


def run_comparison(gt_a_dir: Path, gt_b_dir: Path, stems: list[str],
                   images_dir: Path, use_evidence: bool) -> tuple[dict, list[dict]]:
    per_circuit, all_rows, all_site_pairs = [], [], []
    skipped = []
    no_evidence: dict[str, str] = {}
    for stem in stems:
        gt_a, dec_a = load_side(gt_a_dir, stem)
        gt_b, dec_b = load_side(gt_b_dir, stem)
        if gt_a is None or gt_b is None:
            skipped.append({"stem": stem,
                            "reason": "missing in A" if gt_a is None else "missing in B"})
            continue
        ev = (measure_sites(stem, gt_a, images_dir, dec_a, no_evidence)
              if use_evidence else None)
        summary, rows, site_pairs = compare_circuit(stem, gt_a, gt_b, dec_a, dec_b, ev)
        per_circuit.append(summary)
        all_rows.extend(rows)
        all_site_pairs.extend(site_pairs)

    n = len(per_circuit)
    cat = Counter(r["proposed_category"] for r in all_rows)
    kind = Counter(r["kind"] for r in all_rows)
    sites_shared = sum(c["sites_shared"] for c in per_circuit)
    sites_agree = sum(c["sites_agree"] for c in per_circuit)
    pin_total = sum(c["pin_order_components"] for c in per_circuit)
    pin_agree = sum(c["pin_order_agree"] for c in per_circuit)

    asym = [c["stem"] for c in per_circuit
            if abs(c["net_f1"] - c["net_f1_reversed"]) > 1e-9
            or c["align_matched_b_as_pred"] != c["align_matched_a_as_pred"]]

    report = {
        "annotation_a": str(gt_a_dir),
        "annotation_b": str(gt_b_dir),
        "circuits_compared": n,
        "circuits_skipped": skipped,
        "agreement_before_adjudication": {
            "_meaning": ("every disagreement is counted, including ones a human "
                         "will later resolve as clerical; this is the "
                         "inter-annotator number, not a post-hoc one"),
            "circuits_with_no_disagreement_at_all": sum(
                1 for c in per_circuit if c["disagreements"] == 0),
            "net_partition": {
                "mean_net_f1": (sum(c["net_f1"] for c in per_circuit) / n) if n else None,
                "circuits_exact": sum(1 for c in per_circuit if c["net_partition_exact"]),
                "circuits_exact_rate": (
                    sum(1 for c in per_circuit if c["net_partition_exact"]) / n) if n else None,
                "mean_terminal_pair_f1": (
                    sum(c["terminal_pair_f1"] for c in per_circuit) / n) if n else None,
                "circuits_where_net_f1_is_depressed_by_component_differences": [
                    c["stem"] for c in per_circuit
                    if c["net_f1_depressed_by_component_differences"]],
                "_read_this_first": (
                    "net_f1 runs through benchmark.align_components, which matches "
                    "components WITHIN a class. A class disagreement, or a "
                    "component one annotator drew and the other did not, therefore "
                    "pulls net_f1 below 1.0 even when the two annotators grouped "
                    "every terminal identically. For those circuits (listed above) "
                    "the net number is reporting a component disagreement, not a "
                    "connectivity one."),
            },
            "site_decisions": {
                "sites_adjudicated_by_both": sites_shared,
                "agree": sites_agree,
                "agreement_rate": (sites_agree / sites_shared) if sites_shared else None,
                "sites_only_in_a": sum(c["sites_only_in_a"] for c in per_circuit),
                "sites_only_in_b": sum(c["sites_only_in_b"] for c in per_circuit),
                "cohens_kappa": cohens_kappa(all_site_pairs),
                "coordinate_records": {
                    "_meaning": (
                        "calls recorded as {\"sites_xy\": [{\"xy\": [x, y], "
                        "\"call\": ...}]} in the 1024 frame and resolved to a "
                        "traced site here. A site index belongs to whoever drew "
                        "the component boxes, so an independent annotator quotes "
                        "coordinates instead; unresolved ones are excluded from "
                        "the kappa rather than matched to nearby ink"),
                    "tolerance_px": SITE_XY_TOL_PX,
                    "given_a": sum(c["sites_xy_given_a"] for c in per_circuit),
                    "given_b": sum(c["sites_xy_given_b"] for c in per_circuit),
                    "matched_a": sum(c["sites_xy_matched_a"] for c in per_circuit),
                    "matched_b": sum(c["sites_xy_matched_b"] for c in per_circuit),
                    "matched_via_cluster": sum(
                        c["sites_xy_matched_via_cluster"] for c in per_circuit),
                    "_cluster_meaning": (
                        "the tracer split one drawn intersection into several "
                        "sites a few px apart and the other annotator gave every "
                        "one of them the same call, so which one the coordinate "
                        "names cannot change the comparison"),
                    "unresolved": sum(c["sites_xy_unresolved"] for c in per_circuit),
                },
                "circuits_without_ink_evidence": {
                    "_meaning": (
                        "site calls in these circuits are excluded from the "
                        "agreement and the kappa above. Every other measure -- "
                        "nets, pin order, components -- is unaffected"),
                    "n": len(no_evidence),
                    "why": no_evidence,
                },
            },
            "pin_order_3plus_terminals": {
                "_meaning": ("scored separately because canonicalize_terminals "
                             "makes every other metric here blind to it"),
                "components_compared": pin_total,
                "agree": pin_agree,
                "agreement_rate": (pin_agree / pin_total) if pin_total else None,
                "pure_pin_swaps": sum(c["pin_order_pure_swaps"] for c in per_circuit),
            },
            "components": {
                "circuits_with_equal_count": sum(
                    1 for c in per_circuit if c["component_count_agrees"]),
                "components_only_in_a": sum(c["components_only_in_a"] for c in per_circuit),
                "components_only_in_b": sum(c["components_only_in_b"] for c in per_circuit),
                "class_mismatches": sum(c["class_mismatches"] for c in per_circuit),
            },
        },
        "disagreements": {
            "total": len(all_rows),
            "by_kind": dict(kind),
            "by_proposed_category": {c: cat.get(c, 0) for c in CATEGORIES},
            "auto_categorised": sum(cat.get(c, 0) for c in ("clerical", "ambiguous-ink")),
            "left_for_human_adjudication": cat.get("unresolved", 0),
            "_policy": ("only 'clerical' (component present on one side only) and "
                        "'ambiguous-ink' (junction/crossing flip at a re-measured "
                        "plain X: no dot, no hop, degree>=4) are proposed "
                        "automatically. 'convention', 'genuine-error-A' and "
                        "'genuine-error-B' assert who is right or that a "
                        "difference is stylistic, neither of which the files can "
                        "establish, so they are left for a human."),
        },
        "metric_asymmetry": {
            "_why": ("benchmark.py was written for prediction-vs-GT and is not "
                     "symmetric; see the notes in comparison.json's "
                     "known_asymmetries"),
            "circuits_where_swapping_the_sides_changes_the_result": asym,
            "known_asymmetries": [
                "align_components() gives unmatched PREDICTION components fresh "
                "disjoint ids but leaves unmatched GT components with their own, "
                "so an extra component in B and an extra in A are not scored the "
                "same way; this file therefore reports net_f1 in both directions.",
                "metrics._prf() returns precision 1.0 when the prediction side has "
                "no terminals at all, so an EMPTY annotation B scores perfect "
                "precision; only F1 and the exact-match rate are safe headline "
                "numbers here.",
                "benchmark.score_prediction()'s strict_success requires "
                "unmatched_gt == 0 but ignores unmatched_pred, so B may invent "
                "components and still be 'strict'. It is not used in this report.",
                "benchmark.canonicalize_terminals() reorders terminals by "
                "connectivity signature, which is correct for prediction-vs-GT and "
                "makes net-level metrics structurally blind to pin order; pin "
                "order is measured on raw terminal order instead.",
                "metrics.per_component_recall_accuracy() is recall-only by design "
                "and would read a B that merges every net as perfect; not used.",
            ],
        },
        "per_circuit": per_circuit,
    }
    return report, all_rows


CSV_COLS = ["stem", "kind", "detail", "proposed_category", "evidence",
            "component_a", "component_b", "site", "resolution", "resolution_note"]


def write_outputs(report: dict, rows: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLS})


# ---------------------------------------------------------------------------
# self-test: a differ nobody has tested is worthless
# ---------------------------------------------------------------------------

def self_test(gt_a_dir: Path, images_dir: Path, seed: int, n_flips: int,
              n_swaps: int, n_stems: int, out_dir: Path | None = None) -> int:
    """Synthesise a perturbed copy of A, run the differ, demand EXACT recovery.

    Injects three disagreements whose ground truth we know by construction, then
    checks the recovered set is equal to the injected set -- not a superset, not a
    subset. A differ that finds the injected errors plus four phantoms is as
    useless as one that finds none, so both directions are asserted.

    Also runs the null case first: A against an unperturbed copy of itself must
    produce exactly zero disagreements. Without it, "recovered 6 of 6" could be
    produced by a differ that flags everything.

    The perturbed copy is made in a temp directory. Nothing under data/ is
    written. Artifacts go to ``out_dir`` (default results/blind_review/selftest/)
    and are marked synthetic so they can never be mistaken for a real second
    annotation.
    """
    rng = random.Random(seed)
    stems = sorted(p.stem for p in gt_a_dir.glob("circuit_*.json"))
    tmp = Path(tempfile.mkdtemp(prefix="blind_review_selftest_"))
    gt_b_dir = tmp / "gt_b"
    (gt_b_dir / "decisions").mkdir(parents=True)

    # --- null case: an identical copy must produce zero disagreements -------
    null_report, null_rows = run_comparison(gt_a_dir, gt_a_dir, stems, images_dir, False)
    null_ok = not null_rows
    print(f"null case: A vs an identical copy of A over {len(stems)} circuits -> "
          f"{len(null_rows)} disagreements  {'OK' if null_ok else 'FAIL'}")
    if not null_ok:
        for r in null_rows[:8]:
            print(f"      {r['stem']} {r['kind']}: {r['detail'][:100]}")

    # pick stems that can actually carry each perturbation
    flip_pool, swap_pool = [], []
    for stem in stems:
        gt, dec = load_side(gt_a_dir, stem)
        if dec and any(normalize_call(v) in ("junction", "crossing")
                       for v in (dec.get("sites") or {}).values()):
            flip_pool.append(stem)
        for c in gt["components"]:
            if class_terminals(canonical_class(c["class"])) >= 3:
                nets = [t["net"] for t in sorted(c["terminals"], key=lambda t: t["index"])]
                if len(nets) >= 3 and len({n for n in nets if n}) >= 2:
                    swap_pool.append(stem)
                    break

    flip_stems = rng.sample(flip_pool, min(n_flips, len(flip_pool)))
    swap_stems = rng.sample([s for s in swap_pool if s not in flip_stems],
                            min(n_swaps, len([s for s in swap_pool if s not in flip_stems])))
    rest = [s for s in stems if s not in flip_stems and s not in swap_stems]
    drop_stem = rng.choice(rest)
    subset = sorted(set(flip_stems + swap_stems + [drop_stem]
                        + rng.sample(rest, min(n_stems, len(rest)))))

    # deliver half the flip stems by coordinate, half by index
    xy_stems = set(sorted(flip_stems)[: max(1, len(flip_stems) // 2)])
    xy_delivered: dict[str, int] = {}

    injected_flips, injected_swaps, injected_drops = set(), set(), set()
    for stem in subset:
        gt, dec = load_side(gt_a_dir, stem)
        gt = json.loads(json.dumps(gt))
        dec = json.loads(json.dumps(dec)) if dec else None

        if stem in flip_stems and dec:
            keys = [k for k, v in dec["sites"].items()
                    if normalize_call(v) in ("junction", "crossing")]
            k = rng.choice(sorted(keys, key=lambda s: (len(s), s)))
            dec["sites"][k] = ("crossing" if normalize_call(dec["sites"][k]) == "junction"
                               else "junction")
            injected_flips.add((stem, str(k)))

        if stem in swap_stems:
            cands = [c for c in gt["components"]
                     if class_terminals(canonical_class(c["class"])) >= 3
                     and len(c["terminals"]) >= 3]
            cands = [c for c in cands
                     if len({t["net"] for t in c["terminals"] if t["net"]}) >= 2]
            comp = rng.choice(cands)
            terms = sorted(comp["terminals"], key=lambda t: t["index"])
            # swap two terminals that are genuinely on different nets, otherwise
            # the "perturbation" is a no-op and the test proves nothing
            i, j = next(((x, y) for x in range(len(terms)) for y in range(x + 1, len(terms))
                         if terms[x]["net"] != terms[y]["net"]))
            terms[i]["net"], terms[j]["net"] = terms[j]["net"], terms[i]["net"]
            injected_swaps.add((stem, comp["id"]))

        if stem == drop_stem:
            victim = gt["components"].pop(rng.randrange(len(gt["components"])))
            injected_drops.add((stem, victim["id"]))

        # Half the flip stems are delivered the way a real second annotator has
        # to deliver them -- by coordinate, not by site index, because the index
        # belongs to whoever drew the boxes. Recovery must be identical either
        # way; if it is not, the coordinate path is silently losing calls.
        if stem in xy_stems and dec is not None and dec.get("sites"):
            ev = measure_sites(stem, gt, images_dir, dec)
            if ev is not None:
                xy_entries = []
                for k, v in list(dec["sites"].items()):
                    if not str(k).isdigit() or int(k) not in ev:
                        continue
                    s = ev[int(k)]
                    xy_entries.append({"xy": [s["x"], s["y"]], "call": v})
                    del dec["sites"][k]
                if xy_entries:
                    dec["sites_xy"] = xy_entries
                    xy_delivered[stem] = len(xy_entries)

        (gt_b_dir / f"{stem}.json").write_text(json.dumps(gt, indent=2))
        if dec is not None:
            (gt_b_dir / "decisions" / f"{stem}.json").write_text(json.dumps(dec, indent=2))

    report, rows = run_comparison(gt_a_dir, gt_b_dir, subset, images_dir, True)

    rec_flips = {(r["stem"], str(r["site"])) for r in rows if r["kind"] == "site_decision"}
    rec_swaps = {(r["stem"], r["component_a"]) for r in rows if r["kind"] == "pin_order"}
    rec_drops = {(r["stem"], r["component_a"])
                 for r in rows if r["kind"] == "component_missing_in_B"}
    other = [r for r in rows if r["kind"] not in
             ("site_decision", "pin_order", "component_missing_in_B",
              "site_xy_unresolved")]

    print(f"\nself-test: {len(subset)} circuits, "
          f"{len(injected_flips)} site flips, {len(injected_swaps)} pin swaps, "
          f"{len(injected_drops)} dropped component")
    ok = True
    for label, inj, rec in (("site flips", injected_flips, rec_flips),
                            ("pin swaps", injected_swaps, rec_swaps),
                            ("dropped components", injected_drops, rec_drops)):
        missed, phantom = inj - rec, rec - inj
        good = not missed and not phantom
        ok &= good
        print(f"  {label:20s} injected {len(inj):3d}  recovered {len(rec):3d}  "
              f"missed {len(missed)}  phantom {len(phantom)}  "
              f"{'OK' if good else 'FAIL'}")
        if missed:
            print(f"      missed:  {sorted(missed)[:6]}")
        if phantom:
            print(f"      phantom: {sorted(phantom)[:6]}")

    # categories must be right, not merely present
    cats = Counter(r["proposed_category"] for r in rows if r["kind"] == "component_missing_in_B")
    drop_ok = cats.get("clerical", 0) == len(injected_drops) and len(cats) <= 1
    ok &= drop_ok
    print(f"  {'dropped -> clerical':20s} {cats.get('clerical', 0)}/{len(injected_drops)}"
          f"  {'OK' if drop_ok else 'FAIL'}")
    pin_cats = Counter(r["proposed_category"] for r in rows if r["kind"] == "pin_order")
    pin_ok = pin_cats.get("unresolved", 0) == len(rec_swaps) and len(pin_cats) <= 1
    ok &= pin_ok
    print(f"  {'pin swap -> unresolved':20s} {pin_cats.get('unresolved', 0)}/{len(rec_swaps)}"
          f"  {'OK' if pin_ok else 'FAIL'}")
    site_cats = Counter(r["proposed_category"] for r in rows if r["kind"] == "site_decision")
    print(f"  site flip categories: {dict(site_cats)} "
          f"(ambiguous-ink only where the re-measured ink shows no dot and no hop)")
    bad_site_cat = set(site_cats) - {"ambiguous-ink", "unresolved"}
    ok &= not bad_site_cat
    if bad_site_cat:
        print(f"      FAIL: site flips categorised {bad_site_cat}, which this tool "
              "must never propose automatically")

    # coordinate delivery must be exactly as good as index delivery
    xy_flips = {(s, k) for s, k in injected_flips if s in xy_delivered}
    xy_rec = {(s, k) for s, k in rec_flips if s in xy_delivered}
    cr = report["agreement_before_adjudication"]["site_decisions"]["coordinate_records"]
    xy_ok = (bool(xy_delivered) and xy_flips == xy_rec and cr["unresolved"] == 0
             and cr["matched_b"] == sum(xy_delivered.values()))
    ok &= xy_ok
    print(f"  {'sites by coordinate':20s} {len(xy_delivered)} circuit(s), "
          f"{cr['matched_b']}/{cr['given_b']} coordinates resolved, "
          f"{len(xy_rec)}/{len(xy_flips)} flips recovered  "
          f"{'OK' if xy_ok else 'FAIL'}")
    if not xy_ok:
        print(f"      the coordinate path is not equivalent to the index path; "
              f"unresolved {cr['unresolved']}, missed {sorted(xy_flips - xy_rec)[:4]}")

    if other:
        print(f"  {'phantom other kinds':20s} {len(other)}  FAIL")
        for r in other[:6]:
            print(f"      {r['stem']} {r['kind']}: {r['detail'][:90]}")
        ok = False
    else:
        print(f"  {'phantom other kinds':20s} 0  OK")

    # the point of scoring pin order separately: prove the net metrics cannot see it
    swap_only = [c for c in report["per_circuit"]
                 if c["stem"] in swap_stems and c["stem"] != drop_stem
                 and c["stem"] not in flip_stems]
    blind = [c["stem"] for c in swap_only if c["net_f1"] == 1.0]
    print(f"  net-F1 on pin-swap-only circuits: {len(blind)}/{len(swap_only)} still "
          f"exactly 1.000 -- the net metric is blind to a pin swap, which is why "
          f"pin order is scored on its own")
    ok &= (len(blind) == len(swap_only))
    ok &= null_ok

    out_dir = out_dir or (ROOT / "results/blind_review/selftest")
    report["_SYNTHETIC"] = (
        "THIS IS NOT A REAL SECOND ANNOTATION. Annotation B here is a perturbed "
        "copy of A generated by scripts/compare_annotations.py --self-test, to "
        "validate the differ before any annotator exists. The numbers below are "
        "properties of the injected perturbation, not of anyone's annotation.")
    report["_selftest"] = {
        "seed": seed,
        "null_case_disagreements": len(null_rows),
        "null_case_circuits": len(stems),
        "injected": {"site_flips": sorted(injected_flips),
                     "pin_swaps": sorted(injected_swaps),
                     "dropped_components": sorted(injected_drops)},
        "recovered": {"site_flips": sorted(rec_flips),
                      "pin_swaps": sorted(rec_swaps),
                      "dropped_components": sorted(rec_drops)},
        "phantom_other_kinds": len(other),
        "net_f1_unmoved_by_pin_swap": f"{len(blind)}/{len(swap_only)}",
        "passed": bool(ok),
    }
    write_outputs(report, rows, out_dir / "comparison.json",
                  out_dir / "disagreements.csv")
    shown = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir
    print(f"  artifacts -> {shown}/ (marked synthetic)")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-a", default="data/gt_test_1024",
                    help="our annotation (never modified)")
    ap.add_argument("--gt-b", help="the second annotator's directory")
    ap.add_argument("--stems", default=None,
                    help="manifest.csv or a .txt of stems; default: everything "
                         "present in BOTH directories")
    ap.add_argument("--images-dir", default="data/cleaned_1024",
                    help="frame the GT bboxes live in; used to re-measure site ink")
    ap.add_argument("--out-json", default="results/blind_review/comparison.json")
    ap.add_argument("--out-csv", default="results/blind_review/disagreements.csv")
    ap.add_argument("--no-site-evidence", action="store_true",
                    help="skip re-measuring dot/hop from the photographs; every "
                         "site disagreement then stays 'unresolved'")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the differ against synthetic perturbations "
                         "of A; needs no second annotator")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--flips", type=int, default=6)
    ap.add_argument("--swaps", type=int, default=4)
    ap.add_argument("--extra-stems", type=int, default=10)
    ap.add_argument("--selftest-out", default="results/blind_review/selftest")
    args = ap.parse_args()

    gt_a_dir = ROOT / args.gt_a
    images_dir = ROOT / args.images_dir

    if args.self_test:
        return self_test(gt_a_dir, images_dir, args.seed,
                         args.flips, args.swaps, args.extra_stems,
                         ROOT / args.selftest_out)

    if not args.gt_b:
        ap.error("--gt-b is required (or use --self-test)")
    gt_b_dir = ROOT / args.gt_b
    if not gt_b_dir.is_dir():
        raise SystemExit(f"no such annotation directory: {gt_b_dir}")

    if args.stems:
        stems = read_stem_list(ROOT / args.stems)
    else:
        stems = sorted({p.stem for p in gt_a_dir.glob("circuit_*.json")}
                       & {p.stem for p in gt_b_dir.glob("circuit_*.json")})
    if not stems:
        raise SystemExit("no circuits in common between the two annotations")

    report, rows = run_comparison(gt_a_dir, gt_b_dir, stems, images_dir,
                                  not args.no_site_evidence)
    write_outputs(report, rows, ROOT / args.out_json, ROOT / args.out_csv)

    ag = report["agreement_before_adjudication"]
    print(f"compared {report['circuits_compared']} circuits "
          f"({len(report['circuits_skipped'])} skipped)")
    n_frame = sum(1 for c in report["per_circuit"] if c["frame_mismatch_suspected"])
    if n_frame:
        print("!" * 78)
        print(f"! {n_frame}/{report['circuits_compared']} circuits have NO overlapping "
              "component box at all.")
        print("! That is a coordinate-frame difference, not a disagreement. GT boxes "
              "are in the")
        print("! cleaned_1024 frame (data/README.md). Re-project annotation B and "
              "re-run; every")
        print("! number below is meaningless until you do.")
        print("!" * 78)
    print("AGREEMENT BEFORE ADJUDICATION")
    print(f"  net partition   mean F1 {ag['net_partition']['mean_net_f1']:.4f}, "
          f"exact on {ag['net_partition']['circuits_exact']}/{report['circuits_compared']}")
    sd = ag["site_decisions"]
    k = sd["cohens_kappa"]["kappa"]
    print(f"  site decisions  {sd['agree']}/{sd['sites_adjudicated_by_both']} "
          f"({(sd['agreement_rate'] or 0):.4f}), Cohen's kappa "
          f"{'undefined' if k is None else f'{k:.4f}'}")
    po = ag["pin_order_3plus_terminals"]
    print(f"  pin order       {po['agree']}/{po['components_compared']} "
          f"({(po['agreement_rate'] or 0):.4f}) on 3+-terminal parts, "
          f"{po['pure_pin_swaps']} pure swaps")
    print(f"  disagreements   {report['disagreements']['total']} "
          f"({report['disagreements']['left_for_human_adjudication']} left for a human)")
    print(f"wrote {args.out_json} and {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
