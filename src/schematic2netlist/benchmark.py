"""Benchmark scoring: align a predicted circuit to ground truth and
compute the full topology metric cascade with bootstrap CIs (Phase D).

Fairness problem this module solves: the pipeline and the human GT label
component terminals in *independent* index orders (terminal order is
arbitrary for 2-terminal passives). Comparing raw (component, index)
pairs would penalize a correct circuit for a bookkeeping difference. So
before scoring we (1) align predicted components to GT components by
bounding-box IoU within class (Hungarian; unmatched on either side are
kept with disjoint ids so they *penalize* graph metrics rather than
being dropped), and (2) canonicalize each component's terminal order by
a connectivity signature computed identically in pred and GT.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from schematic2netlist.classes import canonical_class
from schematic2netlist.metrics import (
    graph_edit_distance,
    net_level_metrics,
    normalized_ged,
    per_component_connected_accuracy,
    per_component_recall_accuracy,
    terminal_pair_metrics,
)


def iou_center(a, b) -> float:
    """IoU of two center-based (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def align_components(
    pred: list[dict], gt: list[dict], iou_threshold: float = 0.3
) -> tuple[list[dict], list[dict], dict]:
    """Relabel predicted component ids to their matched GT ids.

    Matches within the same canonical class by IoU (Hungarian). Unmatched
    predicted components get fresh ids disjoint from GT (so their
    terminals count as false connections); unmatched GT components keep
    their ids (their terminals count as missed). Returns
    (pred_relabeled, gt, stats).
    """
    next_free = max((c["id"] for c in gt), default=-1) + 1000
    pred_out = [dict(c) for c in pred]

    # candidate matches only within the same class
    pairs = []
    for pi, pc in enumerate(pred):
        for gi, gc in enumerate(gt):
            if canonical_class(pc["class"]) != canonical_class(gc["class"]):
                continue
            iou = iou_center(pc["bbox"], gc["bbox"])
            if iou >= iou_threshold:
                pairs.append((pi, gi, iou))

    matched_pred, matched_gt = set(), set()
    id_map: dict[int, int] = {}
    if pairs:
        pis = sorted({p[0] for p in pairs})
        gis = sorted({p[1] for p in pairs})
        cost = np.ones((len(pis), len(gis)))
        for pi, gi, iou in pairs:
            cost[pis.index(pi), gis.index(gi)] = 1 - iou
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            pi, gi = pis[r], gis[c]
            if iou_center(pred[pi]["bbox"], gt[gi]["bbox"]) >= iou_threshold:
                id_map[pred[pi]["id"]] = gt[gi]["id"]
                matched_pred.add(pi)
                matched_gt.add(gi)

    for pi, pc in enumerate(pred_out):
        if pi in matched_pred:
            pc["id"] = id_map[pred[pi]["id"]]
        else:
            pc["id"] = next_free
            next_free += 1

    stats = {
        "n_pred": len(pred),
        "n_gt": len(gt),
        "matched": len(matched_pred),
        "unmatched_pred": len(pred) - len(matched_pred),
        "unmatched_gt": len(gt) - len(matched_gt),
    }
    return pred_out, gt, stats


def _terminal_signature(comp: dict, net_partners: dict) -> list:
    """A per-terminal signature: the sorted partner-component ids sharing
    that terminal's net. Computed identically for pred and GT, so sorting
    by it yields a comparable terminal order despite independent indexing."""
    sigs = []
    for idx, net in enumerate(comp["nets"]):
        partners = sorted(net_partners.get(net, set()) - {comp["id"]}) if net else []
        sigs.append((tuple(partners), net is None))
    return sigs


def canonicalize_terminals(components: list[dict]) -> list[dict]:
    """Return components with terminals reordered by connectivity
    signature (stable; ties keep original order)."""
    net_partners: dict = {}
    for c in components:
        for net in c["nets"]:
            if net is not None:
                net_partners.setdefault(net, set()).add(c["id"])

    out = []
    for c in components:
        sigs = _terminal_signature(c, net_partners)
        order = sorted(range(len(c["nets"])), key=lambda i: (sigs[i], i))
        out.append({**c, "nets": [c["nets"][i] for i in order]})
    return out


def score_prediction(
    pred: list[dict], gt: list[dict], iou_threshold: float = 0.3
) -> dict:
    """Full topology metric cascade for one image (pred vs GT).

    pred/gt components are {"id", "class", "nets": [...], "bbox": [cx,cy,w,h]}.
    """
    pred_a, gt_a, stats = align_components(pred, gt, iou_threshold)
    pred_c = canonicalize_terminals(pred_a)
    gt_c = canonicalize_terminals(gt_a)

    tp = terminal_pair_metrics(pred_c, gt_c)
    net = net_level_metrics(pred_c, gt_c)
    pcc = per_component_connected_accuracy(pred_c, gt_c)
    pcr = per_component_recall_accuracy(pred_c, gt_c)
    nged = normalized_ged(pred_c, gt_c)

    strict = (
        stats["unmatched_gt"] == 0
        and tp["f1"] == 1.0
        and net["f1"] == 1.0
    )
    return {
        **stats,
        "terminal_pair_precision": tp["precision"],
        "terminal_pair_recall": tp["recall"],
        "terminal_pair_f1": tp["f1"],
        "net_f1": net["f1"],
        "per_component_connected_acc": pcc,
        "per_component_recall_acc": pcr,
        "nged": nged,
        "strict_success": bool(strict),
    }


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap (mean, lo, hi) for a per-image metric over the test set."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_resamples, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def aggregate(per_image: list[dict], n_resamples: int = 1000, seed: int = 0) -> dict:
    """Aggregate per-image metric dicts into means with bootstrap 95% CIs."""
    keys = [
        "terminal_pair_f1", "net_f1", "per_component_connected_acc",
        "per_component_recall_acc",
        "nged", "strict_success",
    ]
    out: dict = {"n_images": len(per_image)}
    for k in keys:
        vals = [float(r[k]) for r in per_image if k in r]
        mean, lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        out[k] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}
    return out
