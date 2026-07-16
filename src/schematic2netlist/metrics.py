"""Evaluation metrics.

Two families:

1. Coverage statistics (no ground truth needed). These only measure
   whether snapping *returned something*, not whether it was right —
   the legacy "terminal_snap_rate". Kept as clearly-labeled secondary
   coverage statistics.

2. Ground-truth metrics (Phase D), computed against annotated topology
   graphs: terminal-pair precision/recall/F1, net-level F1 via Hungarian
   matching, and normalized graph edit distance.

Graph representation used throughout: a circuit is a list of component
records ``{"id": int, "class": str, "nets": [net_name_or_None, ...]}``
where ``nets[i]`` names the electrical net terminal *i* connects to.
"""

from __future__ import annotations

import itertools
import time

import networkx as nx
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Coverage statistics (legacy semantics preserved for comparability)
# ---------------------------------------------------------------------------


def coverage_stats(components_with_nodes: list[dict]) -> dict:
    """Snap coverage over pipeline output (NOT a correctness measure).

    Semantics match the legacy evaluator: every component (including
    ground symbols, which store one node twice) contributes two
    terminals.
    """
    total_terms = 0
    snapped_terms = 0
    fully_connected = 0
    for c in components_with_nodes:
        nodes = c["nodes"]
        total_terms += 2
        snapped_terms += sum(n is not None for n in nodes)
        if all(n is not None for n in nodes):
            fully_connected += 1
    return {
        "num_components": len(components_with_nodes),
        "terminal_snap_rate": snapped_terms / max(1, total_terms),
        "fully_connected_rate": fully_connected / max(1, len(components_with_nodes)),
    }


# ---------------------------------------------------------------------------
# Ground-truth topology metrics
# ---------------------------------------------------------------------------


def _terminal_net_map(components: list[dict]) -> dict[tuple[int, int], str]:
    """(component_id, terminal_index) -> net name, skipping None nets."""
    out = {}
    for c in components:
        for t, net in enumerate(c["nets"]):
            if net is not None:
                out[(c["id"], t)] = net
    return out


def _terminal_pairs(components: list[dict]) -> set[frozenset]:
    """All unordered pairs of terminals that share a net."""
    by_net: dict[str, list[tuple[int, int]]] = {}
    for term, net in _terminal_net_map(components).items():
        by_net.setdefault(net, []).append(term)
    pairs: set[frozenset] = set()
    for terms in by_net.values():
        for a, b in itertools.combinations(sorted(terms), 2):
            pairs.add(frozenset((a, b)))
    return pairs


def _prf(tp: float, pred_total: float, gt_total: float) -> dict:
    precision = tp / pred_total if pred_total else 1.0
    recall = tp / gt_total if gt_total else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def terminal_pair_metrics(pred: list[dict], gt: list[dict]) -> dict:
    """Terminal-pair P/R/F1: a predicted pair of terminals is correct iff
    both terminals share a net in the ground truth as well.

    Assumes component ids are aligned between prediction and GT.
    """
    pred_pairs = _terminal_pairs(pred)
    gt_pairs = _terminal_pairs(gt)
    tp = len(pred_pairs & gt_pairs)
    return _prf(tp, len(pred_pairs), len(gt_pairs))


def net_level_metrics(pred: list[dict], gt: list[dict]) -> dict:
    """Net-level P/R/F1 via Hungarian matching of predicted nets to GT
    nets by terminal overlap.

    Precision = matched terminal overlap / total predicted terminal
    memberships; recall uses GT memberships.
    """
    def nets_as_sets(components):
        by_net: dict[str, set] = {}
        for term, net in _terminal_net_map(components).items():
            by_net.setdefault(net, set()).add(term)
        return list(by_net.values())

    pred_nets = nets_as_sets(pred)
    gt_nets = nets_as_sets(gt)
    pred_total = sum(len(s) for s in pred_nets)
    gt_total = sum(len(s) for s in gt_nets)

    if not pred_nets or not gt_nets:
        return _prf(0, pred_total, gt_total)

    # cost = negative overlap, so the assignment maximizes overlap
    cost = [[-len(p & g) for g in gt_nets] for p in pred_nets]
    rows, cols = linear_sum_assignment(cost)
    matched_overlap = sum(-cost[r][c] for r, c in zip(rows, cols))
    return _prf(matched_overlap, pred_total, gt_total)


def per_component_connected_accuracy(pred: list[dict], gt: list[dict]) -> float:
    """Fraction of components whose every terminal is on the correct net,
    judged through the terminal-pair relation (net names need not match,
    only the induced grouping)."""
    pred_pairs = _terminal_pairs(pred)
    gt_pairs = _terminal_pairs(gt)
    gt_by_comp: dict[int, set[frozenset]] = {}
    for pair in gt_pairs:
        for term in pair:
            gt_by_comp.setdefault(term[0], set()).add(pair)
    if not gt_by_comp:
        return 1.0
    correct = sum(
        1 for comp, pairs in gt_by_comp.items() if pairs <= pred_pairs
    )
    return correct / len(gt_by_comp)


# ---------------------------------------------------------------------------
# Graph edit distance
# ---------------------------------------------------------------------------


def to_topology_graph(components: list[dict]) -> nx.Graph:
    """Bipartite graph: component nodes (labeled by class) and net nodes
    (labeled "net"), edges for each terminal-net membership."""
    G = nx.Graph()
    for c in components:
        G.add_node(("c", c["id"]), label=c["class"])
    for term, net in _terminal_net_map(components).items():
        net_node = ("net", net)
        if net_node not in G:
            G.add_node(net_node, label="net")
        G.add_edge(("c", term[0]), net_node)
    return G


def _node_match(a: dict, b: dict) -> bool:
    return a.get("label") == b.get("label")


def graph_edit_distance(
    pred: list[dict], gt: list[dict], timeout_s: float = 30.0
) -> float:
    """Exact GED when feasible; falls back to the best upper bound found
    within the time budget for large graphs."""
    G1 = to_topology_graph(pred)
    G2 = to_topology_graph(gt)
    ged = nx.graph_edit_distance(
        G1, G2, node_match=_node_match, timeout=timeout_s
    )
    if ged is not None:
        return float(ged)
    # timeout without an exact answer: take the first (cheapest-found)
    # upper bound within a small extra budget
    best = None
    deadline = time.monotonic() + timeout_s
    for approx in nx.optimize_graph_edit_distance(G1, G2, node_match=_node_match):
        best = approx
        if time.monotonic() > deadline:
            break
    return float(best) if best is not None else float("inf")


def normalized_ged(
    pred: list[dict], gt: list[dict], timeout_s: float = 30.0
) -> float:
    """GED normalized by total graph size:
    nGED = GED / (|V1| + |E1| + |V2| + |E2|); 0.0 when both are empty."""
    G1 = to_topology_graph(pred)
    G2 = to_topology_graph(gt)
    size = (
        G1.number_of_nodes()
        + G1.number_of_edges()
        + G2.number_of_nodes()
        + G2.number_of_edges()
    )
    if size == 0:
        return 0.0
    return graph_edit_distance(pred, gt, timeout_s=timeout_s) / size
