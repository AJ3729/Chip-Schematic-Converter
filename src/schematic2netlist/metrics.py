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
import numpy as np
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


def _pairs_by_component(pairs: set[frozenset]) -> dict[int, set[frozenset]]:
    out: dict[int, set[frozenset]] = {}
    for pair in pairs:
        for term in pair:
            out.setdefault(term[0], set()).add(pair)
    return out


def per_component_recall_accuracy(pred: list[dict], gt: list[dict]) -> float:
    """Fraction of components whose GT terminal-pairs are ALL predicted.

    Recall only: extra predicted pairs are not penalised, so this cannot see
    over-merging. Retained because every result recorded before 2026-07-30 used
    it under the name ``per_component_connected_accuracy``; prefer the exact
    version below.
    """
    gt_by_comp = _pairs_by_component(_terminal_pairs(gt))
    if not gt_by_comp:
        return 1.0
    pred_pairs = _terminal_pairs(pred)
    return sum(1 for pairs in gt_by_comp.values()
               if pairs <= pred_pairs) / len(gt_by_comp)


def per_component_connected_accuracy(pred: list[dict], gt: list[dict]) -> float:
    """Fraction of components whose terminals are on EXACTLY the right nets,
    judged through the terminal-pair relation (net names need not match, only
    the induced grouping).

    THIS WAS RECALL-ONLY AND IS NOW EXACT. The previous implementation asked
    ``gt_pairs_of(c) <= pred_pairs``, which is satisfied by any prediction that
    merely contains the right pairs -- so a circuit with EVERY net welded into
    one scored 1.000, identical to a perfect answer, because welding only ever
    ADDS pairs. Welding is this pipeline's dominant failure mode, which made the
    metric blind to precisely the thing it was being used to track: the reported
    per-component numbers moved with splits and were indifferent to welds.

    A component now counts as correct only when the set of pairs touching its
    terminals is the same in both directions, so a weld to a foreign net fails
    it just as a missing connection does.

    ``per_component_recall_accuracy`` preserves the old behaviour, and the
    benchmark emits both, since every number recorded before 2026-07-30 is the
    recall variant and the two must not be silently compared.
    """
    gt_by_comp = _pairs_by_component(_terminal_pairs(gt))
    if not gt_by_comp:
        return 1.0
    pred_by_comp = _pairs_by_component(_terminal_pairs(pred))
    return sum(1 for comp, pairs in gt_by_comp.items()
               if pred_by_comp.get(comp, set()) == pairs) / len(gt_by_comp)


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


def _cost_of_mapping(G1: nx.Graph, G2: nx.Graph, mapping: dict) -> float:
    """Exact unit-cost edit cost of one given node mapping.

    Because the cost of a CONCRETE edit path is computed, whatever proposed the
    mapping, the result is always a valid upper bound on the true GED.
    """
    sub = sum(1 for u, v in mapping.items()
              if G1.nodes[u].get("label") != G2.nodes[v].get("label"))
    dele = G1.number_of_nodes() - len(mapping)
    ins = G2.number_of_nodes() - len(mapping)

    kept = set()
    e_del = 0
    for u, w in G1.edges:
        mu, mw = mapping.get(u), mapping.get(w)
        if mu is not None and mw is not None and G2.has_edge(mu, mw):
            kept.add(frozenset((mu, mw)))
        else:
            e_del += 1
    e_ins = sum(1 for a, b in G2.edges if frozenset((a, b)) not in kept)
    return float(sub + dele + ins + e_del + e_ins)


def _ged_assignment_bound(G1: nx.Graph, G2: nx.Graph) -> float:
    """Deterministic polynomial upper bound on GED (bipartite assignment).

    Riesen and Bunke's construction: a square cost matrix over
    nodes(G1) + epsilon against nodes(G2) + epsilon, where a substitution costs
    a label mismatch plus half the degree difference (half, so an edge edit is
    not charged to both of its endpoints), and a deletion or insertion costs the
    node plus half its incident edges. Hungarian on that matrix proposes a node
    mapping in O((n+m)^3); the returned value is then the EXACT cost of that
    mapping, so it is a genuine upper bound rather than the matrix's own sum.

    Nodes are sorted before the matrix is built, so the result depends only on
    the two graphs -- not on insertion order, dict iteration, or elapsed time.
    """
    key = lambda t: (str(t[0]), str(t[1]))
    n1, n2 = sorted(G1.nodes, key=key), sorted(G2.nodes, key=key)
    n, m = len(n1), len(n2)
    if n == 0 or m == 0:
        return _cost_of_mapping(G1, G2, {})

    d1 = [G1.degree(u) for u in n1]
    d2 = [G2.degree(v) for v in n2]
    l1 = [G1.nodes[u].get("label") for u in n1]
    l2 = [G2.nodes[v].get("label") for v in n2]

    BIG = 1e6
    size = n + m
    C = np.zeros((size, size), dtype=float)
    for i in range(n):
        for j in range(m):
            C[i, j] = (0.0 if l1[i] == l2[j] else 1.0) \
                + abs(d1[i] - d2[j]) / 2.0
    C[:n, m:] = BIG
    for i in range(n):
        C[i, m + i] = 1.0 + d1[i] / 2.0          # delete u_i
    C[n:, :m] = BIG
    for j in range(m):
        C[n + j, j] = 1.0 + d2[j] / 2.0          # insert v_j
    # the epsilon-to-epsilon block stays 0

    rows, cols = linear_sum_assignment(C)
    mapping = {n1[i]: n2[j] for i, j in zip(rows, cols) if i < n and j < m}
    return _cost_of_mapping(G1, G2, mapping)


def graph_edit_distance(
    pred: list[dict], gt: list[dict], timeout_s: float = 30.0,
    method: str = "assignment",
) -> float:
    """Graph edit distance between the predicted and GT topologies.

    ``method="assignment"`` (the default) returns the deterministic polynomial
    upper bound above. ``method="search"`` is the previous behaviour and is kept
    only so the two can be compared; DO NOT report numbers from it.

    WHY THE DEFAULT CHANGED. The search path called
    ``nx.graph_edit_distance(..., timeout=timeout_s)`` and treated any non-None
    return as exact. It is not: on timeout networkx returns the best bound found
    SO FAR as a plain float, so the "fell back to an upper bound" branch below
    was unreachable and the caller could not tell the two apart. Measured on 18
    test images at a 5 s budget, 13 burned the entire budget and every one
    returned a number -- so roughly 72% of images were reporting a bound whose
    value depends on how much CPU the process happened to receive.

    That is not a subtle risk, it was already corrupting results. Turning
    stitching off changes the recovered topology on ZERO images (every other
    metric is bit-identical, and the pipeline output is identical pixel for
    pixel) yet moved nGED on 24 of 190 images with a bootstrap CI excluding
    zero. A metric that reports a significant change when nothing changed cannot
    be used to decide anything, and every nGED number recorded before this fix
    has to be regenerated.
    """
    G1 = to_topology_graph(pred)
    G2 = to_topology_graph(gt)
    if method == "assignment":
        return _ged_assignment_bound(G1, G2)

    ged = nx.graph_edit_distance(
        G1, G2, node_match=_node_match, timeout=timeout_s
    )
    if ged is not None:
        return float(ged)
    best = None
    deadline = time.monotonic() + timeout_s
    for approx in nx.optimize_graph_edit_distance(G1, G2, node_match=_node_match):
        best = approx
        if time.monotonic() > deadline:
            break
    return float(best) if best is not None else float("inf")


def normalized_ged(
    pred: list[dict], gt: list[dict], timeout_s: float = 30.0,
    method: str = "assignment",
) -> float:
    """GED normalized by total graph size:
    nGED = GED / (|V1| + |E1| + |V2| + |E2|); 0.0 when both are empty.

    Deterministic by default -- see :func:`graph_edit_distance` for why the
    search-based version had to be retired."""
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
    return graph_edit_distance(
        pred, gt, timeout_s=timeout_s, method=method) / size
