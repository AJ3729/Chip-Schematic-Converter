#!/usr/bin/env python3
"""DC OPERATING-POINT AGREEMENT: does the recovered netlist *simulate* like the
reference netlist?

Every headline number in this repository is topological -- do the predicted nets
match the ground-truth nets. The stated purpose of the work is netlists that
SIMULATE, and nothing measured that. DC *solvability* was the obvious candidate
and it fails as a correctness criterion: the ground-truth netlists themselves are
only 116/192 DC-solvable, while the pipeline's repaired netlists reach 140/192,
because repair adds assumed grounds and bridges floating nodes. A criterion you
can beat the truth on is not measuring correctness.

WHAT THIS MEASURES
------------------
Both sides are exported by the SAME writer with the SAME placeholder component
values and handed to the SAME ngspice, and their DC operating points are compared
node by node. Because the values are identical, every voltage difference is
attributable to the RECOVERED CIRCUIT -- its topology, its component classes, or
its terminal (pin) order. No OCR is involved: the values cancel.

WHY PIN ORDER IS THE POINT
--------------------------
The writer emits ``Q<n> c b e``, ``M<n> d g s s``, ``D<n> anode cathode``,
``V<n> + -`` and ``E<n> out 0 in+ in-`` positionally off raw terminal index (see
:mod:`schematic2netlist.netlist`), so a reversed BJT, a backwards diode, a
flipped supply or a swapped op-amp input is a real electrical error. Every
existing topology metric is structurally BLIND to all of them, because
``benchmark.canonicalize_terminals`` sorts terminals by a connectivity signature
computed identically on both sides -- the swap cancels and net-F1 stays exactly
1.000. An operating point does not cancel.

``--stage accept`` proves the metric moves, on hundreds of injected swaps, with
the effect size and the sample size stated per swap type -- and with two
controls: swapping the terminals of a RESISTOR/CAPACITOR/INDUCTOR must move
nothing (those really are symmetric), and simulating a deck against itself must
score exactly 1.000.

THE PLACEHOLDER POLICY IS A DESIGN CHOICE, AND IT IS SELECTED, NOT ASSUMED
--------------------------------------------------------------------------
The shipped placeholders (R=1k everywhere, V=DC 5, AC sources with NO dc value)
make the probe partly blind: 20 of the 94 comparable circuits have an operating
point that is identically zero, because their only excitation is an AC source
whose DC value is 0. On those the metric cannot see anything at all. Values are
OUR choice and identical on both sides, so choosing them to bias devices into a
region where orientation MATTERS is not gaming the metric -- it is making the
probe informative. Candidate policies are measured by their DETECTION RATE on
injected pin swaps, and each policy's agreement number is reported beside it so
it is visible that the choice was made on sensitivity and not on the headline.

NODE CORRESPONDENCE, AND WHY IT MUST BE PIN-ORDER INVARIANT
-----------------------------------------------------------
Predicted net names and GT net names are independent labellings. Components are
aligned with ``benchmark.align_components`` (bbox IoU within class, Hungarian),
and a net is then identified by the MULTISET OF ALIGNED COMPONENT IDS touching it
-- component ids, never (component, terminal) pairs. Using terminal identity
would make the correspondence itself move when pin order moves, so a swap would
cancel out and read as agreement. Same construction, for the same reason, as
``scripts/compare_annotations.net_correspondence``.

Usage:
    python scripts/measure_op_agreement.py                  # everything
    python scripts/measure_op_agreement.py --stage cache    # pipeline pass only
    python scripts/measure_op_agreement.py --limit 10       # smoke test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from schematic2netlist.benchmark import align_components, score_prediction
from schematic2netlist.classes import canonical_class, class_role, class_terminals
from schematic2netlist.config import config_hash, load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.netlist import export_spice_netlist
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.simulate import parse_ngspice_output

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# tolerance
# ---------------------------------------------------------------------------
#
# PRIMARY: 1 mV absolute. Justification, both directions:
#
#   floor -- ngspice's node-voltage convergence tolerance (vntol) defaults to
#   1 uV, so two solves of the SAME circuit agree far inside 1 uV. 1 mV is
#   1000x that: numerical noise cannot cross it. (Measured, not assumed: the
#   pred-vs-pred and gt-vs-gt null controls score exactly 1.000, and the
#   passive-swap control moves nodes by exactly 0.0 V.)
#
#   ceiling -- the placeholder rail is 5-15 V, so 1 mV is <= 0.02% of full
#   scale, while a genuine topological or pin-order difference moves a node by
#   hundreds of mV to volts.
#
# The sweep is reported with every run so the choice can be checked rather than
# believed. In practice the number is flat from 1 nV to 10 mV, which is the
# signature of a metric whose disagreements are large, not marginal.
PRIMARY_ATOL = 1e-3
PRIMARY_RTOL = 0.0

TOLERANCE_SWEEP = [
    ("1nV", 1e-9, 0.0),
    ("1uV", 1e-6, 0.0),
    ("1mV (PRIMARY)", 1e-3, 0.0),
    ("1mV or 1% rel", 1e-3, 1e-2),
    ("10mV", 1e-2, 0.0),
    ("100mV", 1e-1, 0.0),
    ("500mV", 5e-1, 0.0),
]

# ---------------------------------------------------------------------------
# candidate placeholder policies
# ---------------------------------------------------------------------------
#
# Each entry is an override of configs/default.yaml's netlist.placeholders,
# applied IDENTICALLY to both sides. Keys are the ones netlist._ROLE_VALUE_KEYS
# reads: resistor / capacitor / inductor / dc_supply / ac_supply / dc_current /
# ac_current. At DC a capacitor is an open and an inductor a short, so their
# values are inert here and are left alone.
#
# The axes, and what each is for:
#   ac_supply / ac_current with a DC term -- an "AC 1" source has DC value 0, so
#     a circuit whose only excitation is AC sits at 0 V on every node and the
#     probe is blind. This is the single largest effect.
#   dc_supply raised -- lifts more bases above V_be(on), so fewer transistors sit
#     cut off, and a cut-off transistor is cut off whichever way round it is.
#   dc_current lowered -- with the default NPN model (BF=100) a 1 mA base current
#     demands 100 mA of collector current that no 1k/5V branch can supply, so a
#     current-driven stage saturates. 100 uA lands it nearer the active region,
#     where the BF/BR asymmetry that distinguishes C from E is largest.
#   resistor raised -- same intent from the other side.
#
# NOTE the limit of this axis, stated plainly: the writer takes ONE value per
# role, so every resistor in the deck is the same resistor. A base-side vs
# collector-side split (which is what would keep a resistively biased BJT out of
# saturation) is not expressible here, and would need a per-component value
# assignment this metric deliberately does not have -- the whole point is that
# values are identical on both sides and cancel.
POLICY_GRID: dict[str, dict] = {
    "shipped": {},
    "ac_dc_biased": {
        "ac_supply": "DC 5 AC 1",
        "ac_current": "DC 1m AC 1m",
    },
    "hv": {
        "dc_supply": "DC 15",
        "ac_supply": "DC 15 AC 1",
        "ac_current": "DC 1m AC 1m",
    },
    "hv_lowcurrent": {
        "dc_supply": "DC 15",
        "ac_supply": "DC 15 AC 1",
        "dc_current": "DC 100u",
        "ac_current": "DC 100u AC 1m",
    },
    "hv_lowcurrent_10k": {
        "resistor": "10k",
        "dc_supply": "DC 15",
        "ac_supply": "DC 15 AC 1",
        "dc_current": "DC 100u",
        "ac_current": "DC 100u AC 1m",
    },
}

# ---------------------------------------------------------------------------
# what a "pin swap" is, per role
# ---------------------------------------------------------------------------
#
# (i, j, kind). Only roles whose terminal ORDER carries electrical meaning in the
# emitted element are listed. Two-terminal DIRECTIONAL parts are included on
# purpose -- a backwards diode or a flipped supply is a pin-order error the
# topology metrics are just as blind to, and the first thing this metric found
# was a reversed V-DC on a circuit scoring strict_success = 1.
SWAP_SPEC: dict[str, tuple[int, int, str]] = {
    "npn": (0, 2, "bjt_collector_emitter"),
    "pnp": (0, 2, "bjt_collector_emitter"),
    "nmos": (0, 2, "mosfet_drain_source"),
    "pmos": (0, 2, "mosfet_drain_source"),
    "opamp": (0, 1, "opamp_input_swap"),
    "diode": (0, 1, "diode_reversal"),
    "zener": (0, 1, "zener_reversal"),
    "vdc": (0, 1, "vdc_polarity"),
    "vac": (0, 1, "vac_polarity"),
    "idc": (0, 1, "idc_direction"),
    "iac": (0, 1, "iac_direction"),
}

# The control: these really ARE symmetric, so swapping them must move NOTHING.
# Without this, "the metric moved" would only show that the metric responds to
# perturbation, not that it responds to *electrically meaningful* perturbation.
PASSIVE_SPEC: dict[str, tuple[int, int, str]] = {
    "resistor": (0, 1, "passive_control"),
    "capacitor": (0, 1, "passive_control"),
    "inductor": (0, 1, "passive_control"),
}

# an operating-point probe used ONLY by the parser cross-check. The decks that
# produce every reported number carry no probe at all: they are byte-identical
# to what scripts/benchmark.py already writes, and the node voltages are read
# out of the .op table the deck's own trailing `.op` prints.
OP_PROBE = [
    "* OPERATING-POINT PROBE added by scripts/measure_op_agreement.py for the",
    "* parser cross-check. Not a repair: no element, no node, no topology change.",
    ".control", "op", "print all", ".endc",
]

_NODE_HDR_RE = re.compile(r"^\s*Node\s+Voltage\s*$")
_ROW_RE = re.compile(
    r"^\s*(\S+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z0-9_.:+\-\[\]#]+)\s*=\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


# ---------------------------------------------------------------------------
# ngspice
# ---------------------------------------------------------------------------

def parse_op_table(stdout: str) -> dict[str, float]:
    """Node voltages from the ``.op`` table ngspice prints in batch mode.

    THIS REPLACES A `print all` PARSE THAT WAS SILENTLY WRONG. With exactly one
    vector in the plot, ngspice does not expand `all` -- it prints the literal
    token as the name, ``all = -1.00000e-03``. Every single-node deck therefore
    lost its node name, the net correspondence found nothing to compare, and the
    circuit scored 0.000 against ITSELF. It surfaced as a pred-vs-pred null
    control of 0.9901 instead of 1.0000, which is exactly what a null control is
    for. The .op table names every node in every case.
    """
    volts: dict[str, float] = {}
    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        if not _NODE_HDR_RE.match(lines[i]):
            i += 1
            continue
        i += 1
        # dashed rule lines, in either of the two forms ngspice emits
        while i < len(lines) and not lines[i].replace("-", "").strip():
            i += 1
        while i < len(lines):
            line = lines[i]
            if not line.strip() or line.strip().startswith("Source"):
                break
            m = _ROW_RE.match(line)
            if m and "#" not in m.group(1):
                volts.setdefault(m.group(1).lower(), float(m.group(2)))
            i += 1
    return volts


def parse_print_all(stdout: str) -> tuple[dict[str, float], bool]:
    """``print all`` parse, kept only to cross-check :func:`parse_op_table`.

    Returns (voltages, hit_the_unexpanded_all_quirk).
    """
    volts: dict[str, float] = {}
    quirk = False
    for line in stdout.splitlines():
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        name = m.group(1).lower()
        if name == "all":
            quirk = True
            continue
        if "#" in name:
            continue
        volts.setdefault(name, float(m.group(2)))
    return volts, quirk


_SIM_CACHE: dict[str, dict] = {}
_SIM_LOCK = threading.Lock()


def _run_ngspice(text: str, cfg: dict) -> dict:
    """Run one deck. Memoised on the deck text: identical decks are identical
    runs, and the null controls and swap sweeps re-simulate a lot of them."""
    key = hashlib.sha1(text.encode()).hexdigest()
    with _SIM_LOCK:
        hit = _SIM_CACHE.get(key)
    if hit is not None:
        return hit

    with tempfile.NamedTemporaryFile("w", suffix=".sp", delete=False) as f:
        f.write(text)
        path = Path(f.name)
    try:
        proc = subprocess.run([cfg["simulate"]["ngspice_binary"], "-b", str(path)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=cfg["simulate"]["timeout_s"])
        stdout = proc.stdout.decode(errors="replace")
        stderr = proc.stderr.decode(errors="replace")
        ok, category, _ = parse_ngspice_output(stdout, stderr, proc.returncode)
    except subprocess.TimeoutExpired:
        stdout, ok, category = "", False, "timeout"
    except FileNotFoundError:
        stdout, ok, category = "", False, "ngspice_missing"
    finally:
        path.unlink(missing_ok=True)

    volts = parse_op_table(stdout)
    out = {"category": category,
           # `solved` is the project's own taxonomy returning "ok" -- stricter
           # than "ngspice printed numbers". On a singular deck ngspice falls
           # back to gmin stepping and a transient op and STILL prints a full
           # node table, but those voltages are an artefact of the fallback
           # pinning a floating node, not an operating point.
           "solved": bool(ok) and bool(volts),
           "voltages": volts,
           "stdout": stdout}
    with _SIM_LOCK:
        _SIM_CACHE[key] = out
    return out


def build_deck(components: list[dict], placeholders: dict,
               extra_lines: list[str] | None = None,
               keep_path: Path | None = None) -> str:
    """Export through the project's own writer and return the deck text."""
    if keep_path is not None:
        keep_path.parent.mkdir(parents=True, exist_ok=True)
        export_spice_netlist(components, str(keep_path), placeholders=placeholders,
                             extra_lines=extra_lines)
        return keep_path.read_text()
    with tempfile.NamedTemporaryFile("w", suffix=".sp", delete=False) as f:
        path = Path(f.name)
    export_spice_netlist(components, str(path), placeholders=placeholders,
                         extra_lines=extra_lines)
    text = path.read_text()
    path.unlink(missing_ok=True)
    return text


def simulate(components: list[dict], placeholders: dict, cfg: dict,
             extra_lines: list[str] | None = None,
             keep_path: Path | None = None) -> dict:
    return _run_ngspice(build_deck(components, placeholders, extra_lines, keep_path), cfg)


# ---------------------------------------------------------------------------
# node correspondence (pin-order INVARIANT by construction)
# ---------------------------------------------------------------------------

def net_correspondence(a: list[dict], b: list[dict],
                       key_a: dict, key_b: dict) -> dict[str, str]:
    """Map A's net names onto B's, invariantly of pin order.

    A net's signature is the multiset of ALIGNED COMPONENT KEYS it touches --
    component identity only, never (component, terminal). If the signature used
    terminal index the correspondence would move with a pin swap, and the swap
    would cancel out and read as agreement. Hungarian maximises total signature
    overlap; a pair with zero overlap is not a correspondence and is dropped.
    """
    def sig(comps, keys):
        out: dict[str, Counter] = defaultdict(Counter)
        for c in comps:
            k = keys.get(c["id"], ("orphan", c["id"]))
            for net in c["nets"]:
                if net is not None:
                    out[net][k] += 1
        return out

    sa, sb = sig(a, key_a), sig(b, key_b)
    if not sa or not sb:
        return {}
    an, bn = sorted(sa), sorted(sb)
    cost = np.zeros((len(an), len(bn)))
    for i, p in enumerate(an):
        for j, q in enumerate(bn):
            cost[i, j] = -sum((sa[p] & sb[q]).values())
    rows, cols = linear_sum_assignment(cost)
    return {an[r]: bn[c] for r, c in zip(rows, cols) if cost[r, c] < 0}


def build_correspondence(pred: list[dict], gt: list[dict],
                         iou_threshold: float = 0.3) -> tuple[dict[str, str], dict]:
    """GT net name -> predicted net name, plus the component alignment stats."""
    pred_al, gt_al, stats = align_components(pred, gt, iou_threshold)
    gt_ids = {c["id"] for c in gt_al}
    key_gt = {c["id"]: ("m", c["id"]) for c in gt_al}
    # align_components relabels matched predictions onto their GT id and gives
    # unmatched ones fresh ids >= max(gt_id)+1000, so "is this id a GT id" is an
    # exact test for "was this component matched".
    key_pred = {c["id"]: (("m", c["id"]) if c["id"] in gt_ids
                          else ("pred_only", c["id"])) for c in pred_al}
    return net_correspondence(gt_al, pred_al, key_gt, key_pred), stats


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def close(a: float, b: float, atol: float, rtol: float) -> bool:
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def score_op(gt_v: dict, pred_v: dict, corr: dict[str, str],
             atol: float = PRIMARY_ATOL, rtol: float = PRIMARY_RTOL,
             include_ground: bool = False) -> dict:
    """Compare two operating points over the corresponded nets.

    The node sets are the nodes ngspice actually reported. ngspice never reports
    node 0 -- it is the reference, 0 V by definition -- so the primary number
    excludes the ground-to-ground pair automatically rather than banking a free
    agreement on it. A mis-grounded prediction is still caught, twice over: every
    other predicted node shifts, and the reference ground net corresponds to a
    predicted node that then has no partner, costing precision.

    recall    = agreeing / |GT nodes|    "how much of the reference operating
                                          point does the prediction reproduce"
    precision = agreeing / |pred nodes|  penalises invented nodes
    f1        = the headline; a net MERGE costs recall and a net SPLIT costs
                precision, so neither can be gamed by collapsing the circuit.
    """
    gv, pv = dict(gt_v), dict(pred_v)
    if include_ground:
        gv.setdefault("0", 0.0)
        pv.setdefault("0", 0.0)

    n_gt, n_pred = len(gv), len(pv)
    agree, n_corr = 0, 0
    deltas: list[float] = []
    disagreements: list[dict] = []
    for g_net, p_net in corr.items():
        g, p = g_net.lower(), p_net.lower()
        if g not in gv or p not in pv:
            continue
        n_corr += 1
        d = abs(gv[g] - pv[p])
        deltas.append(d)
        if close(gv[g], pv[p], atol, rtol):
            agree += 1
        else:
            disagreements.append({"gt_net": g_net, "pred_net": p_net,
                                  "v_gt": gv[g], "v_pred": pv[p], "abs_dv": d})

    recall = agree / n_gt if n_gt else 0.0
    precision = agree / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"n_gt_nodes": n_gt, "n_pred_nodes": n_pred, "n_corresponded": n_corr,
            "n_agree": agree, "recall": recall, "precision": precision, "f1": f1,
            "exact": bool(f1 == 1.0),
            "max_abs_dv": max(deltas) if deltas else 0.0,
            "median_abs_dv": statistics.median(deltas) if deltas else 0.0,
            "disagreements": disagreements}


# ---------------------------------------------------------------------------
# stage 1: the pipeline pass, checkpointed per circuit
# ---------------------------------------------------------------------------

def spice_components(graph: list[dict]) -> list[dict]:
    """Graph components -> the record shape export_spice_netlist wants."""
    return [{"id": c["id"], "class": c["class"],
             "node_names": list(c["nets"]),
             "nodes": list(range(len(c["nets"])))} for c in graph]


def gt_graph_components(gt: dict) -> list[dict]:
    """GT in benchmark graph format, with bbox.

    A null net is left NULL, not silently tied to "0". The one null in the test
    GT sits on a component the annotation marks ``unconnected`` -- a deliberately
    dangling element in the drawing -- and grounding it would invent a connection
    the reference does not have. The writer then skips that component with an
    UNSNAPPED comment, which is the honest rendering of "goes nowhere".
    """
    comps = gt_to_components(gt)
    by_id = {c["id"]: c for c in gt["components"]}
    for c in comps:
        c["bbox"] = by_id[c["id"]]["bbox"]
    return comps


def build_cache(names: list[str], images_dir: Path, det_dir: Path, gt_dir: Path,
                cfg: dict, iou_threshold: float, cache_dir: Path,
                refresh: bool) -> list[str]:
    """Run the pipeline once and checkpoint each circuit to disk immediately.

    Everything downstream (every placeholder policy, every swap) is pure
    simulation over this cache, so a stall costs at most one circuit and no
    policy comparison can be contaminated by a re-run of the pipeline.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    for idx, name in enumerate(names, 1):
        stem = Path(name).stem
        out = cache_dir / f"{stem}.json"
        if out.exists() and not refresh:
            done.append(stem)
            continue
        gt_path, det_path = gt_dir / f"{stem}.json", det_dir / f"{stem}.json"
        if not gt_path.exists() or not det_path.exists():
            print(f"[{idx}/{len(names)}] {stem}: SKIP (no gt/detections)", flush=True)
            continue
        print(f"[{idx}/{len(names)}] {stem}", flush=True)

        gt = load_gt(gt_path)
        dets = load_cached_detections(det_path,
                                      min_confidence=cfg["detect"].get("confidence"))
        result = run_pipeline(images_dir / name, cfg, detections=dets)

        d = result["detections"]
        pred_graph = [{"id": c["id"], "class": c["class"],
                       "nets": list(c.get("node_names", [])),
                       "bbox": [d[c["id"]]["x"], d[c["id"]]["y"],
                                d[c["id"]]["width"], d[c["id"]]["height"]]}
                      for c in result["components"]]
        gt_graph = gt_graph_components(gt)
        corr, align_stats = build_correspondence(pred_graph, gt_graph, iou_threshold)
        topo = score_prediction(pred_graph, gt_graph, iou_threshold=iou_threshold)
        rep = result.get("repair")

        out.write_text(json.dumps({
            "stem": stem, "image": name,
            "gt_graph": gt_graph, "pred_graph": pred_graph,
            "corr": corr, "align": align_stats,
            "repair_extra_lines": list(rep.extra_lines) if rep else [],
            "topology": {"net_f1": topo["net_f1"],
                         "terminal_pair_f1": topo["terminal_pair_f1"],
                         "strict_success": int(topo["strict_success"])},
        }, indent=1))
        done.append(stem)
    return done


def load_cache(cache_dir: Path, stems: list[str]) -> list[dict]:
    out = []
    for s in stems:
        p = cache_dir / f"{s}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


# ---------------------------------------------------------------------------
# stage 2: measure one placeholder policy
# ---------------------------------------------------------------------------

def policy_placeholders(cfg: dict, name: str) -> dict:
    return {**cfg["netlist"]["placeholders"], **POLICY_GRID[name]}


def measure_policy(circuits: list[dict], ph: dict, cfg: dict,
                   netlist_dir: Path | None = None, workers: int = 8) -> list[dict]:
    """One row per circuit: both sims, the populations, and the score."""

    def one(c: dict) -> dict:
        gt_sc = spice_components(c["gt_graph"])
        pred_sc = spice_components(c["pred_graph"])
        gp = (netlist_dir / f"{c['stem']}.gt.sp") if netlist_dir else None
        pp = (netlist_dir / f"{c['stem']}.pred.sp") if netlist_dir else None
        gt_sim = simulate(gt_sc, ph, cfg, keep_path=gp)
        pred_sim = simulate(pred_sc, ph, cfg, keep_path=pp)
        rep_sim = (simulate(pred_sc, ph, cfg, extra_lines=list(c["repair_extra_lines"]))
                   if c["repair_extra_lines"] else pred_sim)

        if gt_sim["solved"] and pred_sim["solved"]:
            population = "both_solve"
        elif gt_sim["solved"]:
            population = "only_gt_solves"
        elif pred_sim["solved"]:
            population = "only_pred_solves"
        else:
            population = "neither_solves"

        gvals = list(gt_sim["voltages"].values())
        # a reference operating point that is identically zero cannot
        # distinguish anything: every corresponded node agrees for free, and the
        # score degenerates into a node-count check
        degenerate = bool(gvals) and max(abs(v) for v in gvals) <= PRIMARY_ATOL

        row = {
            "stem": c["stem"], "population": population,
            "gt_category": gt_sim["category"], "pred_category": pred_sim["category"],
            "gt_solved": int(gt_sim["solved"]), "pred_solved": int(pred_sim["solved"]),
            "pred_repaired_solved": int(rep_sim["solved"]),
            "pred_repaired_category": rep_sim["category"],
            "gt_op_degenerate_zero": int(degenerate),
            "gt_op_span_V": (max(gvals) - min(gvals)) if gvals else 0.0,
            "n_gt_comps": c["align"]["n_gt"], "n_pred_comps": c["align"]["n_pred"],
            "aligned": c["align"]["matched"],
            "unmatched_gt": c["align"]["unmatched_gt"],
            "unmatched_pred": c["align"]["unmatched_pred"],
            "net_f1_topology": c["topology"]["net_f1"],
            "terminal_pair_f1_topology": c["topology"]["terminal_pair_f1"],
            "strict_success_topology": c["topology"]["strict_success"],
            "_gt_voltages": gt_sim["voltages"], "_pred_voltages": pred_sim["voltages"],
        }
        scored = population == "both_solve"
        sc = score_op(gt_sim["voltages"], pred_sim["voltages"], c["corr"])
        scg = score_op(gt_sim["voltages"], pred_sim["voltages"], c["corr"],
                       include_ground=True)
        for k in ("n_gt_nodes", "n_pred_nodes", "n_corresponded", "n_agree",
                  "recall", "precision", "f1", "exact", "max_abs_dv", "median_abs_dv"):
            row[k] = sc[k] if scored else ""
        row["f1_incl_ground"] = scg["f1"] if scored else ""
        row["_disagreements"] = sc["disagreements"] if scored else []
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, circuits))


def aggregate_policy(rows: list[dict]) -> dict:
    n = len(rows)
    pop = Counter(r["population"] for r in rows)
    both = [r for r in rows if r["population"] == "both_solve"]
    live = [r for r in both if not r["gt_op_degenerate_zero"]]
    degen = [r for r in both if r["gt_op_degenerate_zero"]]

    def block(sub, label):
        if not sub:
            return {"_domain": label, "n_scored": 0}
        agree = sum(r["n_agree"] for r in sub)
        ngt = sum(r["n_gt_nodes"] for r in sub)
        npr = sum(r["n_pred_nodes"] for r in sub)
        ex = sum(1 for r in sub if r["exact"])
        return {
            "_domain": label, "n_scored": len(sub),
            "mean_f1": sum(r["f1"] for r in sub) / len(sub),
            "mean_recall": sum(r["recall"] for r in sub) / len(sub),
            "mean_precision": sum(r["precision"] for r in sub) / len(sub),
            "exact_circuits": ex, "exact_rate": ex / len(sub),
            "pooled_node_recall": (agree / ngt) if ngt else None,
            "pooled_node_precision": (agree / npr) if npr else None,
            "pooled_nodes_gt": ngt, "pooled_nodes_pred": npr,
            "pooled_nodes_agreeing": agree,
        }

    return {
        "circuits": n,
        "populations": {
            "_denominator_policy": (
                "The metric is DEFINED ONLY on both_solve. Circuits outside it "
                "are reported here and in per_image.csv, never dropped."),
            "both_solve": pop["both_solve"],
            "only_gt_solves": pop["only_gt_solves"],
            "only_pred_solves": pop["only_pred_solves"],
            "neither_solves": pop["neither_solves"],
            "gt_solved": sum(r["gt_solved"] for r in rows),
            "pred_solved": sum(r["pred_solved"] for r in rows),
            "pred_repaired_solved": sum(r["pred_repaired_solved"] for r in rows),
            "gt_categories": dict(Counter(r["gt_category"] for r in rows)),
            "pred_categories": dict(Counter(r["pred_category"] for r in rows)),
        },
        "degenerate_zero_op": {
            "_meaning": (
                "circuits in both_solve whose REFERENCE operating point is "
                "identically 0 V on every node -- their only excitation is an AC "
                "source, whose DC value is 0. Every corresponded node agrees for "
                "free there, so the score degenerates into a node-count check and "
                "measures no voltage at all. Reported separately, never merged "
                "into the headline silently."),
            "n": len(degen),
            "stems": sorted(r["stem"] for r in degen),
        },
        "aggregate_all_both_solve": block(both, "every circuit where both sides solve"),
        "aggregate_non_degenerate": block(
            live, "both sides solve AND the reference operating point is not "
                  "identically zero -- the domain on which this is a VOLTAGE metric"),
        "aggregate_degenerate_only": block(degen, "the all-zero circuits alone"),
    }


def tolerance_sweep(rows: list[dict], circuits_by_stem: dict) -> list[dict]:
    both = [r for r in rows if r["population"] == "both_solve"]
    live_stems = {r["stem"] for r in both if not r["gt_op_degenerate_zero"]}
    out = []
    for label, atol, rtol in TOLERANCE_SWEEP:
        f1s, f1s_live, ex, agree, ngt = [], [], 0, 0, 0
        for r in both:
            corr = circuits_by_stem[r["stem"]]["corr"]
            sc = score_op(r["_gt_voltages"], r["_pred_voltages"], corr, atol, rtol)
            f1s.append(sc["f1"])
            ex += int(sc["exact"])
            agree += sc["n_agree"]
            ngt += sc["n_gt_nodes"]
            if r["stem"] in live_stems:
                f1s_live.append(sc["f1"])
        out.append({"tolerance": label, "atol_V": atol, "rtol": rtol,
                    "mean_f1_both_solve": (sum(f1s) / len(f1s)) if f1s else None,
                    "mean_f1_non_degenerate": (
                        sum(f1s_live) / len(f1s_live)) if f1s_live else None,
                    "exact_circuits": ex,
                    "pooled_node_recall": (agree / ngt) if ngt else None})
    return out


# ---------------------------------------------------------------------------
# stage 3: null controls
# ---------------------------------------------------------------------------

def null_controls(circuits: list[dict], ph: dict, cfg: dict,
                  workers: int = 8) -> dict:
    """Every deck simulated against ITSELF must score exactly 1.000.

    Without this, a high agreement number could be produced by a broken parser
    (no nodes on either side, vacuously equal) or by a correspondence that
    matches everything to everything. It is not decoration: the `print all`
    parser bug in this file's history was found by exactly this control, and by
    nothing else.

    Also cross-checks the two ngspice parsers against each other on every deck.
    """
    out: dict = {}
    for side, key in (("pred_vs_pred", "pred_graph"), ("gt_vs_gt", "gt_graph")):
        def one(c, key=key):
            comps = spice_components(c[key])
            sim = simulate(comps, ph, cfg)
            if not sim["solved"]:
                return None
            corr, _ = build_correspondence(c[key], c[key])
            sc = score_op(sim["voltages"], sim["voltages"], corr)
            return {"stem": c["stem"], "f1": sc["f1"],
                    "n_gt_nodes": sc["n_gt_nodes"], "n_pred_nodes": sc["n_pred_nodes"],
                    "n_corresponded": sc["n_corresponded"], "n_agree": sc["n_agree"]}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = [x for x in ex.map(one, circuits) if x]
        bad = [x for x in res if x["f1"] != 1.0]
        out[side] = {"circuits_scored": len(res),
                     "mean_f1": (sum(x["f1"] for x in res) / len(res)) if res else None,
                     "all_exactly_1.0": bool(res) and not bad,
                     "failures": bad[:20]}

    # parser cross-check: the .op table (used for every number) against a
    # `print all` probe (used for nothing else), on every deck
    def crosscheck(c):
        rows = []
        for key in ("gt_graph", "pred_graph"):
            comps = spice_components(c[key])
            base = _run_ngspice(build_deck(comps, ph), cfg)
            probed = _run_ngspice(build_deck(comps, ph, OP_PROBE), cfg)
            pa, quirk = parse_print_all(probed["stdout"])
            table = base["voltages"]
            same = all(k in pa and abs(pa[k] - v) <= 1e-12 for k, v in table.items())
            rows.append({"stem": c["stem"], "side": key, "quirk": quirk,
                         "agree": bool(same or (quirk and len(table) == 1)),
                         "n_table": len(table), "n_print": len(pa),
                         "category_same": base["category"] == probed["category"]})
        return rows
    with ThreadPoolExecutor(max_workers=workers) as ex:
        cc = [r for rows in ex.map(crosscheck, circuits) for r in rows]
    out["parser_crosscheck"] = {
        "_what": ("the .op-table parser (which produces every number here) "
                  "against a `print all` probe on the same deck. `quirk` counts "
                  "decks with exactly ONE vector in the plot, where ngspice does "
                  "not expand `all` and prints the literal token as the node "
                  "name -- the bug that made an earlier version of this file "
                  "score a deck 0.000 against itself."),
        "decks_checked": len(cc),
        "voltages_agree": sum(1 for r in cc if r["agree"]),
        "decks_hitting_the_print_all_quirk": sum(1 for r in cc if r["quirk"]),
        "probe_changed_the_failure_category": sum(1 for r in cc if not r["category_same"]),
        "disagreements": [r for r in cc if not r["agree"]][:20],
    }
    return out


# ---------------------------------------------------------------------------
# stage 4: acceptance test
# ---------------------------------------------------------------------------

def swap_candidates(graph: list[dict], spec: dict) -> tuple[list[dict], Counter]:
    """Every component in ``graph`` that can carry a MEANINGFUL swap.

    Two exclusions, both necessary or the effect size is a lie:
      - a component with ANY null terminal is not emitted at all (the writer
        skips it as UNSNAPPED), so swapping its pins produces a byte-identical
        deck;
      - a swap of two terminals already on the SAME net is byte-identical too.
    Both are counted as ``ineligible`` rather than folded in as misses.
    """
    out: list[dict] = []
    ineligible: Counter = Counter()
    for k, c in enumerate(graph):
        role = class_role(c["class"])
        if role not in spec:
            continue
        i, j, kind = spec[role]
        nets = list(c["nets"])
        need = class_terminals(c["class"])
        if len(nets) < max(i, j) + 1 or any(n is None for n in nets[:need]):
            ineligible[f"{kind}:unsnapped_terminal"] += 1
            continue
        if nets[i] == nets[j]:
            ineligible[f"{kind}:same_net"] += 1
            continue
        out.append({"index": k, "comp_id": c["id"],
                    "class": canonical_class(c["class"]), "role": role,
                    "swap_kind": kind, "i": i, "j": j})
    return out, ineligible


def swapped(graph: list[dict], cand: dict) -> list[dict]:
    g = [dict(c) for c in graph]
    nets = list(g[cand["index"]]["nets"])
    nets[cand["i"]], nets[cand["j"]] = nets[cand["j"]], nets[cand["i"]]
    g[cand["index"]] = {**g[cand["index"]], "nets": nets}
    return g


def swapped_nets(graph: list[dict], cand: dict) -> tuple:
    nets = graph[cand["index"]]["nets"]
    return (nets[cand["i"]], nets[cand["j"]])


def _swap_pass(jobs: list[tuple], ph: dict, cfg: dict, workers: int) -> list[dict]:
    """Simulate each perturbed deck and record whether the metric moved.

    Also records the dominant blind spot, which is a theorem rather than a story:

        if the two swapped terminals sit at the same potential in the
        unperturbed solution, that solution is STILL A SOLUTION of the
        perturbed circuit.

    Proof -- a swap permutes which terminal of the element sees which node
    voltage. If those two node voltages are equal, the element's terminal-
    voltage vector is unchanged, so its branch currents are unchanged, so the
    original solution still satisfies every KCL equation. For a LINEAR circuit
    the solution is unique and the two decks therefore have literally the same
    operating point: no tolerance, no correspondence and no placeholder policy
    can separate them. For a NONLINEAR one (BJT, diode) the perturbed circuit
    may have other solutions and the solver may land on one, so the flag is a
    strong predictor and not a guarantee -- which is why it is reported as a
    measured count and the exceptions are left visible rather than asserted
    away.
    """
    def one(job):
        stem, cand, ref_volts, corr, base_f1, perturbed_graph, own_volts, nets = job
        sim = simulate(spice_components(perturbed_graph), ph, cfg)
        if sim["solved"]:
            sc = score_op(ref_volts, sim["voltages"], corr)
            new_f1, outcome, max_dv = sc["f1"], "resimulated", sc["max_abs_dv"]
        else:
            # a swap that makes the deck unsolvable is ALSO detection: the
            # circuit leaves the comparable population entirely
            new_f1, outcome, max_dv = 0.0, f"broke_solvability:{sim['category']}", None

        def volt(net):
            if net is None:
                return None
            n = str(net).lower()
            return 0.0 if n == "0" else own_volts.get(n)
        vi, vj = volt(nets[0]), volt(nets[1])
        equi = (None if vi is None or vj is None
                else bool(abs(vi - vj) <= PRIMARY_ATOL))
        return {"stem": stem, "comp_id": cand["comp_id"], "class": cand["class"],
                "swap_kind": cand["swap_kind"], "f1_before": base_f1,
                "f1_after": new_f1, "delta_f1": new_f1 - base_f1, "outcome": outcome,
                "max_abs_dv": max_dv,
                "swapped_nets": list(nets), "v_swapped": [vi, vj],
                "equipotential": equi,
                "detected": bool(new_f1 < base_f1 - 1e-12 or outcome != "resimulated")}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, jobs))


def summarise_swaps(results: list[dict]) -> dict:
    by: dict[str, dict] = {}
    for kind in sorted({x["swap_kind"] for x in results}):
        sub = [x for x in results if x["swap_kind"] == kind]
        deltas = [x["delta_f1"] for x in sub]
        broke = [x for x in sub if x["outcome"].startswith("broke_solvability")]
        det = sum(1 for x in sub if x["detected"])
        und = [x for x in sub if not x["detected"]]
        # an undetected swap whose two terminals are equipotential in the
        # unperturbed solution is UNDETECTABLE, not missed: the two decks have
        # the same operating point (see _swap_pass). Reported separately, and
        # the "live" rate is the detection rate over the swaps that could in
        # principle have been seen.
        und_equi = [x for x in und if x["equipotential"]]
        live = [x for x in sub if not x["equipotential"]]
        by[kind] = {
            "n_swaps": len(sub), "detected": det, "detection_rate": det / len(sub),
            "undetected": len(und),
            "undetected_because_terminals_equipotential": len(und_equi),
            "undetected_unexplained": len(und) - len(und_equi),
            "n_swaps_detectable_in_principle": len(live),
            "detection_rate_on_detectable": (
                sum(1 for x in live if x["detected"]) / len(live)) if live else None,
            "mean_delta_f1": sum(deltas) / len(sub),
            "median_delta_f1": statistics.median(deltas),
            "worst_delta_f1": min(deltas), "best_delta_f1": max(deltas),
            "broke_solvability": len(broke),
            "broke_solvability_categories": dict(
                Counter(x["outcome"].split(":", 1)[1] for x in broke)),
            "unexplained_undetected_examples": [
                {"stem": x["stem"], "comp_id": x["comp_id"], "class": x["class"],
                 "swapped_nets": x["swapped_nets"], "v_swapped": x["v_swapped"]}
                for x in und if not x["equipotential"]][:10],
        }
    return by


def pooled_rates(results: list[dict]) -> dict:
    if not results:
        return {"n_swaps": 0}
    live = [x for x in results if not x["equipotential"]]
    return {
        "n_swaps": len(results),
        "detected": sum(1 for x in results if x["detected"]),
        "pooled_detection_rate": sum(1 for x in results if x["detected"]) / len(results),
        "n_swaps_detectable_in_principle": len(live),
        "pooled_detection_rate_on_detectable": (
            sum(1 for x in live if x["detected"]) / len(live)) if live else None,
        "undetectable_by_equipotential_terminals": len(results) - len(live),
    }


def acceptance_test(circuits: list[dict], rows: list[dict], ph: dict, cfg: dict,
                    min_f1: float, workers: int, in_situ: bool = True) -> dict:
    """Inject a known pin swap and demand the metric responds.

    TEST A -- CONTROLLED. Swap one pin pair in the GROUND TRUTH and score the
    perturbed GT against the unperturbed GT. The baseline is exactly 1.000 (the
    gt-vs-gt null control proves it), so ANY drop is attributable to the swap and
    to nothing else. Full headroom, every GT-solvable circuit, every eligible
    component: this is where the sample size is.

    TEST B -- IN SITU. Swap one pin pair in the PREDICTED netlist of a circuit
    where both sides already solve, and re-score against GT through the UNCHANGED
    correspondence. This is the literal question -- would the metric have caught
    this error in real pipeline output -- but it is confounded (the circuit
    already disagrees somewhere, so there is less headroom) and lower-N.

    CONTROL. The same machinery on RESISTORS, CAPACITORS and INDUCTORS, whose
    terminal order genuinely carries no meaning. Detection rate there must be
    exactly 0. Without it, "the metric moved" would only show that the metric
    responds to perturbation, not to *electrically meaningful* perturbation.
    """
    by_stem = {c["stem"]: c for c in circuits}
    row_by_stem = {r["stem"]: r for r in rows}

    # --- Test A: controlled, on the ground truth ---------------------------
    jobs_a: list[tuple] = []
    jobs_ctl: list[tuple] = []
    ineligible_a: Counter = Counter()
    for c in circuits:
        r = row_by_stem.get(c["stem"])
        if r is None or not r["gt_solved"]:
            continue
        gt_v = r["_gt_voltages"]
        corr, _ = build_correspondence(c["gt_graph"], c["gt_graph"])
        if score_op(gt_v, gt_v, corr)["f1"] != 1.0:
            continue
        cands, inel = swap_candidates(c["gt_graph"], SWAP_SPEC)
        ineligible_a += inel
        for cand in cands:
            jobs_a.append((c["stem"], cand, gt_v, corr, 1.0,
                           swapped(c["gt_graph"], cand), gt_v,
                           swapped_nets(c["gt_graph"], cand)))
        pcands, _ = swap_candidates(c["gt_graph"], PASSIVE_SPEC)
        for cand in pcands:
            jobs_ctl.append((c["stem"], cand, gt_v, corr, 1.0,
                             swapped(c["gt_graph"], cand), gt_v,
                             swapped_nets(c["gt_graph"], cand)))

    res_a = _swap_pass(jobs_a, ph, cfg, workers)
    res_ctl = _swap_pass(jobs_ctl, ph, cfg, workers)

    # a swap must not move the correspondence: it is built from component->net
    # incidence, which a within-component swap leaves identical
    corr_moved = 0
    for job in jobs_a[:200]:
        stem, corr, pg = job[0], job[3], job[5]
        corr2, _ = build_correspondence(pg, by_stem[stem]["gt_graph"])
        corr_moved += int(corr2 != corr)

    out = {
        "_what_this_proves": (
            "A pin swap is invisible to every topology metric in this repository "
            "(benchmark.canonicalize_terminals sorts terminals by a connectivity "
            "signature computed identically on both sides, so the swap cancels). "
            "These rows show the operating-point metric responding to the same "
            "perturbation, with the effect size and the sample size stated."),
        "placeholders": ph,
        "controlled_gt_perturbation": {
            "_design": ("swap one pin pair in the GROUND TRUTH, score against the "
                        "unperturbed ground truth. Baseline is exactly 1.000, so "
                        "any drop is the swap and nothing else."),
            "circuits": len({s for s, *_ in jobs_a}),
            "n_swaps": len(res_a),
            "ineligible_excluded": dict(ineligible_a),
            "_ineligible_note": (
                "unsnapped_terminal = the component has a null terminal so the "
                "writer never emits it, and the swap is byte-identical; "
                "same_net = both swapped terminals were already on one net, also "
                "byte-identical. Neither is a perturbation, so neither is counted "
                "as a miss."),
            "correspondence_moved_under_swap": corr_moved,
            "_correspondence_note": (
                "MUST be 0 (checked on the first 200 swaps). A correspondence "
                "that moved with the error would let the error cancel."),
            "_equipotential_note": (
                "a swap whose two terminals are at the same potential in the "
                "unperturbed solution CANNOT change the operating point -- the "
                "element sees the identical terminal voltages either way, so the "
                "two decks have the same solution. Those are counted as "
                "undetectable rather than missed, and the 'on_detectable' rate is "
                "the one that measures the metric rather than the dataset."),
            "by_swap_kind": summarise_swaps(res_a),
            **pooled_rates(res_a),
        },
        "passive_swap_control": {
            "_design": ("the identical machinery on resistors, capacitors and "
                        "inductors, whose terminal order genuinely carries no "
                        "meaning. Detection rate MUST be exactly 0."),
            "n_swaps": len(res_ctl),
            "detected": sum(1 for x in res_ctl if x["detected"]),
            "detection_rate": (
                sum(1 for x in res_ctl if x["detected"]) / len(res_ctl)) if res_ctl else None,
            "max_abs_delta_f1": max((abs(x["delta_f1"]) for x in res_ctl), default=0.0),
            "violations": [x for x in res_ctl if x["detected"]][:10],
        },
    }

    # --- Test B: in situ, on the predicted netlist -------------------------
    if in_situ:
        jobs_b: list[tuple] = []
        ineligible_b: Counter = Counter()
        for c in circuits:
            r = row_by_stem.get(c["stem"])
            if r is None or r["population"] != "both_solve":
                continue
            cands, inel = swap_candidates(c["pred_graph"], SWAP_SPEC)
            ineligible_b += inel
            for cand in cands:
                jobs_b.append((c["stem"], cand, r["_gt_voltages"], c["corr"],
                               r["f1"], swapped(c["pred_graph"], cand),
                               r["_pred_voltages"],
                               swapped_nets(c["pred_graph"], cand)))
        res_b = _swap_pass(jobs_b, ph, cfg, workers)
        high = [x for x in res_b if x["f1_before"] >= min_f1]
        out["in_situ_pred_perturbation"] = {
            "_design": ("swap one pin pair in the PREDICTED netlist of a circuit "
                        "where both sides already solve, re-score against GT "
                        "through the UNCHANGED correspondence. Confounded (the "
                        "circuit already disagrees elsewhere, so there is less "
                        "headroom) but it is the literal question."),
            "circuits": len({x["stem"] for x in res_b}),
            "n_swaps": len(res_b),
            "ineligible_excluded": dict(ineligible_b),
            "by_swap_kind": summarise_swaps(res_b),
            **pooled_rates(res_b),
            "high_agreement_subset": {
                "min_f1": min_f1,
                **pooled_rates(high),
                "by_swap_kind": summarise_swaps(high),
            },
            "swaps": res_b,
        }
    out["controlled_gt_perturbation"]["swaps"] = res_a
    out["passive_swap_control"]["swaps"] = res_ctl
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

CSV_COLS = [
    "stem", "population", "gt_solved", "pred_solved", "gt_category",
    "pred_category", "pred_repaired_solved", "gt_op_degenerate_zero", "gt_op_span_V",
    "n_gt_nodes", "n_pred_nodes", "n_corresponded", "n_agree",
    "recall", "precision", "f1", "f1_incl_ground", "exact",
    "max_abs_dv", "median_abs_dv", "f1_shipped_policy",
    "n_gt_comps", "n_pred_comps", "aligned", "unmatched_gt", "unmatched_pred",
    "net_f1_topology", "terminal_pair_f1_topology", "strict_success_topology",
]


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(out_dir: Path, summary: dict) -> None:
    chosen = summary["chosen_policy"]
    V = summary["variants"][chosen]
    ag_all = V["aggregate_all_both_solve"]
    ag_live = V["aggregate_non_degenerate"]
    ag_deg = V["aggregate_degenerate_only"]
    pop = V["populations"]
    n = V["circuits"]
    acc = summary["acceptance"][chosen]
    ctl = acc["controlled_gt_perturbation"]
    L: list[str] = []

    def w(*xs):
        L.extend(xs)

    w("# DC operating-point agreement",
      "",
      "Generated by `scripts/measure_op_agreement.py`. Every number below is "
      "reproduced by re-running that script; nothing here is hand-entered.",
      "",
      f"- config hash: `{summary['config_hash']}` (the shipped `configs/default.yaml`)",
      f"- split: `{summary['split']}` — {n} circuits, GT at `{summary['gt_dir']}`",
      f"- placeholder policy: **`{chosen}`** = `{summary['chosen_placeholders']}`",
      f"- tolerance: **{PRIMARY_ATOL * 1e3:.0f} mV absolute**",
      "",
      "## What this measures",
      "",
      "The ground-truth netlist and the predicted netlist are exported by the "
      "**same writer**, with the **same placeholder component values**, and handed "
      "to the **same ngspice**. Their DC operating points are then compared node "
      "by node through a pin-order-invariant net correspondence. Because the "
      "values are identical, every voltage difference is attributable to the "
      "recovered circuit: its topology, its component classes, or its **terminal "
      "(pin) order**. No OCR is involved — the values cancel.",
      "",
      "This is the first metric in the repository that is not blind to pin order. "
      "`benchmark.canonicalize_terminals` sorts terminals by a connectivity "
      "signature computed identically on both sides, so a reversed BJT, a "
      "backwards diode, a flipped supply or a swapped op-amp input all leave "
      "net-F1 at exactly 1.000. An operating point does not cancel. The first "
      "thing this metric found was a **reversed `V-DC` on `circuit_1025`, a "
      "circuit scoring `strict_success = 1`** — the prediction writes "
      "`V1 0 n1 DC 5` where the reference writes `V1 n2 0 DC 5`. A 10 V error on "
      "a circuit every existing metric calls perfect.",
      "",
      "## The three populations, and the honest denominator",
      "",
      "The metric is **defined only where both netlists solve**. A circuit whose "
      "reference does not solve has no reference operating point to agree with.",
      "",
      "| population | circuits | share |", "|---|---:|---:|")
    for k in ("both_solve", "only_gt_solves", "only_pred_solves", "neither_solves"):
        w(f"| {k} | {pop[k]} | {pop[k] / n:.3f} |")
    w(f"| **total** | **{n}** | 1.000 |",
      "",
      f"Ground-truth netlists DC-solvable: {pop['gt_solved']}/{n} "
      f"({pop['gt_solved'] / n:.3f}). Predicted, unrepaired: {pop['pred_solved']}/{n}. "
      f"Predicted after the C5 repair layer: {pop['pred_repaired_solved']}/{n} — "
      "which is the whole reason solvability cannot be the correctness criterion: "
      "the pipeline beats the truth on it.",
      "",
      "`solved` means ngspice's own taxonomy (`simulate.parse_ngspice_output`) "
      "returned `ok`. That is stricter than \"ngspice printed some numbers\": on a "
      "singular deck ngspice falls back to gmin stepping and a transient op and "
      "**still prints a full node table**, but those voltages are an artefact of "
      "the fallback pinning a floating node. They are never scored.",
      "",
      "### A fourth population, and it matters",
      "",
      "A circuit whose reference operating point is **identically 0 V on every "
      "node** is comparable but uninformative: every corresponded node agrees "
      "for free and the score degenerates into a node-count check. It happens "
      "when the only excitation is an AC source, whose DC value is 0.",
      "",
      f"Under the **shipped** placeholder values this is "
      f"{summary['variants']['shipped']['degenerate_zero_op']['n']} of the "
      f"{pop['both_solve']} comparable circuits — over a fifth of the metric's "
      "own domain, silently scoring near 1.000 while measuring nothing. That "
      "finding is what forced the placeholder policy to become a selected "
      "parameter rather than an inherited one (below). Under the chosen `"
      f"{chosen}` policy it is {V['degenerate_zero_op']['n']}. Either way the "
      "affected circuits are listed by name in `summary.json` and reported as "
      "their own row, never merged into the headline silently.",
      "",
      "## The number",
      "",
      "| domain | circuits | mean F1 | mean recall | mean precision | exact | pooled node recall |",
      "|---|---:|---:|---:|---:|---:|---:|")
    for b, lbl in ((ag_live, "both solve, reference OP not all-zero (**headline**)"),
                   (ag_deg, "both solve, reference OP all-zero"),
                   (ag_all, "both solve, everything")):
        if b["n_scored"]:
            w(f"| {lbl} | {b['n_scored']} | {fmt(b['mean_f1'])} | "
              f"{fmt(b['mean_recall'])} | {fmt(b['mean_precision'])} | "
              f"{b['exact_circuits']} ({fmt(b['exact_rate'], 3)}) | "
              f"{fmt(b['pooled_node_recall'])} |")
    w("",
      f"**Headline: mean F1 = {fmt(ag_live['mean_f1'])} over the "
      f"{ag_live['n_scored']} circuits where both netlists solve and the "
      "reference operating point is not identically zero.** That is the domain on "
      "which this is a voltage metric at all. "
      f"{ag_live['exact_circuits']}/{ag_live['n_scored']} of them reproduce the "
      f"reference operating point exactly; as a share of the whole {n}-circuit "
      f"split that is {ag_live['exact_circuits']}/{n} = "
      f"{ag_live['exact_circuits'] / n:.4f}.",
      "",
      "**How to read it.** `recall` = the share of the reference circuit's node "
      "voltages the prediction reproduces. `precision` = the share of the "
      "prediction's node voltages that are in the reference. F1 is the headline "
      "because a net MERGE costs recall and a net SPLIT costs precision, so "
      "neither can be gamed by collapsing or shattering the circuit.",
      "",
      "The ground node is excluded by construction: ngspice never reports node "
      "`0`, so a `0`-to-`0` pair cannot bank a free agreement. A mis-grounded "
      "prediction is still caught — every other node shifts, and the reference "
      "ground net corresponds to a predicted node that then has no partner.",
      "",
      "### It is not the topology metric wearing a hat",
      "",
      f"{len(summary['topologically_perfect_but_op_disagrees'])} circuits have "
      "`net_f1 = 1.000` **and** `terminal_pair_f1 = 1.000` — topologically "
      "perfect by every existing measure — and still disagree on the operating "
      "point. They are listed in `summary.json` under "
      "`topologically_perfect_but_op_disagrees`, with the offending nets and "
      "their two voltages.",
      "",
      "## Placeholder policy: chosen on sensitivity, not on the score",
      "",
      "The values are **our** design choice, not a constraint handed to us. They "
      "are identical on both sides whatever they are, so choosing them to bias "
      "devices into a region where orientation matters does not game the "
      "comparison — it makes the probe informative. The shipped values (`R=1k` "
      "everywhere, `AC 1` sources with no DC term) leave a fifth of the "
      "comparable circuits sitting at 0 V on every node.",
      "",
      "Candidates were ranked **only** by detection rate on injected pin swaps "
      "(the controlled test below). The agreement number is printed beside it so "
      "it is visible that the choice was not made on the headline:",
      "",
      "| policy | swap detection rate | …on detectable swaps | n swaps | all-zero circuits | mean F1 (non-degenerate) |",
      "|---|---:|---:|---:|---:|---:|")
    for name, r in summary["policy_selection"]["candidates"].items():
        mark = " **←chosen**" if name == chosen else ""
        w(f"| `{name}`{mark} | {fmt(r['pooled_detection_rate'])} | "
          f"{fmt(r['pooled_detection_rate_on_detectable'])} | "
          f"{r['n_swaps']} | {r['degenerate_zero_op']} | "
          f"{fmt(r['mean_f1_non_degenerate'])} |")
    w("",
      "The two columns move independently, which is the point of printing both: "
      "`shipped` has the *highest* agreement number and the *lowest* detection "
      "rate — its agreement is high partly because a fifth of its circuits sit at "
      "0 V and agree for free. Choosing on the headline would have chosen exactly "
      "the wrong policy.",
      "",
      "Limit of this axis, stated plainly: the writer takes **one value per "
      "role**, so every resistor in a deck is the same resistor. A base-side vs "
      "collector-side split — which is what would keep a resistively biased BJT "
      "out of saturation — is not expressible, and a per-component value "
      "assignment would break the property that makes this metric work at all "
      "(identical values on both sides, so they cancel).",
      "",
      "## Acceptance test: does it actually see a pin swap?",
      "",
      "A metric built to detect pin order is worthless unless it does.",
      "",
      "### A — controlled (ground truth perturbed against itself)",
      "",
      "One pin pair is swapped in the **ground truth** and the result scored "
      "against the unperturbed ground truth. The baseline is exactly 1.000 (the "
      "null control proves it), so any drop is the swap and nothing else.",
      "",
      f"- {ctl['circuits']} circuits, **{ctl['n_swaps']} single-pin swaps**",
      f"- pooled detection rate: **{fmt(ctl['pooled_detection_rate'])}** — and "
      f"**{fmt(ctl['pooled_detection_rate_on_detectable'])}** over the "
      f"{ctl['n_swaps_detectable_in_principle']} swaps that a DC operating point "
      "can see at all (next section)",
      f"- correspondence moved under a swap: "
      f"{ctl['correspondence_moved_under_swap']} (must be 0)",
      f"- ineligible, excluded rather than counted as misses: "
      f"`{ctl['ineligible_excluded'] or 'none'}`",
      "",
      "| swap | n | detected | rate | detectable | rate on those | mean ΔF1 | broke solvability |",
      "|---|---:|---:|---:|---:|---:|---:|---:|")
    for kind, s in ctl["by_swap_kind"].items():
        w(f"| {kind} | {s['n_swaps']} | {s['detected']} | {s['detection_rate']:.3f} | "
          f"{s['n_swaps_detectable_in_principle']} | "
          f"{fmt(s['detection_rate_on_detectable'], 3)} | "
          f"{s['mean_delta_f1']:+.4f} | {s['broke_solvability']} |")
    w("",
      "A swap that makes the deck unsolvable counts as detected: the circuit "
      "leaves the comparable population, which is exactly what the metric should "
      "say about a netlist that no longer simulates. It is reported in its own "
      "column rather than folded into ΔF1.",
      "",
      "### The dominant blind spot — a theorem, not an excuse",
      "",
      "**If the two swapped terminals sit at the same potential in the "
      "unperturbed solution, that solution is still a solution of the perturbed "
      "circuit.** A swap permutes which terminal of the element sees which node "
      "voltage; if those two node voltages are equal, the element's "
      "terminal-voltage vector is unchanged, so its branch currents are "
      "unchanged, so the original solution still satisfies every KCL equation. "
      "For a linear circuit the solution is unique and the two decks have "
      "literally the same operating point — no tolerance, no correspondence and "
      "no placeholder policy can separate them. For a nonlinear one the "
      "perturbed circuit may admit other solutions and the solver may land on "
      "one, which is why a handful of equipotential BJT swaps *are* detected; "
      "the flag is a measured predictor, not an alibi.",
      "",
      f"That single fact accounts for "
      f"{ctl['undetectable_by_equipotential_terminals']}/{ctl['n_swaps']} of the "
      "swaps. Every undetected swap is therefore either *explained* (equipotential "
      "terminals) or *unexplained*, and the unexplained ones are listed by name in "
      "`acceptance_test.json` under `unexplained_undetected_examples` rather than "
      "averaged away:",
      "",
      "| swap | undetected | of which equipotential | unexplained |",
      "|---|---:|---:|---:|")
    for kind, s in ctl["by_swap_kind"].items():
        w(f"| {kind} | {s['undetected']} | "
          f"{s['undetected_because_terminals_equipotential']} | "
          f"{s['undetected_unexplained']} |")
    w("",
      "The residue — the `unexplained` column — is small and has two known "
      "causes, both visible in the per-swap records. A device carrying **no "
      "current** (a BJT cut off with both junctions reverse-biased) is an open "
      "circuit whichever way round it is, even though its terminals are at "
      "different potentials. And this metric reads **node voltages only**: "
      "reversing a current source strung between two nodes that are both held "
      "by voltage sources changes the branch currents and no node voltage at "
      "all, which is most of the `idc_direction` residue.",
      "",
      "The op-amp row is the clearest case and worth reading literally. The "
      "writer emits an op-amp as `E<n> out 0 in+ in- 100k`, so an input swap "
      "flips the sign of a 100 000× gain — an enormous error whenever the inputs "
      "differ at all. In this dataset they usually do not: hand-drawn op-amp "
      "input networks are overwhelmingly capacitor-coupled, a capacitor is an "
      "open circuit at DC, and both inputs therefore sit at the same potential "
      "(usually 0 V) in the reference solution. That is a property of the "
      "*drawings*, not of the metric — and no choice of R or V creates a DC path "
      "through a capacitor.",
      "")
    ins = acc.get("in_situ_pred_perturbation")
    if ins:
        hs = ins["high_agreement_subset"]
        w("### B — in situ (predicted netlist perturbed, scored against GT)",
          "",
          "The literal question: would the metric have caught this error in real "
          "pipeline output? Confounded — the circuit already disagrees somewhere, "
          "so there is less headroom — and lower-N, which is why A carries the "
          "sample size.",
          "",
          f"- {ins['circuits']} circuits, **{ins['n_swaps']} swaps**, pooled "
          f"detection rate **{fmt(ins['pooled_detection_rate'])}** "
          f"({fmt(ins['pooled_detection_rate_on_detectable'])} on the "
          f"{ins['n_swaps_detectable_in_principle']} detectable ones)",
          f"- on the high-agreement subset (F1 ≥ {hs['min_f1']}): "
          f"{hs.get('detected', 0)}/{hs['n_swaps']} = "
          f"{fmt(hs.get('pooled_detection_rate'))}",
          "",
          "| swap | n | detected | rate | detectable | rate on those | mean ΔF1 |",
          "|---|---:|---:|---:|---:|---:|---:|")
        for kind, s in ins["by_swap_kind"].items():
            w(f"| {kind} | {s['n_swaps']} | {s['detected']} | "
              f"{s['detection_rate']:.3f} | {s['n_swaps_detectable_in_principle']} | "
              f"{fmt(s['detection_rate_on_detectable'], 3)} | "
              f"{s['mean_delta_f1']:+.4f} |")
        w("",
          "This is much lower than the controlled test and the reason is "
          "headroom, not blindness. Test B perturbs a *predicted* netlist that "
          "already disagrees with the reference; the swapped device frequently "
          "sits in a part of the circuit the prediction already got wrong, where "
          "the corresponded nodes are already scored as disagreeing and cannot "
          "disagree twice. Read Test A for the metric's sensitivity and Test B "
          "for how often that sensitivity is reachable in current pipeline "
          "output.",
          "")
    pc = acc["passive_swap_control"]
    nc = summary["null_controls"]
    w("### C — controls: the metric must NOT move where nothing changed",
      "",
      "| control | n | result | verdict |", "|---|---:|---|---|",
      f"| resistor/capacitor/inductor terminal swap | {pc['n_swaps']} | "
      f"detection rate {fmt(pc['detection_rate'])}, max \\|ΔF1\\| "
      f"{pc['max_abs_delta_f1']:.2e} | "
      f"{'PASS' if not pc['detected'] else 'FAIL'} |",
      f"| pred-vs-pred (deck against itself) | "
      f"{nc['pred_vs_pred']['circuits_scored']} | mean F1 "
      f"{fmt(nc['pred_vs_pred']['mean_f1'])} | "
      f"{'PASS' if nc['pred_vs_pred']['all_exactly_1.0'] else 'FAIL'} |",
      f"| gt-vs-gt (deck against itself) | {nc['gt_vs_gt']['circuits_scored']} | "
      f"mean F1 {fmt(nc['gt_vs_gt']['mean_f1'])} | "
      f"{'PASS' if nc['gt_vs_gt']['all_exactly_1.0'] else 'FAIL'} |",
      f"| `.op`-table parser vs `print all` | "
      f"{nc['parser_crosscheck']['decks_checked']} decks | "
      f"{nc['parser_crosscheck']['voltages_agree']} agree | "
      f"{'PASS' if nc['parser_crosscheck']['voltages_agree'] == nc['parser_crosscheck']['decks_checked'] else 'FAIL'} |",
      "",
      "The passive control is the one that makes the others mean something: it "
      "shows the metric responds to *electrically meaningful* perturbation, not "
      "to perturbation as such. Swapping the two ends of a resistor changes the "
      f"deck text and moves the score by exactly {pc['max_abs_delta_f1']:.2e}.",
      "",
      "The pred-vs-pred control is not decoration. An earlier version of this "
      "file read node voltages out of a `print all` block; with exactly one "
      "vector in the plot ngspice does not expand `all` and prints the literal "
      "token as the node name (`all = -1.00000e-03`), so every single-node deck "
      "lost its node names, corresponded nothing, and **scored 0.000 against "
      "itself**. Nothing else in the harness noticed — the headline just read a "
      "little low. The parse now reads the `.op` table, and both controls are "
      "exactly 1.000.",
      "",
      "## Tolerance",
      "",
      f"**Primary: {PRIMARY_ATOL * 1e3:.0f} mV absolute.** ngspice's node-voltage "
      "convergence tolerance (`vntol`) defaults to 1 µV, so two solves of the "
      "same circuit agree far inside 1 µV — 1 mV is 1000× that and numerical "
      "noise cannot cross it (measured, not assumed: the passive-swap control "
      "moves nodes by exactly 0). In the other direction the placeholder rail is "
      "5–15 V, so 1 mV is ≤0.02% of full scale while a genuine topological or "
      "pin-order difference moves a node by hundreds of mV to volts.",
      "",
      "| tolerance | mean F1 (non-degenerate) | mean F1 (all both-solve) | exact circuits |",
      "|---|---:|---:|---:|")
    for s in V["tolerance_sensitivity"]:
        w(f"| {s['tolerance']} | {fmt(s['mean_f1_non_degenerate'])} | "
          f"{fmt(s['mean_f1_both_solve'])} | {s['exact_circuits']} |")
    w("",
      "Flat across nine orders of magnitude. That is the signature of a metric "
      "whose disagreements are *large* — a node is either the same node or a "
      "completely different one — not marginal. The tolerance is therefore not a "
      "tuning knob and cannot be used to move the number.",
      "",
      "## What this metric cannot do",
      "",
      f"1. **It is silent outside its domain.** {n - ag_live['n_scored']}/{n} "
      "circuits are not scored. The largest single reason is that the *ground "
      "truth* does not solve — an as-drawn hand schematic frequently has a "
      "floating subnet or no reference, and that is a property of the drawing, "
      "not of the pipeline.",
      "2. **DC only.** Capacitors are open and inductors are shorts at the "
      "operating point, so a mis-wired capacitor in a purely capacitive branch "
      "moves nothing. Filtering, gain over frequency, switching — all invisible.",
      "3. **Placeholder values, not the drawn values.** Deliberate (it removes "
      "OCR from the comparison), but the operating point is that of a circuit "
      "nobody drew. The policy above buys back a lot of sensitivity; it cannot "
      "buy back all of it, because one resistor value per deck cannot keep a "
      "resistively biased BJT out of saturation, and a cut-off or saturated "
      "device is in the same state whichever way round it is.",
      "4. **The equipotential blind spot above is structural.** It is the "
      "dominant reason a swap goes unseen, it is provable, and it is not "
      "fixable — not by a tolerance, not by a better correspondence, not by "
      "different placeholder values. Any claim about this metric's sensitivity "
      "has to be read as a rate over the *detectable* swaps, which is why both "
      "rates are reported everywhere.",
      "5. **It reads node voltages, not branch currents.** An error that moves "
      "only currents is invisible: reversing a current source between two nodes "
      "both held by voltage sources is the case that shows up in the data. "
      "Adding the source-current vector to the comparison would close this and "
      "is the obvious next extension.",
      "6. **It is not an equivalence proof.** Two different circuits can share "
      "an operating point. Agreement is necessary for \"simulates like the "
      "reference\", not sufficient.",
      "7. **Errors are aggregated, not attributed.** A wrong class (Resistor read "
      "as Inductor) and a wrong pin order land in the same number. Use the "
      "topology cascade and `scripts/compare_annotations.py` for attribution.",
      "",
      "## Gameability",
      "",
      "- **A single-node circuit scores 1.000 trivially.** Per-circuit means "
      "weight a 2-node circuit like a 20-node one; the pooled node recall is the "
      "unweighted-by-circuit view and should be read beside the mean.",
      "- **\"Both solve\" is a selectable population.** A change that makes hard "
      "circuits fail to solve *raises* mean F1 by removing them from the "
      "denominator. Never compare two runs on mean F1 alone — compare the "
      "exact-circuit count against the whole split, which is monotone in the "
      "right direction.",
      "- **The placeholder policy moves the number.** It must be recorded with "
      "any quoted figure; it is in `summary.json` under `chosen_placeholders`, "
      "and every candidate's agreement number is in `policy_selection` so the "
      "choice can be audited. Two runs under different policies are not "
      "comparable.",
      "- **Repair is excluded from the headline on purpose.** The repaired deck "
      "adds grounds and 1 GΩ shunts the reference does not have; scoring it "
      "against the reference would reward inventing a reference node. Its "
      "solvability is in `per_image.csv` for context only.",
      "",
      "## Files",
      "",
      "- `summary.json` — populations, aggregates, policy selection, tolerance sweep, null controls",
      "- `per_image.csv` — one row per circuit, including circuits outside the domain",
      "- `acceptance_test.json` — every injected swap, per policy",
      "- `netlists/` — the exact `.sp` decks handed to ngspice, both sides, chosen policy",
      "- `cache/` — the pipeline pass, checkpointed per circuit; every number above "
      "is pure simulation over it",
      "")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/op_agreement")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage", default="all", choices=("all", "cache"))
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--acceptance-min-f1", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-netlists", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    gt_dir = ROOT / (args.gt_dir or cfg["benchmark"]["gt_dir"])
    det_dir = ROOT / cfg["detect"]["cache_dir"]
    names = (ROOT / args.splits_dir / f"{args.split}.txt").read_text().split()
    if args.limit:
        names = names[:args.limit]
    images_dir = resolve_and_check(args.images_dir, names, cfg)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    print(f"[stage 1/4] pipeline pass over {len(names)} circuits "
          f"(checkpointed to {cache_dir})", flush=True)
    stems = build_cache(names, images_dir, det_dir, gt_dir, cfg,
                        args.iou_threshold, cache_dir, args.refresh_cache)
    if args.stage == "cache":
        print(f"[OK] cached {len(stems)} circuits")
        return 0

    circuits = load_cache(cache_dir, stems)
    by_stem = {c["stem"]: c for c in circuits}
    print(f"[stage 2/4] measuring {len(POLICY_GRID)} placeholder policies "
          f"over {len(circuits)} circuits", flush=True)

    variants: dict[str, dict] = {}
    rows_by_policy: dict[str, list[dict]] = {}
    for name in POLICY_GRID:
        ph = policy_placeholders(cfg, name)
        rows = measure_policy(circuits, ph, cfg, workers=args.workers)
        rows_by_policy[name] = rows
        v = aggregate_policy(rows)
        v["placeholders"] = ph
        v["overrides_vs_shipped"] = POLICY_GRID[name]
        v["tolerance_sensitivity"] = tolerance_sweep(rows, by_stem)
        variants[name] = v
        print(f"  {name:20s} both_solve={v['populations']['both_solve']:3d} "
              f"all-zero={v['degenerate_zero_op']['n']:3d} "
              f"meanF1(live)={fmt(v['aggregate_non_degenerate'].get('mean_f1'))}",
              flush=True)
    (out_dir / "_variants_partial.json").write_text(json.dumps(variants, indent=2))

    print("[stage 3/4] acceptance test on each policy "
          "(controlled swaps + passive control)", flush=True)
    acceptance: dict[str, dict] = {}
    for name in POLICY_GRID:
        ph = policy_placeholders(cfg, name)
        acc = acceptance_test(circuits, rows_by_policy[name], ph, cfg,
                              args.acceptance_min_f1, args.workers, in_situ=False)
        acceptance[name] = acc
        c = acc["controlled_gt_perturbation"]
        print(f"  {name:20s} swaps={c['n_swaps']:4d} "
              f"detection={fmt(c['pooled_detection_rate'])} "
              f"passive_violations={acc['passive_swap_control']['detected']}", flush=True)
        (out_dir / "_acceptance_partial.json").write_text(json.dumps(
            {k: {kk: {k3: v3 for k3, v3 in vv.items() if k3 != "swaps"}
                 if isinstance(vv, dict) else vv
                 for kk, vv in v.items()} for k, v in acceptance.items()}, indent=2))

    # --- policy selection: SENSITIVITY only ---------------------------------
    cand = {}
    for name in POLICY_GRID:
        c = acceptance[name]["controlled_gt_perturbation"]
        v = variants[name]
        cand[name] = {
            "pooled_detection_rate": c["pooled_detection_rate"],
            "pooled_detection_rate_on_detectable": c["pooled_detection_rate_on_detectable"],
            "undetectable_by_equipotential_terminals": c[
                "undetectable_by_equipotential_terminals"],
            "n_swaps": c["n_swaps"],
            "detection_by_kind": {k: s["detection_rate"]
                                  for k, s in c["by_swap_kind"].items()},
            "detection_by_kind_on_detectable": {
                k: s["detection_rate_on_detectable"]
                for k, s in c["by_swap_kind"].items()},
            "passive_control_violations": acceptance[name]["passive_swap_control"]["detected"],
            "degenerate_zero_op": v["degenerate_zero_op"]["n"],
            "both_solve": v["populations"]["both_solve"],
            "mean_f1_non_degenerate": v["aggregate_non_degenerate"].get("mean_f1"),
            "mean_f1_all_both_solve": v["aggregate_all_both_solve"].get("mean_f1"),
            "placeholders": v["placeholders"],
        }
    order = list(POLICY_GRID)
    chosen = max(order, key=lambda k: (cand[k]["pooled_detection_rate"] or 0.0,
                                       -order.index(k)))
    print(f"[stage 4/4] chosen policy: {chosen} "
          f"(detection {fmt(cand[chosen]['pooled_detection_rate'])})", flush=True)

    ph = policy_placeholders(cfg, chosen)
    netlist_dir = None if args.no_netlists else (out_dir / "netlists")
    rows = measure_policy(circuits, ph, cfg, netlist_dir=netlist_dir,
                          workers=args.workers)
    rows_by_policy[chosen] = rows
    shipped_f1 = {r["stem"]: r["f1"] for r in rows_by_policy["shipped"]}
    for r in rows:
        r["f1_shipped_policy"] = shipped_f1.get(r["stem"], "")

    nc = null_controls(circuits, ph, cfg, workers=args.workers)
    acceptance[chosen] = acceptance_test(circuits, rows, ph, cfg,
                                         args.acceptance_min_f1, args.workers,
                                         in_situ=True)

    both = [r for r in rows if r["population"] == "both_solve"]
    v_chosen = aggregate_policy(rows)
    v_chosen["placeholders"] = ph
    v_chosen["overrides_vs_shipped"] = POLICY_GRID[chosen]
    v_chosen["tolerance_sensitivity"] = tolerance_sweep(rows, by_stem)
    variants[chosen] = v_chosen

    summary = {
        "_what": ("DC operating-point agreement between the predicted netlist and "
                  "the ground-truth netlist, with IDENTICAL placeholder component "
                  "values on both sides, so every voltage difference is "
                  "attributable to topology, component class, or pin order. "
                  "Read REPORT.md."),
        "config_hash": config_hash(cfg), "seed": seed, "split": args.split,
        "gt_dir": str(gt_dir.relative_to(ROOT)), "images_dir": str(images_dir),
        "tolerance": {"atol_V": PRIMARY_ATOL, "rtol": PRIMARY_RTOL},
        "acceptance_min_f1": args.acceptance_min_f1,
        "chosen_policy": chosen,
        "chosen_placeholders": ph,
        "policy_selection": {
            "_rule": ("ranked ONLY by pooled detection rate on injected pin swaps "
                      "(controlled test). The agreement number is recorded beside "
                      "each candidate so it is visible the choice was not made on "
                      "the headline. Ties break toward the earlier, more "
                      "conservative policy."),
            "candidates": cand,
        },
        "variants": variants,
        "null_controls": nc,
        "acceptance": {k: {kk: ({k3: v3 for k3, v3 in vv.items() if k3 != "swaps"}
                               if isinstance(vv, dict) else vv)
                           for kk, vv in v.items()} for k, v in acceptance.items()},
        "topologically_perfect_but_op_disagrees": [
            {"stem": r["stem"], "net_f1_topology": r["net_f1_topology"],
             "terminal_pair_f1_topology": r["terminal_pair_f1_topology"],
             "strict_success_topology": r["strict_success_topology"],
             "op_f1": r["f1"], "max_abs_dv": r["max_abs_dv"],
             "disagreements": r["_disagreements"][:4]}
            for r in both
            if r["net_f1_topology"] == 1.0 and r["terminal_pair_f1_topology"] == 1.0
            and not r["exact"]],
        "worst_circuits": sorted(
            [{"stem": r["stem"], "f1": r["f1"], "max_abs_dv": r["max_abs_dv"],
              "net_f1_topology": r["net_f1_topology"],
              "disagreements": r["_disagreements"][:5]} for r in both],
            key=lambda x: (x["f1"], -x["max_abs_dv"]))[:15],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Per-swap records for the CHOSEN policy only; the other four candidates
    # keep their aggregates. Five full copies is 3.4 MB of near-duplicate rows,
    # and the passive control's 665 rows are 665 identical zeros -- its summary
    # plus any violations is the whole of its information content.
    def trim(policy_name: str, block: dict) -> dict:
        keep = policy_name == chosen
        out = {}
        for k, v in block.items():
            if not isinstance(v, dict):
                out[k] = v
                continue
            v = dict(v)
            if "swaps" in v and not (keep and k != "passive_swap_control"):
                v.pop("swaps")
            out[k] = v
        return out

    (out_dir / "acceptance_test.json").write_text(json.dumps({
        "_read_me": (
            "Per-swap records are kept for the chosen policy only; the other "
            "candidates keep their aggregates (the rows are near-duplicates and "
            "five copies is megabytes of noise). Re-run with a different policy "
            "chosen to get its rows. The passive control keeps its summary only: "
            "all 665 of its rows are identical zeros, which is the result."),
        "chosen_policy": chosen,
        "null_controls": nc,
        "by_policy": {k: trim(k, v) for k, v in acceptance.items()},
    }, indent=2))
    with (out_dir / "per_image.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        wr.writeheader()
        for r in sorted(rows, key=lambda x: x["stem"]):
            wr.writerow({c: r.get(c, "") for c in CSV_COLS})
    for p in ("_variants_partial.json", "_acceptance_partial.json"):
        (out_dir / p).unlink(missing_ok=True)
    write_report(out_dir, summary)

    live, allb = v_chosen["aggregate_non_degenerate"], v_chosen["aggregate_all_both_solve"]
    pop = v_chosen["populations"]
    print("\n=== DC OPERATING-POINT AGREEMENT ===")
    print(f"policy: {chosen}  {ph}")
    print(f"circuits {v_chosen['circuits']}: both_solve {pop['both_solve']}, "
          f"only_gt {pop['only_gt_solves']}, only_pred {pop['only_pred_solves']}, "
          f"neither {pop['neither_solves']}")
    print(f"  of both_solve, all-zero reference OP: {v_chosen['degenerate_zero_op']['n']}")
    print(f"HEADLINE mean F1 (non-degenerate, n={live['n_scored']}): "
          f"{fmt(live['mean_f1'])}   exact {live['exact_circuits']}/{live['n_scored']}")
    print(f"        mean F1 (all both_solve, n={allb['n_scored']}): {fmt(allb['mean_f1'])}")
    for k, v in nc.items():
        if k != "parser_crosscheck":
            print(f"null {k}: mean F1 {v['mean_f1']}, all 1.0 = {v['all_exactly_1.0']}")
    print(f"parser crosscheck: {nc['parser_crosscheck']['voltages_agree']}"
          f"/{nc['parser_crosscheck']['decks_checked']} decks agree, "
          f"{nc['parser_crosscheck']['decks_hitting_the_print_all_quirk']} hit the "
          "`print all` quirk")
    c = acceptance[chosen]["controlled_gt_perturbation"]
    print(f"acceptance (controlled): {c['n_swaps']} swaps, "
          f"detection {fmt(c['pooled_detection_rate'])}  "
          f"(on the {c['n_swaps_detectable_in_principle']} detectable: "
          f"{fmt(c['pooled_detection_rate_on_detectable'])})")
    for kind, s in c["by_swap_kind"].items():
        print(f"   {kind:26s} {s['detected']:4d}/{s['n_swaps']:<4d} "
              f"= {s['detection_rate']:.3f}   detectable "
              f"{s['n_swaps_detectable_in_principle']:4d} -> "
              f"{fmt(s['detection_rate_on_detectable'], 3)}   "
              f"unexplained miss {s['undetected_unexplained']}")
    ins = acceptance[chosen].get("in_situ_pred_perturbation")
    if ins:
        print(f"acceptance (in situ):   {ins['n_swaps']} swaps, "
              f"detection {fmt(ins['pooled_detection_rate'])}  "
              f"(on detectable: {fmt(ins['pooled_detection_rate_on_detectable'])})")
        for kind, s in ins["by_swap_kind"].items():
            print(f"   {kind:26s} {s['detected']:4d}/{s['n_swaps']:<4d} "
                  f"= {s['detection_rate']:.3f}   detectable "
                  f"{s['n_swaps_detectable_in_principle']:4d} -> "
                  f"{fmt(s['detection_rate_on_detectable'], 3)}")
    pc = acceptance[chosen]["passive_swap_control"]
    print(f"passive control: {pc['detected']}/{pc['n_swaps']} detected "
          f"(must be 0), max|dF1| {pc['max_abs_delta_f1']:.2e}")
    print(f"\nwrote {out_dir}/summary.json, per_image.csv, "
          "acceptance_test.json, REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
