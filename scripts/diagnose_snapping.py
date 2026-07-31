#!/usr/bin/env python3
"""WHY does snapping lose 13% of components when connectivity is perfect?

The oracle attributes per-component accuracy at 1024 px as
A 0.5979 -> B 0.6654 -> C 0.8675 -> D 1.0, so the snapping stage costs
**0.1325** on its own (``results/oracle_1024/summary.json``). That number
says how much, never why, and 'improve snapping' is not a fix.

This script runs the oracle's mode C — GT boxes, GT connectivity rendered
as orthogonal conductors with an outward stub at every pin — and attributes
each individual terminal. Mode C is the right condition precisely because
it removes every excuse: the conductor is unambiguously present, straight,
and labelled, so whatever snapping still gets wrong is snapping's own
logic and not a wire-mask artefact.

Attribution needs no Hungarian alignment. ``render_gt_node_map`` returns
``label_of_net``, so the node id a terminal snapped to maps back to a GT
net name directly, and predicted terminal *k* can be compared to GT
terminal *k*.

Per-terminal verdicts:

  ok               snapped to the node carrying this terminal's own net
  missing          None, but GT has a net here — no boundary crossing found
  swapped          this terminal took ANOTHER terminal's net of the SAME
                   component, and that terminal took this one's: a pure
                   identity/permutation error. The right nodes were found
                   and the port template ordered them wrongly.
  wrong_net        snapped to a net that is not on this component at all —
                   a localization error, a different failure entirely
  gt_null          GT itself has no net here; not snapping's fault

**Swaps and wrong-nets are charged to different budgets, and conflating
them is the trap this script exists to avoid.** ``canonicalize_terminals``
reorders terminals by connectivity signature for prediction AND ground
truth before scoring, and ``per_component_connected_accuracy`` judges the
induced grouping — so the benchmark is *largely* blind to terminal order,
and a component holding the right SET of nets usually scores correct
however those nets are arranged.

Only largely, though: canonicalization sorts by ``(signature, original
index)``, so two terminals with the SAME connectivity signature keep their
original relative order and that order leaks into the terminal-pair
indices. Ties are common exactly where they matter — a component whose
pins reach identical partner sets. So a permutation can still move the
metric, and a set-level change certainly does. The two are reported
separately below rather than assumed equivalent.

Permutations also matter for their own sake: they are what C3 (port
templates) claims to fix, and a swapped MOSFET drain/source or diode
anode/cathode emits SPICE with wrong polarity.

**Caveat on the ordered metric.** Digitize-HCD's GT netlists carry only
``{index, net}`` per terminal — no port names — and
``scripts/bootstrap_gt_merged.py`` filled that index order from the
pipeline's own ``node_names`` at bootstrap time, with human verification
covering the NETS rather than which slot is Drain versus Source. The GT
order is not arbitrary (ground lands in the Source slot for 100% of
MOSFET-N and the Emitter slot for 95.9% of BJT-NPN, matching the
templates' naming), but the geometric boundary-walk order the bootstrap
inherited predicts that same pattern, since grounds are drawn at the
bottom and walked last. So ``exact_ordered`` should be read as agreement
with the GT's indexing convention, NOT as pin-identity accuracy.
Validating pin identity needs a small hand-labelled port-identity set,
which this dataset does not provide.

The script therefore reports per-component accuracy BOTH ways:

  exact_ordered     every terminal on its own net, order included
                    -> what a trustworthy netlist needs (C3's metric)
  set_correct       the multiset of nets is right, order ignored
                    -> what the benchmark scores; compare to 0.8675

``exact_ordered - set_correct`` is the permutation-only population:
invisible to the benchmark, fatal to the netlist.

Also reported: how often the boundary walk finds the wrong NUMBER of
distinct nodes (the count is what ``snap_boundary`` truncates or pads),
and the pose-fit statistics of ``match_ports`` — how often the template
was trusted at all, versus falling back to boundary order.

Usage:
    python scripts/diagnose_snapping.py --limit 190
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from schematic2netlist.classes import canonical_class, class_terminals, is_ground
from schematic2netlist.config import load_config
from schematic2netlist.determinism import set_global_seed, write_run_metadata
from schematic2netlist.frames import resolve_and_check
from schematic2netlist.gt import load_gt
from schematic2netlist.oracle_render import render_gt_node_map
from schematic2netlist.snapping import (
    _boundary_run_sites,
    build_component_pin_nets,
)
from schematic2netlist.nodes import bbox_xyxy

from oracle import gt_detections  # noqa: E402  (same-dir import)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=190)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out-dir", default="results/snapping_diag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_global_seed(cfg["seed"])
    gt_dir = args.gt_dir or cfg["benchmark"]["gt_dir"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out, cfg, seed, extra={"gt_dir": gt_dir,
                                              "mode": "oracle C"})

    names = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    names = names[: args.limit]
    images_dir = resolve_and_check(None, names, cfg)

    verdicts = Counter()
    by_class = defaultdict(Counter)
    by_nterm = defaultdict(Counter)
    count_mismatch = Counter()
    pose_used = Counter()
    pose_dist = []
    rows: list[dict] = []
    n_images_scored = 0
    comp_exact = [0, 0]          # [all terminals ok IN ORDER, total]
    comp_setok = 0               # right multiset of nets, order ignored
    comp_kind = Counter()        # exact / permutation_only / set_wrong
    setwrong_by_nterm = defaultdict(lambda: [0, 0])

    for i, nm in enumerate(names, 1):
        stem = nm.rsplit(".", 1)[0]
        gt = load_gt(f"{gt_dir}/{stem}.json")
        img = cv2.imread(f"{images_dir}/{nm}")
        if img is None:
            continue
        node_map, label_of_net, report = render_gt_node_map(gt, img.shape)
        if not report["ok"]:
            continue                     # same subset the oracle scores
        n_images_scored += 1

        gdets = gt_detections(gt)
        comps = build_component_pin_nets(gdets, node_map, cfg)
        # snapping skips non-electrical classes, so index by detection id
        gt_by_id = {c["id"]: c for c in gt["components"]}
        net_of_label = {v: k for k, v in label_of_net.items()}

        for comp in comps:
            det = gdets[comp["id"]]
            gcomp = gt_by_id.get(comp["id"])
            if gcomp is None:
                continue
            cls = canonical_class(det["class"])
            gt_nets = [t["net"] for t in gcomp["terminals"]]
            pred_nets = [None if n is None else net_of_label.get(int(n))
                         for n in comp["nodes"]]
            if is_ground(det["class"]):
                gt_nets, pred_nets = gt_nets[:1], pred_nets[:1]
            n_t = len(gt_nets)

            # how many DISTINCT nodes did the boundary walk actually see?
            x1, y1, x2, y2 = bbox_xyxy(det)
            s = cfg["snapping"]
            seen = []
            for r in range(s["expand_step"], s["max_expand"] + 1,
                           s["expand_step"]):
                found = _boundary_run_sites(node_map, x1 - r, y1 - r,
                                            x2 + r, y2 + r)
                if len(found) > len(seen):
                    seen = found
                if len(found) >= n_t:
                    break
            n_seen = len({nid for nid, _, _ in seen})
            if n_seen != n_t:
                count_mismatch[f"{'under' if n_seen < n_t else 'over'}"
                               f" (saw {n_seen}, want {n_t})"] += 1

            pose_used["template" if comp.get("ports") else "fallback"] += 1
            if comp.get("ports"):
                pose_dist.append(comp["ports"]["mean_dist_frac"])

            own = {n for n in gt_nets if n is not None}
            all_ok = True
            for k in range(n_t):
                g, p = gt_nets[k], pred_nets[k]
                if g is None:
                    v = "gt_null"
                elif p is None:
                    v = "missing"
                elif p == g:
                    v = "ok"
                elif p in own:
                    # took a net belonging to another terminal of THIS
                    # component: the nodes were found, the order was wrong
                    v = "swapped"
                else:
                    v = "wrong_net"
                if v not in ("ok", "gt_null"):
                    all_ok = False
                verdicts[v] += 1
                by_class[cls][v] += 1
                by_nterm[n_t][v] += 1
                rows.append({
                    "image": nm, "comp_id": comp["id"], "class": cls,
                    "terminal": k, "n_terminals": n_t,
                    "gt_net": g, "pred_net": p, "verdict": v,
                    "n_nodes_seen": n_seen,
                    "pose": (comp.get("ports") or {}).get("pose", ""),
                    "pose_dist_frac": (comp.get("ports") or {}).get(
                        "mean_dist_frac", ""),
                })
            comp_exact[1] += 1
            comp_exact[0] += int(all_ok)
            # order-insensitive: does the component hold the right MULTISET
            # of nets? This is what survives canonicalize_terminals and so
            # what the oracle's 0.8675 actually measures.
            g_ms = sorted(n for n in gt_nets if n is not None)
            p_ms = sorted(n for n in pred_nets if n is not None)
            set_ok = g_ms == p_ms
            comp_setok += int(set_ok)
            comp_kind["exact_ordered" if all_ok else
                       ("permutation_only" if set_ok else "set_wrong")] += 1
            setwrong_by_nterm[n_t][1] += 1
            setwrong_by_nterm[n_t][0] += int(not set_ok)

        if i % 20 == 0:
            print(f"[{i}/{len(names)}] images scored={n_images_scored} "
                  f"terminals={sum(verdicts.values())}", flush=True)

    tot = sum(verdicts.values())
    scored = tot - verdicts["gt_null"]

    def pct(n):
        return f"{n/max(scored,1):.1%}"

    print(f"\n=== MODE C: connectivity is PERFECT, so every error below is "
          f"snapping's own ===")
    print(f"images scored {n_images_scored}/{len(names)}   "
          f"components {comp_exact[1]}   terminals {tot} "
          f"({verdicts['gt_null']} with no GT net, excluded)")
    n_c = max(comp_exact[1], 1)
    print(f"\nper-component accuracy, two ways:")
    print(f"  exact_ordered (netlist-correct, C3's metric)  "
          f"{comp_exact[0]}/{comp_exact[1]} = {comp_exact[0]/n_c:.4f}")
    print(f"  set_correct   (what the benchmark scores)     "
          f"{comp_setok}/{comp_exact[1]} = {comp_setok/n_c:.4f}")
    print(f"    -> compare set_correct to oracle mode C's 0.8675")
    print(f"\n  breakdown:")
    for k in ("exact_ordered", "permutation_only", "set_wrong"):
        print(f"    {k:17s} {comp_kind[k]:5d}  {comp_kind[k]/n_c:6.1%}")
    print(f"\n  canonicalize_terminals normalizes MOST of the "
          f"permutation_only population\n  away, so the set_wrong "
          f"{comp_kind['set_wrong']/n_c:.1%} is where per-component headroom "
          f"mainly lives. Permutations\n  survive only where two terminals "
          f"tie on connectivity signature (the sort\n  breaks ties by "
          f"original index). The permutation_only "
          f"{comp_kind['permutation_only']/n_c:.1%} is primarily a\n  "
          f"netlist-polarity defect -- and is measured against a GT whose "
          f"terminal ORDER\n  came from the pipeline itself at bootstrap, so "
          f"read it as convention agreement,\n  not pin-identity accuracy "
          f"(see the module docstring).\n")
    print(f"  set_wrong rate by terminal count:")
    for n_t in sorted(setwrong_by_nterm):
        bad, tot_c = setwrong_by_nterm[n_t]
        print(f"    {n_t}-terminal  {bad:4d}/{tot_c:4d} = {bad/max(tot_c,1):6.1%}")
    print("\nterminal verdicts:")
    for v in ("ok", "swapped", "wrong_net", "missing"):
        print(f"  {v:10s} {verdicts[v]:6d}  {pct(verdicts[v])}")

    err = verdicts["swapped"] + verdicts["wrong_net"] + verdicts["missing"]
    if err:
        print(f"\nof the {err} wrong terminals:")
        for v in ("swapped", "wrong_net", "missing"):
            print(f"  {v:10s} {verdicts[v]/err:.1%}")

    print(f"\nby terminal count (does the 3-pin case fail differently?):")
    print(f"  {'n_term':>6s} {'terminals':>9s} {'ok':>7s} {'swapped':>8s} "
          f"{'wrong':>7s} {'missing':>8s}")
    for n_t in sorted(by_nterm):
        c = by_nterm[n_t]
        s = sum(c.values()) - c["gt_null"]
        print(f"  {n_t:6d} {s:9d} {c['ok']/max(s,1):7.1%} "
              f"{c['swapped']/max(s,1):8.1%} {c['wrong_net']/max(s,1):7.1%} "
              f"{c['missing']/max(s,1):8.1%}")

    print(f"\nboundary walk found the wrong NUMBER of distinct nodes:")
    for k, n in count_mismatch.most_common(8):
        print(f"  {k:28s} {n:5d}")
    if not count_mismatch:
        print("  never — the walk always saw exactly as many nodes as pins")

    print(f"\nport template: {pose_used['template']} trusted, "
          f"{pose_used['fallback']} fell back to boundary order")
    if pose_dist:
        import statistics as st
        print(f"  pose mean_dist_frac: median {st.median(pose_dist):.3f}  "
              f"mean {st.mean(pose_dist):.3f}  "
              f"(rejected above {0.45})")

    print(f"\nworst classes by error rate (>=20 terminals):")
    ranked = []
    for cls, c in by_class.items():
        s = sum(c.values()) - c["gt_null"]
        if s >= 20:
            ranked.append((1 - c["ok"] / s, cls, s, c))
    for rate, cls, s, c in sorted(ranked, reverse=True)[:10]:
        print(f"  {cls:22s} n={s:5d}  err {rate:6.1%}  "
              f"(swap {c['swapped']:4d} wrong {c['wrong_net']:4d} "
              f"missing {c['missing']:4d})")

    with (out / "per_terminal.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "mode": "oracle C (GT boxes + GT-rendered connectivity)",
        "n_images_scored": n_images_scored,
        "n_components": comp_exact[1],
        "per_component_exact_ordered": round(comp_exact[0] / n_c, 4),
        "per_component_set_correct": round(comp_setok / n_c, 4),
        "component_kind": dict(comp_kind),
        "component_kind_rates": {k: round(v / n_c, 4)
                                 for k, v in comp_kind.items()},
        "set_wrong_rate_by_n_terminals": {
            str(k): round(v[0] / max(v[1], 1), 4)
            for k, v in sorted(setwrong_by_nterm.items())},
        "note": ("Only set_wrong is per-component headroom; "
                 "canonicalize_terminals makes permutation_only invisible "
                 "to every benchmark number, though it still corrupts "
                 "SPICE pin order for directional devices."),
        "n_terminals_scored": scored,
        "verdicts": dict(verdicts),
        "verdict_rates": {k: round(v / max(scored, 1), 4)
                          for k, v in verdicts.items() if k != "gt_null"},
        "by_n_terminals": {str(k): dict(v) for k, v in by_nterm.items()},
        "node_count_mismatch": dict(count_mismatch),
        "pose_used": dict(pose_used),
        "by_class": {k: dict(v) for k, v in by_class.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}/per_terminal.csv + summary.json")


if __name__ == "__main__":
    main()
