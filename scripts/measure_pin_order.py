#!/usr/bin/env python3
"""How often does the pipeline emit the RIGHT PIN ORDER on a 3+-terminal part?

No existing metric answers this. `benchmark.canonicalize_terminals` sorts a
component's terminals by a connectivity signature computed identically in
prediction and ground truth, so a swapped collector/emitter cancels and net F1
stays at exactly 1.000. But `netlist.py` writes `Q<c> <b> <e>`,
`M<d> <g> <s> <s>` and `E<out> 0 <in+> <in->` straight off raw terminal index,
so pin order decides whether the emitted SPICE is the circuit that was drawn.
A reversed BJT runs reverse-active; swapped op-amp inputs turn negative
feedback into positive.

METHOD. Align predicted components to GT with the repo's own
`align_components` (IoU, within class). Then build a net correspondence that is
itself PIN-ORDER INVARIANT -- a net is identified by the SET of aligned
component ids touching it, never by terminal index -- because a correspondence
that moved with the swap would let the very error being measured cancel out.
Map predicted nets to GT nets by that signature and ask, for each matched
multi-terminal component, whether predicted terminal k lands on the same GT net
as GT terminal k.

Scored only where the answer is unambiguous: the component's terminals sit on
distinct nets and every one of those nets maps uniquely. Everything else is
reported as undecidable rather than silently counted either way.

Usage:
    python scripts/measure_pin_order.py                      # shipped config
    python scripts/measure_pin_order.py --port-head          # with the learned head
    python scripts/measure_pin_order.py --compare            # both, side by side
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.benchmark import align_components      # noqa: E402
from schematic2netlist.classes import canonical_class, class_terminals  # noqa: E402
from schematic2netlist.config import load_config              # noqa: E402
from schematic2netlist.detect import load_cached_detections   # noqa: E402
from schematic2netlist.gt import load_gt                      # noqa: E402
from schematic2netlist.pipeline import run_pipeline           # noqa: E402


def evaluate(cfg: dict, split: str, gt_dir: Path, limit: int | None = None) -> dict:
    img_dir = Path(cfg["preprocess"]["images_dir"])
    det_dir = Path(cfg["detect"]["cache_dir"])
    stems = [Path(n).stem for n in (ROOT / f"data/splits/{split}.txt").read_text().split()]
    if limit:
        stems = stems[:limit]

    tot = ok = wrong = undec = 0
    by_class: dict = collections.defaultdict(lambda: [0, 0])
    perms: collections.Counter = collections.Counter()
    wrong_examples = []

    for n, s in enumerate(stems, 1):
        gp, dp, ip = gt_dir / f"{s}.json", det_dir / f"{s}.json", img_dir / f"{s}.jpg"
        if not (gp.exists() and dp.exists() and ip.exists()):
            continue
        gt = load_gt(str(gp))
        res = run_pipeline(ip, cfg, detections=load_cached_detections(str(dp)))
        dets = res["detections"]

        pred = [{"id": c["id"], "class": c["class"],
                 "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                          dets[c["id"]]["width"], dets[c["id"]]["height"]],
                 "nets": list(c.get("node_names") or [])}
                for c in res["components"] if 0 <= c["id"] < len(dets)]
        gtc = [{"id": c["id"], "class": c["class"], "bbox": c["bbox"],
                "nets": [t.get("net") for t in c["terminals"]]}
               for c in gt["components"]]

        pa, ga, _ = align_components(pred, gtc, 0.3)
        matched = {c["id"] for c in pa} & {c["id"] for c in gtc}

        def sigs(cs):
            m = collections.defaultdict(set)
            for c in cs:
                for nn in c["nets"]:
                    if nn is not None:
                        m[nn].add(c["id"])
            return m

        ps, gs = sigs(pa), sigs(ga)
        by_sig = collections.defaultdict(list)
        for nm, sg in gs.items():
            by_sig[frozenset(sg)].append(nm)
        p2g = {nm: by_sig[frozenset(sg)][0] for nm, sg in ps.items()
               if len(by_sig.get(frozenset(sg), [])) == 1}

        gbi = {c["id"]: c for c in ga}
        for c in pa:
            if c["id"] not in matched:
                continue
            g = gbi[c["id"]]
            cls = canonical_class(g["class"])
            if class_terminals(cls) < 3:
                continue
            tot += 1
            gn, pn = g["nets"], c["nets"]
            if (len(gn) != len(pn) or len(set(gn)) != len(gn)
                    or any(x is None for x in gn)):
                undec += 1
                continue
            mapped = [p2g.get(x) for x in pn]
            if any(m is None for m in mapped):
                undec += 1
                continue
            if mapped == list(gn):
                ok += 1
                by_class[cls][0] += 1
            else:
                wrong += 1
                by_class[cls][1] += 1
                try:
                    perms[tuple(gn.index(m) for m in mapped)] += 1
                except ValueError:
                    perms[("?",)] += 1
                if len(wrong_examples) < 12:
                    wrong_examples.append(
                        {"image": s, "comp": g["id"], "class": cls,
                         "gt": list(gn), "pred_mapped": mapped})
        if n % 40 == 0:
            print(f"  ...{n}/{len(stems)}", flush=True)

    dec = ok + wrong
    return {
        "split": split, "gt_dir": str(gt_dir),
        "multi_terminal_matched": tot, "decidable": dec, "undecidable": undec,
        "correct": ok, "wrong": wrong,
        "accuracy": round(ok / dec, 4) if dec else None,
        "by_class": {k: {"correct": v[0], "decidable": v[0] + v[1],
                         "accuracy": round(v[0] / (v[0] + v[1]), 4)}
                     for k, v in sorted(by_class.items())},
        "wrong_permutations": {str(k): v for k, v in perms.most_common()},
        "wrong_examples": wrong_examples,
    }


def show(tag: str, r: dict) -> None:
    print(f"\n=== {tag} ===")
    print(f"  matched multi-terminal components : {r['multi_terminal_matched']}")
    print(f"  decidable / undecidable           : {r['decidable']} / {r['undecidable']}")
    print(f"  CORRECT PIN ORDER                 : {r['correct']}/{r['decidable']} "
          f"= {r['accuracy']}")
    for c, v in r["by_class"].items():
        print(f"    {c:12s} {v['correct']:4d}/{v['decidable']:<4d} = {v['accuracy']:.4f}")
    if r["wrong_permutations"]:
        print("  wrong permutations:", dict(list(r["wrong_permutations"].items())[:5]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--port-head", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="run with the head off AND on, and report the delta")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/pin_order")
    args = ap.parse_args()

    cfg = load_config(None)
    gt_dir = Path(args.gt_dir or cfg["benchmark"]["gt_dir"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    if args.compare or not args.port_head:
        c = json.loads(json.dumps(cfg))
        c["snapping"]["port_head"]["enabled"] = False
        print("running WITHOUT the learned port head ...", flush=True)
        runs["templates_only"] = evaluate(c, args.split, gt_dir, args.limit)
    if args.compare or args.port_head:
        c = json.loads(json.dumps(cfg))
        c["snapping"]["port_head"]["enabled"] = True
        print("running WITH the learned port head ...", flush=True)
        runs["port_head"] = evaluate(c, args.split, gt_dir, args.limit)

    for k, v in runs.items():
        show(k, v)
    if len(runs) == 2:
        a, b = runs["templates_only"], runs["port_head"]
        print(f"\n  DELTA: {a['accuracy']} -> {b['accuracy']}  "
              f"({b['correct'] - a['correct']:+d} components)")
    (out / "summary.json").write_text(json.dumps(runs, indent=1) + "\n")
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
