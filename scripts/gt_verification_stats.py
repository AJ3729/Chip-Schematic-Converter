#!/usr/bin/env python3
"""Derive the ground-truth verification statistics from the decision records.

The manuscript needs to say how much judgement went into the annotation —
that is what separates a verified benchmark from a bootstrapped one, and a
reviewer has no reason to take "carefully verified" on faith. Every number
it needs is already implicit in ``<gt>/decisions/<stem>.json``, which stores
the junction/crossing call at every critical site plus each repointed
terminal, so nothing here has to be typed out of the prose report.

What a decision file holds, and what each field means for the table:

    sites        every intersection that was adjudicated. "junction" and
                 "crossing" are the two calls; "none" means the ink meets
                 but nothing joins; a list is an explicit edge grouping for
                 a site too complex for a single call.
    ports        a terminal repointed off the lead the tracer chose — the
                 dominant correction, usually a handwritten value label
                 brushing the component box.
    merge        wire fragments re-joined across a scan gap.
    manual_nets  a net asserted where the component box swallowed the
                 contact, so it could not be traced.
    bridges      a candidate gap bridge rejected as not-touching.
    classes      a published COCO class corrected against the drawn symbol.
    unconnected  a lead the drafter drew going nowhere, marked deliberate.

Second-reader agreement cannot be derived this way — it is a fact about a
process, not about the files — so it is read from a small committed JSON
that the verification report and this table both cite.

Usage:
    python scripts/gt_verification_stats.py --gt-dir data/gt_test_1024
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default="data/gt_test_1024")
    ap.add_argument("--out", default="results/gt_verification/stats.json")
    args = ap.parse_args()

    gt_dir = ROOT / args.gt_dir
    dec_dir = gt_dir / "decisions"
    if not dec_dir.is_dir():
        raise SystemExit(f"no decision records under {dec_dir}")

    sites = Counter()
    counts = Counter()
    n_files = 0
    for p in sorted(dec_dir.glob("*.json")):
        d = json.loads(p.read_text())
        n_files += 1
        for v in d.get("sites", {}).values():
            # a list is an explicit edge grouping, not a single call
            sites["edge_group" if isinstance(v, list) else v] += 1
        # `ports` is {component_id: {terminal_index: port_index}}, so its
        # length is COMPONENTS touched, not terminals moved. The manuscript
        # wants the terminal count — that is the size of the dominant error
        # mode — so descend one level. Reporting 586 here would understate
        # the correction by more than half.
        for cid, moved in (d.get("ports") or {}).items():
            counts["ports_components"] += 1
            counts["ports_terminals"] += (len(moved)
                                          if hasattr(moved, "__len__") else 1)
        for field in ("merge", "manual_nets", "bridges",
                      "classes", "unconnected"):
            v = d.get(field)
            if v:
                counts[field] += len(v) if hasattr(v, "__len__") else 1

    comps = terms = 0
    nets: set[tuple[str, str]] = set()
    unconnected = 0
    verified = Counter()
    for p in sorted(gt_dir.glob("circuit_*.json")):
        g = json.loads(p.read_text())
        verified[bool(g.get("verified"))] += 1
        for c in g["components"]:
            comps += 1
            for t in c["terminals"]:
                terms += 1
                if t.get("net"):
                    nets.add((p.stem, t["net"]))
                else:
                    unconnected += 1

    sr_path = gt_dir / "meta" / "second_reader.json"
    second_reader = json.loads(sr_path.read_text()) if sr_path.exists() else None

    out = {
        "gt_dir": args.gt_dir,
        "images": n_files,
        "components": comps,
        "terminals": terms,
        "nets": len(nets),
        "terminals_with_no_net": unconnected,
        "files_verified": verified[True],
        "files_unverified": verified[False],
        "sites_adjudicated": sum(sites.values()),
        "sites": dict(sites),
        "corrections": dict(counts),
        "second_reader": second_reader,
    }
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")

    print(f"{args.gt_dir}: {n_files} images, {comps} components, "
          f"{terms} terminals, {len(nets)} nets")
    print(f"  sites adjudicated {out['sites_adjudicated']}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(sites.items())))
    print("  corrections: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if second_reader is None:
        print(f"  NOTE: no {sr_path.relative_to(ROOT)}; second-reader "
              "agreement will be absent from the table")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
