#!/usr/bin/env python3
"""Collect every corruption condition into one table, with the caveat attached.

Reports strict success, terminal-pair F1, net F1 and per-component connected
accuracy per condition, each as a delta against the clean control.

GEOMETRIC conditions carry an extra column, frame_drift_px, and it is not
decoration. Ground truth is expressed in the 1024 frame that preprocessing
produced from the UNCORRUPTED scan. Rotate or skew the input and preprocessing
recovers a slightly different frame; components are matched to GT at IoU 0.3 in
that frame, so once the drift approaches a component's own size, a correct
reconstruction is scored wrong for a reason that has nothing to do with the
pipeline. The drift is measured by projecting each GT box centre through the
clean transform and again through the condition's own transform and taking the
median displacement, so a reader can see how much of any geometric drop is
frame misalignment rather than failure.

Usage:
    python scripts/robustness/collect.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import corruptions as C  # noqa: E402
from schematic2netlist.preprocess import project_point, unproject_point  # noqa: E402

RES = ROOT / "results/robustness"
METRICS = ("strict_success", "terminal_pair_f1", "net_f1",
           "per_component_connected_acc", "nged")


def frame_drift(cond: str, gt_dir: Path) -> float | None:
    """Median displacement of GT box centres between clean and this condition."""
    a = ROOT / "data/robustness/transforms/clean.json"
    b = ROOT / "data/robustness/transforms" / f"{cond}.json"
    if not (a.exists() and b.exists()):
        return None
    ta, tb = json.loads(a.read_text()), json.loads(b.read_text())
    d = []
    for gp in sorted(gt_dir.glob("*.json")):
        stem = gp.stem
        if stem not in ta or stem not in tb:
            continue
        try:
            g = json.loads(gp.read_text())
        except Exception:
            continue
        for c in g.get("components", [])[:6]:
            bx, by = c["bbox"][0], c["bbox"][1]
            # clean frame -> original pixels -> this condition's frame
            ox, oy = unproject_point(ta[stem], bx, by)
            nx, ny = project_point(tb[stem], ox, oy)
            d.append(((nx - bx) ** 2 + (ny - by) ** 2) ** 0.5)
    return statistics.median(d) if d else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", default="data/gt_test_1024")
    ap.add_argument("--out", default="results/robustness/SUMMARY.json")
    a = ap.parse_args()

    fam = {c: f for c, f, _ in C.conditions()}
    sev = {c: s for c, _, s in C.conditions()}
    rows = {}
    for cond in fam:
        f = RES / cond / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        t = s.get("topology", {})
        rows[cond] = {
            "family": fam[cond], "severity": sev[cond],
            "scored": s.get("scored"),
            **{m: t.get(m, {}).get("mean") for m in METRICS},
        }
    if "clean" not in rows:
        sys.exit("no clean control -- run it first; deltas are meaningless without it")

    base = rows["clean"]
    for cond, r in rows.items():
        for m in METRICS:
            if r.get(m) is not None and base.get(m) is not None:
                r[f"d_{m}"] = round(r[m] - base[m], 6)
        if r["family"] == "geometric":
            r["frame_drift_px"] = frame_drift(cond, ROOT / a.gt_dir)

    out = {
        "control": {"condition": "clean", "strict_success": base["strict_success"],
                    "published_seed0": 0.53125,
                    "reproduces_published": abs(base["strict_success"] - 0.53125) < 1e-9},
        "n_conditions": len(rows),
        "conditions": rows,
    }
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"control strict={base['strict_success']:.4f} "
          f"(published 0.53125, reproduces={out['control']['reproduces_published']})\n")
    hdr = f"{'condition':18}{'strict':>8}{'Δ':>9}{'pairF1':>8}{'Δ':>9}{'drift px':>10}"
    print(hdr); print("-" * len(hdr))
    for cond in sorted(rows, key=lambda c: (rows[c]["family"], c)):
        r = rows[cond]
        dr = r.get("frame_drift_px")
        print(f"{cond:18}{r['strict_success']:8.4f}{r.get('d_strict_success', 0):+9.4f}"
              f"{r['terminal_pair_f1']:8.4f}{r.get('d_terminal_pair_f1', 0):+9.4f}"
              f"{'' if dr is None else f'{dr:10.1f}'}")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
