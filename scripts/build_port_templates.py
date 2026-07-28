#!/usr/bin/env python3
"""Build per-class port templates from Digitize-HCD port-location data
(contribution C3, MSP path — deterministic, no training required).

Digitize-HCD ships, per class, thousands of 320x320 component crops with
pixel port coordinates and (for directional parts) **port names**:

    Resistor/XY Coordinates/resistor_0.txt   ->  "171 26 182 293"
    Diode/...                                ->  "Anode 43 156\\nCathode 272 185"
    MOSFET-N/...                             ->  "Drain .. \\nGate .. \\nSource .."

No published work uses this modality. It gives pin identity and polarity
directly, which the pipeline previously could not recover at all
(terminal order was arbitrary — a documented correctness bug).

Hand-drawn symbols appear in any orientation, so a single mean position
per port is meaningless. Splitting merely into horizontal/vertical is
not enough either: a diode pointing left and one pointing right are
both "horizontal" but are mirror images, so averaging them puts every
port in the middle of the crop (measured: median error 0.22 of the crop
width for diodes, 0.43 for MOSFET-P). This script therefore bins each
crop by **pose** — the angle of the vector between the first and last
port in canonical order, quantized to 8 sectors — derived from the port
constellation itself, never from image content.

Port ORDER within a template is fixed by identity (Anode before
Cathode, Drain/Gate/Source), so downstream code can emit correct pin
order and polarity.

Two accuracies are reported, because they answer different questions:

- **oracle pose** — given the true pose, how well does the template
  place ports? This measures template quality, and is the ceiling for
  any pose-selection scheme.
- **axis-only pose** — using only horizontal-vs-vertical, which is all
  a bounding box reveals without reading the symbol. This is the
  honest inference-time baseline, and the gap between the two is
  exactly what a learned port model (the [IDEAL] path) would buy.

Output: configs/port_templates.json (committed — it is derived data,
small, and part of the released artifact).

Usage:
    python scripts/build_port_templates.py
    python scripts/build_port_templates.py --eval-holdout 0.2
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from schematic2netlist.classes import canonical_class, canonical_classes

DEFAULT_ROOT = Path(
    "data/digitize_hcd/extracted/Digitize-HCD Dataset/Component Port Location Data"
)
CROP = 320.0

# Canonical port order per class, matching the names Digitize-HCD
# actually uses AND the SPICE argument order the netlist writer emits.
# Classes absent here are unnamed in the port data (their terminals are
# electrically interchangeable, or the annotation carries no identity),
# and are ordered geometrically instead — I-AC, V-AC and the one-port
# DC source are unnamed upstream, so no polarity can be claimed for them.
PORT_ORDER = {
    "Diode": ["Anode", "Cathode"],
    "Zener Diode": ["Anode", "Cathode"],
    "V-DC": ["Positive", "Negative"],
    # SPICE current sources take (from, to): positive current flows from
    # the first node through the source to the second.
    "I-DC": ["Flowing From", "Flowing To"],
    "MOSFET-N": ["Drain", "Gate", "Source"],
    "MOSFET-P": ["Drain", "Gate", "Source"],
    "BJT-NPN": ["Collector", "Base", "Emitter"],
    "BJT-PNP": ["Collector", "Base", "Emitter"],
    "Op-Amp": ["In+", "In-", "Out"],
}


def parse_xy(text: str) -> list[tuple[str | None, float, float]]:
    """Parse a port file into [(name|None, x, y), ...]."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    ports: list[tuple[str | None, float, float]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 3 and not parts[0].lstrip("-").isdigit():
            name = " ".join(parts[:-2])
            ports.append((name, float(parts[-2]), float(parts[-1])))
        else:
            nums = [float(p) for p in parts]
            for i in range(0, len(nums) - 1, 2):
                ports.append((None, nums[i], nums[i + 1]))
    return ports


N_SECTORS = 8


def axis_of(ports: list[tuple[str | None, float, float]]) -> str:
    """Dominant spread axis — all a bounding box alone can tell you."""
    if len(ports) < 2:
        return "any"
    xs = [p[1] for p in ports]
    ys = [p[2] for p in ports]
    return "horizontal" if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else "vertical"


def pose_of(ports: list[tuple[str | None, float, float]]) -> str:
    """Pose bin: the first->last port direction, quantized to 8 sectors.

    Distinguishes mirror images that ``axis_of`` merges (a diode with
    its anode left of the cathode from one with it on the right), which
    is what makes a directional template meaningful at all.
    """
    if len(ports) < 2:
        return "any"
    import math

    dx = ports[-1][1] - ports[0][1]
    dy = ports[-1][2] - ports[0][2]
    if dx == 0 and dy == 0:
        return "any"
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    sector = int((ang + 180.0 / N_SECTORS) % 360.0 // (360.0 / N_SECTORS))
    return f"pose{sector}"


def order_ports(cls: str, ports: list) -> list | None:
    """Put ports into canonical order; None if the file is unusable."""
    order = PORT_ORDER.get(cls)
    if order and all(p[0] for p in ports):
        by_name = {p[0]: p for p in ports}
        if not all(n in by_name for n in order):
            return None
        return [by_name[n] for n in order]
    if order:
        return None            # named class but the file has no names
    # unnamed (symmetric) parts: order geometrically for a stable
    # template. Their terminals are electrically interchangeable, so
    # this ordering is a convention, not a claim about identity.
    if axis_of(ports) == "horizontal":
        return sorted(ports, key=lambda p: p[1])
    return sorted(ports, key=lambda p: p[2])


def build(root: Path, holdout: float, seed: int) -> tuple[dict, dict]:
    dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    templates: dict = {}
    eval_rows: dict = {}

    for dir_name in dirs:
        # Template keys MUST be canonical class names — the port data
        # directory for the one-port source is spelled "V-DC (one-port)"
        # while the published class is "V-DC (one port)", and a raw key
        # would silently never match at inference.
        cls = canonical_class(dir_name)
        if cls not in set(canonical_classes()):
            print(f"skip {dir_name!r}: not a canonical class ({cls!r})")
            continue
        xy_dir = root / dir_name / "XY Coordinates"
        if not xy_dir.is_dir():
            continue
        files = sorted(xy_dir.glob("*.txt"))
        if not files:
            continue

        # deterministic holdout by index (no RNG: reproducible everywhere)
        stride = int(1 / holdout) if holdout > 0 else 0
        fit, held = [], []
        for i, p in enumerate(files):
            (held if (stride and i % stride == 0) else fit).append(p)

        by_pose: dict[str, list] = defaultdict(list)
        by_axis: dict[str, list] = defaultdict(list)
        n_bad = 0
        for p in fit:
            ports = parse_xy(p.read_text())
            ordered = order_ports(cls, ports)
            if ordered is None:
                n_bad += 1
                continue
            by_pose[pose_of(ordered)].append(ordered)
            by_axis[axis_of(ordered)].append(ordered)

        cls_tpl: dict = {
            "n_ports": None, "port_names": PORT_ORDER.get(cls),
            "poses": _fit_group(by_pose), "axes": _fit_group(by_axis),
            "n_fit": len(fit), "n_unusable": n_bad,
        }
        for group in ("poses", "axes"):
            for entry in cls_tpl[group].values():
                cls_tpl["n_ports"] = len(entry["ports"])
        if cls_tpl["poses"]:
            templates[cls] = cls_tpl

        eval_rows[cls] = held

    return templates, eval_rows


def _fit_group(grouped: dict[str, list], min_samples: int = 20) -> dict:
    """Median port positions per bin, dropping bins too sparse to trust."""
    out: dict = {}
    for key, samples in grouped.items():
        n_ports = statistics.mode(len(s) for s in samples)
        samples = [s for s in samples if len(s) == n_ports]
        if len(samples) < min_samples:
            continue
        pts = []
        for k in range(n_ports):
            xs = [s[k][1] / CROP for s in samples]
            ys = [s[k][2] / CROP for s in samples]
            pts.append({
                "x": round(statistics.median(xs), 4),
                "y": round(statistics.median(ys), 4),
                "x_iqr": round(_iqr(xs), 4),
                "y_iqr": round(_iqr(ys), 4),
            })
        out[key] = {"n": len(samples), "ports": pts}
    return out


def _iqr(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[int(0.75 * (n - 1))] - s[int(0.25 * (n - 1))]


def _err_stats(errs: list[float]) -> dict:
    errs = sorted(errs)
    return {
        "n_ports_scored": len(errs),
        "median_norm_err": round(errs[len(errs) // 2], 4),
        "mean_norm_err": round(sum(errs) / len(errs), 4),
        "p90_norm_err": round(errs[int(0.9 * (len(errs) - 1))], 4),
        "frac_within_0.10": round(sum(1 for e in errs if e <= 0.10) / len(errs), 4),
        "frac_within_0.15": round(sum(1 for e in errs if e <= 0.15) / len(errs), 4),
    }


def evaluate(templates: dict, eval_rows: dict) -> dict:
    """Port-localization error on held-out crops, under both pose
    regimes (see module docstring).

    Error is the normalized distance between a template's predicted port
    position and the annotated one; 1.0 is the crop's full width.
    """
    out = {}
    for cls, files in eval_rows.items():
        tpl = templates.get(cls)
        if not tpl or not files:
            continue
        errs = {"oracle_pose": [], "axis_only": []}
        matched, skipped = 0, 0
        for p in files:
            ports = parse_xy(p.read_text())
            ordered = order_ports(cls, ports)
            if ordered is None:
                skipped += 1
                continue
            slots = {
                "oracle_pose": tpl["poses"].get(pose_of(ordered)),
                "axis_only": tpl["axes"].get(axis_of(ordered)),
            }
            if any(s is None or len(s["ports"]) != len(ordered)
                   for s in slots.values()):
                skipped += 1
                continue
            matched += 1
            for regime, slot in slots.items():
                for k, port in enumerate(ordered):
                    dx = port[1] / CROP - slot["ports"][k]["x"]
                    dy = port[2] / CROP - slot["ports"][k]["y"]
                    errs[regime].append((dx * dx + dy * dy) ** 0.5)
        if errs["oracle_pose"]:
            out[cls] = {
                "n_crops": matched,
                "n_skipped": skipped,
                "directional": cls in PORT_ORDER,
                "oracle_pose": _err_stats(errs["oracle_pose"]),
                "axis_only": _err_stats(errs["axis_only"]),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default="configs/port_templates.json")
    ap.add_argument("--eval-out", default="results/ports/template_accuracy.json")
    ap.add_argument("--eval-holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"port data not found: {root}")

    templates, eval_rows = build(root, args.eval_holdout, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(templates, indent=2) + "\n")
    print(f"wrote {args.out}: {len(templates)} classes")
    for cls, t in sorted(templates.items()):
        print(f"  {cls:18s} ports={t['n_ports']} "
              f"pose bins={len(t['poses'])} axis bins={len(t['axes'])}")

    if args.eval_holdout > 0:
        acc = evaluate(templates, eval_rows)
        Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.eval_out).write_text(json.dumps(acc, indent=2) + "\n")
        print(f"\nheld-out port localization ({args.eval_out}):")
        print(f"  {'class':18s} {'n':>6s} | {'oracle-pose':>19s} | "
              f"{'axis-only':>19s}")
        print(f"  {'':18s} {'':>6s} | {'median':>9s} {'<=0.10':>9s} | "
              f"{'median':>9s} {'<=0.10':>9s}")
        for cls, a in sorted(acc.items()):
            o, x = a["oracle_pose"], a["axis_only"]
            print(f"  {cls:18s} {a['n_crops']:6d} | {o['median_norm_err']:9.4f} "
                  f"{o['frac_within_0.10']:9.2%} | {x['median_norm_err']:9.4f} "
                  f"{x['frac_within_0.10']:9.2%}")


if __name__ == "__main__":
    main()
