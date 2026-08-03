#!/usr/bin/env python3
"""Generate every number and table the manuscript uses from committed
results/ artifacts (project rule: NO hand-typed numbers in the paper).

Outputs:
    paper/generated/numbers.tex   — \\newcommand macros used in prose
    paper/tables/*.tex            — booktabs table bodies \\input by main

Sources:
    results/detection/{summary.json,per_class_ap.csv,seed_stats.json}
    results/ablations/wire_method.csv
    results/benchmark/seed{0,1,2}/summary.json      (3-seed, if present)
    results/v5_stitch_crossover/summary.json        (current default run)

Rerun after any results change:
    python scripts/make_paper_tables.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "generated"
TAB = ROOT / "paper" / "tables"

# Which set of result directories to read. Every committed 512-px run was
# superseded when preprocess.target_size became 1024, and the two must
# never be mixed in one table — so the choice is made once, here, rather
# than path by path.
VARIANTS = {
    # THE REPORTED SET. Every artifact here is on the 192-image test split,
    # which no parameter was ever selected on. The "1024" set below reads the
    # same pipeline on the 190 images that WERE swept, and is a validation
    # number whatever its run_meta.json calls it (data/README.md ->
    # "the 2026-08-03 role swap"). Never mix the two in one table.
    # Detection carries a caveat the others do not: the YOLO weights were
    # early-stopped on these 192 images, so detection here is optimistic by a
    # measured +0.017 mAP@0.5 against detection_test192/val. See
    # src/schematic2netlist/splits.py.
    "test": {
        "detection": "results/detection_test192/test",
        "ablation": "results/ablations_test192/wire_method.csv",
        "default_run": "results/paper_test/seeds/seed0",
        "seeds": "results/paper_test/seeds",
        "oracle": "results/oracle_test192",
        "repair": "results/repair_test192",
        "stratified": "results/stratified_test192",
        "ports": "results/ports",          # templates are scale-invariant
    },
    # benchmark_1024_final, not benchmark_1024: the latter is the working area
    # and still holds the pre-2026-07-30 runs, whose nGED was produced by the
    # timeout-truncated GED search and is not reproducible (see metrics.py).
    # The _final runs are on the deterministic bound AND on the current default
    # (bridge_span 7 + connectivity repair), and must not be mixed with the old
    # ones in a single table.
    "1024": {
        "detection": "results/detection_1024",
        "ablation": "results/ablations_1024/wire_method.csv",
        "default_run": "results/benchmark_1024_final/seed0",
        "seeds": "results/benchmark_1024_final",
        "oracle": "results/oracle_1024",
        "repair": "results/repair_1024",
        "stratified": "results/stratified_1024",
        "ports": "results/ports",          # templates are scale-invariant
    },
    "512": {
        "detection": "results/detection",   # numbers as cited; detection_512fixed reproduces them
        "ablation": "results/ablations/wire_method.csv",
        "default_run": "results/v5_stitch_crossover",
        "seeds": "results/benchmark",
        "oracle": "results/oracle",
        "repair": "results/repair",
        "stratified": "results/stratified",
        "ports": "results/ports",
    },
}
SRC = dict(VARIANTS["test"])   # the reported, held-out split

ABL_LABELS = {
    # csv label -> table row label (paper-facing). EVERY arm present in the
    # csv must appear here: gen_ablation_table skips rows it cannot name, so
    # a missing key silently drops a row and the ablation quietly shortens.
    # The 2026-08-03 replay renamed the arms and this table went from twelve
    # rows to two before anyone looked at it.
    "v1_classical_directional": "classical (canny + directional snap)",
    "v2_ink_boundary_snap": "+ ink wires + boundary snap",
    "v3_plus_stitching": "+ mask-hole stitching",
    "v4_plus_crossover": "+ crossover-aware nets",
    "v5_plus_crossover_DEFAULT": "+ port templates",
    "v6_plus_bridge_span7": "+ bridge span 7",
    "v7_plus_connectivity_repair": "+ connectivity repair",
    "v8_plus_snap_expand80": "+ snap expand 80",
    "v9_plus_blob_thresholds": "+ blob thresholds",
    "v10_plus_sauvola": "+ Sauvola binarization",
    "v11_plus_class_head": "+ class head",
    "v12_plus_head_ensemble": "+ class-head ensemble (shipped)",
    # superseded 512-era labels, kept so --variant 512 still tables
    "v2_newpreproc": "+ fixed preprocessing",
    "v3_ink_boundary_snap": "+ ink wires + boundary snap",
    "v4_plus_stitching": "+ mask-hole stitching",
}

TOPO_COLS = [
    ("terminal_pair_f1", "term-pair $F_1$"),
    ("net_f1", "net $F_1$"),
    ("per_component_connected_acc", "per-comp.\\ acc."),
    ("nged", "nGED $\\downarrow$"),
    ("strict_success", "strict"),
]


def f3(x) -> str:
    return f"{float(x):.3f}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def load_ablation() -> list[dict]:
    with (ROOT / SRC["ablation"]).open() as fh:
        return list(csv.DictReader(fh))


def gen_numbers(abl: list[dict]) -> None:
    by_label = {r["label"]: r for r in abl}
    classical = by_label["v1_classical_directional"]
    v5 = by_label["v5_plus_crossover_DEFAULT"]
    # The ablation grew past v5. AblTpFOneVFive now means the v5 STAGE, which is
    # no longer the shipped configuration, so the prose must not use it as the
    # headline -- *Final* is the last row and follows the table automatically.
    final = abl[-1]

    det = json.loads((ROOT / SRC["detection"] / "summary.json").read_text())
    seed_stats_path = ROOT / SRC["detection"] / "seed_stats.json"
    if not seed_stats_path.exists():
        # a single-seed number must not be silently substituted for the
        # 3-seed mean+-std the manuscript reports
        raise SystemExit(
            f"missing {seed_stats_path}. Generate it with:\n"
            f"  ./venv/bin/python scripts/detector_comparison.py "
            f"--data data/yolo_1024/dataset.yaml --out-dir {SRC['detection']}")
    det_seeds = json.loads(seed_stats_path.read_text())
    s8 = det_seeds["yolov8s"]

    v5sum = json.loads(
        (ROOT / SRC["default_run"] / "summary.json").read_text()
    )
    rep = v5sum["repair"]

    lines = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "% Regenerate after any results change.",
        macro("NumTestImages", str(v5sum["scored"])),
        *_support_macros(),
        *_oracle_macros(),
        *_repair_verify_macros(),
        *_bucket_macros(),
        *_detection_floor_macros(),
        macro("DetMapFifty", f3(s8["map50"]["mean"])),
        macro("DetMapFiftyStd", f"{s8['map50']['std']:.4f}"),
        macro("DetMapFiftyNinetyFive", f3(s8["map50_95"]["mean"])),
        macro("AblNetFOneClassical", f3(classical["net_f1"])),
        macro("AblNetFOneVFive", f3(v5["net_f1"])),
        macro("AblTpFOneClassical", f3(classical["terminal_pair_f1"])),
        macro("AblTpFOneVFive", f3(v5["terminal_pair_f1"])),
        macro("AblNetFOneFinal", f3(final["net_f1"])),
        macro("AblTpFOneFinal", f3(final["terminal_pair_f1"])),
        macro("AblStrictFinal", f3(final["strict_success"])),
        macro("AblStrictClassical", f3(classical["strict_success"])),
        macro("AblFinalLabel", final["label"].replace("_", " ")),
        macro("RepSolvBefore", f3(rep["solvable_before_rate"])),
        macro("RepSolvAfter", f3(rep["solvable_after_rate"])),
        macro("RepMeanAssumptions", f"{rep['mean_assumptions']:.1f}"),
        macro("RepMeanGauge", f"{rep['mean_gauge']:.1f}"),
        macro("RepSpiceValid", f3(rep["spice_valid_rate"])),
    ]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {GEN/'numbers.tex'} ({len(lines)-2} macros)")


def _signed(x: float) -> str:
    return f"{x:+.4f}"


def _support_macros() -> list[str]:
    """Test-split class supports, so the Dataset section can state its
    range without anyone typing a count."""
    p = ROOT / SRC["detection"] / "per_class_ap.csv"
    if not p.exists():
        return ["% per-class supports absent — run scripts/eval_detector.py"]
    with p.open() as fh:
        rows = [(r["class"], int(r["support"])) for r in csv.DictReader(fh)]
    lo = min(rows, key=lambda r: r[1])
    hi = max(rows, key=lambda r: r[1])
    return [
        macro("SupportMin", str(lo[1])),
        macro("SupportMinClass", lo[0]),
        macro("SupportMax", str(hi[1])),
        macro("SupportMaxClass", hi[0]),
        macro("NumClasses", str(len(rows))),
    ]


def _oracle_macros() -> list[str]:
    p = ROOT / SRC["oracle"] / "summary.json"
    if not p.exists():
        return ["% oracle results absent — run scripts/oracle.py"]
    s = json.loads(p.read_text())
    a = s["attribution_tp_f1"]
    return [
        macro("OracleDetection", _signed(a["detection"])),
        macro("OracleWires", _signed(a["wires"])),
        macro("OracleSnapping", _signed(a["snapping"])),
        macro("OracleNValid", str(s["n_mode_c_valid"])),
        # The oracle runs on a prefix of the split, not all of it — mode C
        # synthesises wiring per image and is expensive. Without this macro
        # the prose said "N of \NumTestImages", i.e. 59 of 192, when the
        # denominator is the 60 images actually attempted.
        macro("OracleNAttempted", str(s["n_images"])),
    ]


def _detection_floor_macros() -> list[str]:
    """Worst per-class AP@0.5 — the claim "saturated for every class"."""
    p = ROOT / SRC["detection"] / "per_class_ap.csv"
    if not p.exists():
        return []
    with p.open() as fh:
        aps = [(float(r["ap50"]), r["class"]) for r in csv.DictReader(fh)]
    lo = min(aps)
    return [macro("DetMinClassApFifty", f3(lo[0])),
            macro("DetMinClassApFiftyClass", lo[1])]


def _bucket_macros() -> list[str]:
    """The precision-bucket edges, so prose about the cliff cannot drift."""
    p = ROOT / SRC["stratified"] / "precision_buckets.json"
    if not p.exists():
        return ["% precision buckets absent — run scripts/precision_buckets.py"]
    d = json.loads(p.read_text())
    top = d["buckets"][-1]
    return [
        macro("BucketTopEdge", f"{top['precision_range'][0]:.1f}"),
        macro("BucketTopRate", f"{top['strict_rate']:.3f}"),
        macro("BucketTopN", str(top["n_images"])),
        macro("BucketTopShare", f"{d['share_of_strict_in_top_bucket']:.3f}"),
    ]


def _repair_verify_macros() -> list[str]:
    p = ROOT / SRC["repair"] / "summary.json"
    if not p.exists():
        return ["% repair results absent — run scripts/benchmark_repair.py"]
    s = json.loads(p.read_text())
    out = [
        macro("RepLiftCiLo", f3(s["lift_ci95_lo"])),
        macro("RepLiftCiHi", f3(s["lift_ci95_hi"])),
        macro("RepRegressed", str(s["regressed_images"])),
    ]
    if "topology_violations" in s:
        out += [
            macro("RepTopoViolations", str(s["topology_violations"])),
            macro("RepVerifiedImages", str(s["verified_images"])),
        ]
    if s.get("ground_accuracy_gauge_gnd_symbol_resolved") is not None:
        out += [
            macro("GroundAccResolved",
                  f3(s["ground_accuracy_gauge_gnd_symbol_resolved"])),
            macro("GroundAccStrict",
                  f3(s["ground_accuracy_gauge_gnd_symbol_strict"])),
            macro("GroundNGnd", str(s["ground_n_gauge_gnd_symbol"])),
        ]
    return out


def gen_port_table() -> None:
    p = ROOT / SRC["ports"] / "template_accuracy.json"
    if not p.exists():
        print(f"skip port table: {SRC['ports']}/template_accuracy.json absent")
        return
    acc = json.loads(p.read_text())
    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{lrcccc}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{oracle pose} & "
        "\\multicolumn{2}{c}{axis only} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "class & crops & median & $\\le$0.10 & median & $\\le$0.10 \\\\",
        "\\midrule",
    ]
    for cls, a in sorted(acc.items()):
        o, x = a["oracle_pose"], a["axis_only"]
        out.append(
            f"{cls} & {a['n_crops']} & {o['median_norm_err']:.3f} & "
            f"{o['frac_within_0.10'] * 100:.0f}\\% & "
            f"{x['median_norm_err']:.3f} & "
            f"{x['frac_within_0.10'] * 100:.0f}\\% \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "port_templates.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'port_templates.tex'} ({len(acc)} classes)")


def gen_stratified_table() -> None:
    p = ROOT / SRC["stratified"] / "stratified.csv"
    if not p.exists():
        print("skip stratified table: run scripts/analyze_failures.py")
        return
    with p.open() as fh:
        rows = list(csv.DictReader(fh))
    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{lrcccc}",
        "\\toprule",
        "stratum & $n$ & term-pair $F_1$ & net $F_1$ & per-comp. & strict \\\\",
        "\\midrule",
    ]
    for r in rows:
        label = r["stratum"].replace("<=", "$\\le$").replace(">", "$>$")
        out.append(
            f"{label} & {r['n_images']} & {float(r['terminal_pair_f1']):.3f} & "
            f"{float(r['net_f1']):.3f} & "
            f"{float(r['per_component_connected_acc']):.3f} & "
            f"{float(r['strict_success']):.3f} \\\\"
        )
        if r["stratum"] == "all":
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "stratified.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'stratified.tex'} ({len(rows)} strata)")


def gen_detection_table() -> None:
    """Per-class detection, with the val column that carries the caveat.

    The detector was early-stopped on the 192 test images, so its numbers
    there are optimistic. The size of that optimism belongs in the table
    rather than in a footnote a reader can skip: when the sibling run on the
    190 images the detector never saw is available, both columns are shown
    and the difference is visible per class.
    """
    with (ROOT / SRC["detection"] / "per_class_ap.csv").open() as fh:
        rows = list(csv.DictReader(fh))

    sib = ROOT / SRC["detection"] / ".." / "val" / "per_class_ap.csv"
    held_out = {}
    if sib.exists():
        with sib.open() as fh:
            held_out = {r["class"]: r for r in csv.DictReader(fh)}

    if held_out:
        head = ("class & support & AP@0.5 & AP@0.5:0.95 & "
                "AP@0.5 & AP@0.5:0.95 \\\\")
        spec, pre = "lrrrrr", (
            "& & \\multicolumn{2}{c}{test (early-stopped on)} & "
            "\\multicolumn{2}{c}{val (unseen by detector)} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}")
    else:
        head, spec, pre = "class & support & AP@0.5 & AP@0.5:0.95 \\\\", "lrrr", None

    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        *([pre] if pre else []),
        head,
        "\\midrule",
    ]
    for r in rows:
        cells = f"{r['class']} & {r['support']} & {f3(r['ap50'])} & {f3(r['ap50_95'])}"
        if held_out:
            h = held_out.get(r["class"])
            cells += (f" & {f3(h['ap50'])} & {f3(h['ap50_95'])}" if h
                      else " & -- & --")
        out.append(cells + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "detection_per_class.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'detection_per_class.tex'} ({len(rows)} classes"
          + (", both splits)" if held_out else ")"))


def gen_ablation_table(abl: list[dict]) -> None:
    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{l" + "c" * len(TOPO_COLS) + "}",
        "\\toprule",
        "configuration & "
        + " & ".join(h for _, h in TOPO_COLS)
        + " \\\\",
        "\\midrule",
    ]
    for r in abl:
        if r["label"] not in ABL_LABELS:
            continue  # crossover-null side branches: narrated, not tabled
        cells = []
        for key, _ in TOPO_COLS:
            if r.get(key):
                cells.append(
                    f"{f3(r[key])} [{f3(r[key + '_ci95_lo'])}, "
                    f"{f3(r[key + '_ci95_hi'])}]"
                )
            else:
                cells.append("--")
        out.append(f"{ABL_LABELS[r['label']]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "wire_method_ablation.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'wire_method_ablation.tex'}")


def gen_3seed_table() -> None:
    seed_dirs = sorted((ROOT / SRC["seeds"]).glob("seed*"))
    summaries = []
    for d in seed_dirs:
        p = d / "summary.json"
        if p.exists():
            summaries.append(json.loads(p.read_text()))
    out = ["% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit."]
    if len(summaries) < 2:
        out.append(
            "% 3-seed results not present yet — run scripts/benchmark.py "
            f"per seed into {SRC['seeds']}/seed<N>/."
        )
        out.append("\\multicolumn{2}{c}{\\emph{3-seed run pending}}")
    else:
        out += [
            "\\begin{tabular}{lc}",
            "\\toprule",
            f"metric & mean $\\pm$ std ({len(summaries)} seeds) \\\\",
            "\\midrule",
        ]
        for key, header in TOPO_COLS:
            vals = [s["topology"][key]["mean"] for s in summaries]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out.append(f"{header} & {mean:.3f} $\\pm$ {std:.3f} \\\\")
        for key, header in [
            ("solvable_before_rate", "DC-solvable (before repair)"),
            ("solvable_after_rate", "DC-solvable (after repair)"),
        ]:
            vals = [s["repair"][key] for s in summaries]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out.append(f"{header} & {mean:.3f} $\\pm$ {std:.3f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "benchmark_3seed.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'benchmark_3seed.tex'} ({len(summaries)} seeds found)")


def gen_repair_table() -> None:
    v5 = json.loads((ROOT / SRC["default_run"] / "summary.json").read_text())
    rep = v5["repair"]
    rows = [
        ("SPICE syntactic validity", f3(rep["spice_valid_rate"])),
        ("DC-solvable before repair", f3(rep["solvable_before_rate"])),
        ("DC-solvable after repair", f3(rep["solvable_after_rate"])),
        ("mean ledger entries / circuit (assumption)", f"{rep['mean_assumptions']:.1f}"),
        ("mean ledger entries / circuit (gauge)", f"{rep['mean_gauge']:.1f}"),
    ]

    # topology preservation and gauge accuracy come from the dedicated
    # repair evaluation, which recomputes them per image rather than
    # asserting the invariant
    rp = ROOT / SRC["repair"] / "summary.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        rows.append((
            "solvability lift (paired bootstrap 95\\% CI)",
            f"{r['solvability_lift']:.3f} "
            f"[{r['lift_ci95_lo']:.3f}, {r['lift_ci95_hi']:.3f}]",
        ))
        rows.append(("circuits made worse by repair", str(r["regressed_images"])))
        if "topology_violations" in r:
            rows.append((
                "topology changed by repair",
                f"{r['topology_violations']} / {r['verified_images']}",
            ))
        if r.get("ground_accuracy_gauge_gnd_symbol_resolved") is not None:
            rows.append((
                "ground-choice accuracy, GND symbol present",
                f"{r['ground_accuracy_gauge_gnd_symbol_resolved']:.3f} "
                f"($n$={r['ground_n_gauge_gnd_symbol']}, decidable cases)",
            ))
        if r.get("ground_n_assumed"):
            rows.append((
                "ground-choice accuracy, no GND symbol",
                f"{r['ground_accuracy_assumed_strict']:.3f} "
                f"($n$={r['ground_n_assumed']})",
            ))

    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{lc}",
        "\\toprule",
    ]
    out += [f"{name} & {val} \\\\" for name, val in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "repair_summary.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'repair_summary.tex'}")


def gen_splits_table() -> None:
    """Split composition — the C1 scale claim, and the swap made explicit."""
    meta = json.loads(
        (ROOT / "data/splits/splits_meta.json").read_text())
    gt = {"test": "data/gt_test_1024", "val": "data/gt_val_1024"}

    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "split & images & components & terminals & nets & role \\\\",
        "\\midrule",
    ]
    roles = {
        "train": "detector training",
        "val": "parameter selection",
        "test": "reported, never selected on",
    }
    for split in ("train", "val", "test"):
        n_img = meta["counts"][split]
        if split in gt:
            comps = terms = 0
            nets = set()
            for p in sorted((ROOT / gt[split]).glob("circuit_*.json")):
                g = json.loads(p.read_text())
                for c in g["components"]:
                    comps += 1
                    for t in c["terminals"]:
                        terms += 1
                        if t.get("net"):
                            nets.add((p.stem, t["net"]))
            cells = f"{comps} & {terms} & {len(nets)}"
        else:
            # no connectivity GT on train; the box annotations are published
            # and the count would invite a false comparison
            cells = "-- & -- & --"
        out.append(f"{split} & {n_img} & {cells} & {roles[split]} \\\\")

    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "splits.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'splits.tex'}")


def gen_gt_verification_table() -> None:
    """How much judgement the ground truth actually took (C1)."""
    p = ROOT / "results/gt_verification/stats.json"
    if not p.exists():
        print("  skip gt_verification: run scripts/gt_verification_stats.py")
        return
    s = json.loads(p.read_text())
    sites, corr = s["sites"], s["corrections"]

    rowsets = [
        ("intersection sites adjudicated", s["sites_adjudicated"]),
        ("\\quad read as junction", sites.get("junction", 0)),
        ("\\quad read as crossing", sites.get("crossing", 0)),
        ("\\quad explicit edge grouping", sites.get("edge_group", 0)),
        ("\\quad ink meets, nothing joins", sites.get("none", 0)),
        ("terminals repointed to a different lead",
         corr.get("ports_terminals", 0)),
        ("nets asserted where the box swallowed the contact",
         corr.get("manual_nets", 0)),
        ("wire fragments re-joined across a scan gap", corr.get("merge", 0)),
        ("gap bridges rejected as not-touching", corr.get("bridges", 0)),
        ("published component classes corrected", corr.get("classes", 0)),
        ("components marked deliberately unconnected",
         corr.get("unconnected", 0)),
    ]
    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "recorded decision & count \\\\",
        "\\midrule",
        *[f"{k} & {v} \\\\" for k, v in rowsets],
    ]

    sr = s.get("second_reader")
    if sr:
        out += ["\\midrule",
                "\\multicolumn{2}{l}{\\emph{second reader, re-derived from the "
                "drawing first}} \\\\"]
        for smp in sr["samples"]:
            three = ("" if smp["three_terminal_parts"] in (None, 0)
                     else f" ({smp['three_terminal_parts']} 3-terminal parts)")
            out.append(
                f"\\quad {smp['sample']}{three} & "
                f"{smp['disagreements']}/{smp['files']} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "gt_verification.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'gt_verification.tex'} ({len(rowsets)} decision types)")


def gen_generalization_table() -> None:
    """Validation vs test: the answer to 'you tuned on your test set'."""
    p = ROOT / "results/split_swap/val_vs_test.json"
    if not p.exists():
        print("  skip generalization: run scripts/compare_splits.py")
        return
    d = json.loads(p.read_text())
    names = {
        "strict_success": "strict end-to-end success",
        "terminal_pair_f1": "terminal-pair $F_1$",
        "net_f1": "net $F_1$",
        "per_component_connected_acc": "per-component accuracy",
        "per_component_recall_acc": "per-component recall accuracy",
        "nged": "nGED $\\downarrow$",
    }
    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "metric & val (190, tuned) & test (192, held out) & $\\Delta$ & $p$ \\\\",
        "\\midrule",
    ]
    for r in d["metrics"]:
        out.append(
            f"{names.get(r['metric'], r['metric'])} & {r['val_mean']:.4f} & "
            f"{r['test_mean']:.4f} & {r['delta']:+.4f} & {r['p_value']:.3f} \\\\")

    prof = d["difficulty_profile"]
    keys = [("mean_components", "mean components"),
            ("mean_gt_nets", "mean nets"),
            ("pct_with_crossover", "\\% with a wire crossover"),
            ("pct_with_3plus_terminal", "\\% with a 3+-terminal device")]
    out += ["\\midrule",
            "\\multicolumn{5}{l}{\\emph{difficulty profile — the splits are "
            "matched, so the gap is not an easier test set}} \\\\"]
    for k, label in keys:
        out.append(f"{label} & {prof['val'][k]} & {prof['test'][k]} & & \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "generalization.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'generalization.tex'}")


def gen_oracle_table() -> None:
    """Stage attribution: replace one stage at a time with ground truth."""
    o = json.loads((ROOT / SRC["oracle"] / "summary.json").read_text())
    m, attr = o["means_on_valid_subset"], o["attribution_tp_f1"]
    desc = {
        "A": "predicted (baseline)",
        "B": "+ GT detections",
        "C": "+ GT connectivity",
        "D": "all GT (ceiling)",
    }
    out = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py — do not edit.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "mode & substituted & term-pair $F_1$ & net $F_1$ & per-comp.\\ acc. \\\\",
        "\\midrule",
    ]
    for k in ("A", "B", "C", "D"):
        out.append(f"{k} & {desc[k]} & {f3(m[k]['tp_f1'])} & "
                   f"{f3(m[k]['net_f1'])} & {f3(m[k]['percomp'])} \\\\")
    out += ["\\midrule",
            "\\multicolumn{5}{l}{\\emph{terminal-pair $F_1$ attributable to "
            "each stage}} \\\\"]
    for k in ("detection", "wires", "snapping"):
        out.append(f"\\multicolumn{{2}}{{l}}{{\\quad {k}}} & "
                   f"{_signed(attr[k])} & & \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "oracle_attribution.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {TAB/'oracle_attribution.tex'} "
          f"(n={o['n_mode_c_valid']}/{o['n_images']})")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="test",
                    help="which result set to read. 'test' (default) is the "
                         "192-image held-out split and is what the manuscript "
                         "reports; '1024' is the same pipeline on the 190 "
                         "images every parameter was tuned on, i.e. a "
                         "validation number; '512' is superseded. Never mix "
                         "two variants in one table.")
    args = ap.parse_args()
    SRC.clear(); SRC.update(VARIANTS[args.variant])
    print(f"variant={args.variant}: reading {SRC['default_run']}")

    GEN.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    abl = load_ablation()
    gen_numbers(abl)
    gen_detection_table()
    gen_ablation_table(abl)
    gen_3seed_table()
    gen_repair_table()
    gen_port_table()
    gen_stratified_table()
    gen_splits_table()
    gen_gt_verification_table()
    gen_generalization_table()
    gen_oracle_table()


if __name__ == "__main__":
    main()
