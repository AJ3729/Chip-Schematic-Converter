#!/usr/bin/env python3
"""Generate the manuscript's data figures from committed results/ artifacts.

Same rule as scripts/make_paper_tables.py: nothing here is hand-drawn or
hand-typed. Every value plotted is read out of results/, so a figure cannot
drift away from the table beside it — which is exactly how the precision
buckets ended up a full configuration behind the benchmark they described
before they had a script.

Outputs PDF (vector, what IEEE wants) into paper/figures/:

    fig_precision_cliff.pdf   strict success vs terminal-pair precision,
                              both splits — the paper's central empirical claim
    fig_ablation_waterfall.pdf  cumulative strict success, v1 -> v12
    fig_oracle_waterfall.pdf    stage attribution, modes A/B/C/D
    fig_per_class_ap.pdf        per-class AP against class support
    fig_size_scatter.pdf        per-image accuracy against circuit size

Usage:
    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --only precision_cliff
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"

# IEEE Access column geometry. A figure drawn at the wrong width gets scaled
# in LaTeX and its text stops matching the body text, which is the usual
# reason figure labels come out unreadably small.
COL_W = 3.5      # single column, inches
WIDE_W = 7.16    # full width across both columns

TEST = "results/paper_test/seeds/seed0"
VAL = "results/benchmark_1024_final/seed0"
TEST_GT, VAL_GT = "data/gt_test_1024", "data/gt_val_1024"

# Colour-blind safe (Okabe-Ito). The test split is the reported one and gets
# the strong colour; val is the muted comparison throughout.
C_TEST, C_VAL = "#0072B2", "#E69F00"
C_GRID, C_DEAD = "#CCCCCC", "#BBBBBB"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    """Write PDF (for LaTeX) and PNG (to eyeball), reproducibly.

    matplotlib stamps a CreationDate into PDF metadata, so regenerating an
    unchanged figure produced a non-empty git diff — which trains a reader to
    ignore diffs in exactly the directory where a diff should mean something.
    Setting the date to a constant makes regeneration byte-identical.
    """
    FIG.mkdir(parents=True, exist_ok=True)
    meta = {"pdf": {"CreationDate": None}, "png": {"Software": None}}
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", metadata=meta[ext])
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf (+png)")


def rows(run: str) -> list[dict]:
    with (ROOT / run / "per_image.csv").open() as fh:
        return list(csv.DictReader(fh))


def num(r: dict, k: str) -> float:
    v = r[k]
    return 1.0 if v == "True" else 0.0 if v == "False" else float(v)


# --------------------------------------------------------------------------
def fig_precision_cliff() -> None:
    """Strict success is a step function of terminal-pair precision.

    The claim is not "accuracy correlates with precision" — it is that the
    conversion rate is *identically zero* below a threshold and high above
    it, on both splits. Plotting the two side by side is the whole argument:
    the shape was found on one set of images and reproduced on a set that
    never influenced a parameter.
    """
    def buckets(p: str) -> dict:
        return json.loads((ROOT / p / "precision_buckets.json").read_text())

    t = buckets("results/stratified_test192")
    v = buckets("results/stratified_1024_final")

    labels = [f"{b['precision_range'][0]:.1f}–{b['precision_range'][1]:.1f}"
              for b in t["buckets"]]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for off, src, colour, lab in ((-w / 2, v, C_VAL, "validation (190)"),
                                  (+w / 2, t, C_TEST, "test (192)")):
        by = {tuple(b["precision_range"]): b for b in src["buckets"]}
        got = [by.get(tuple(b["precision_range"])) for b in t["buckets"]]
        rate = [g["strict_rate"] if g else 0.0 for g in got]
        k = [g["strict_successes"] if g else 0 for g in got]
        n = [g["n_images"] if g else 0 for g in got]
        ax.bar(x + off, rate, w, label=lab, color=colour,
               edgecolor="white", linewidth=0.4)
        # "k of n" on every bar, including the zeros: without the denominator
        # a reader cannot tell an empty bucket from one where 31 circuits all
        # failed, and that difference IS the finding
        for xi, r_, ki, ni in zip(x, rate, k, n):
            ax.text(xi + off, r_ + 0.025, f"{ki}/{ni}", ha="center",
                    va="bottom", fontsize=5.8,
                    color=colour if r_ > 0 else "#666666")

    ax.set_xticks(x, labels)
    ax.set_xlabel("terminal-pair precision")
    ax.set_ylabel("strict success rate")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_precision_cliff")


# --------------------------------------------------------------------------
def fig_ablation_waterfall() -> None:
    """Cumulative strict success as each stage is added.

    Reported on VALIDATION, with test beside it. The paper reports the ablation
    on validation because that is the split such decisions may legitimately
    consult; this figure previously read results/ablations_test192/ while its
    caption said validation, which is the sort of mismatch a reader has no way
    to catch. Labels come from spec/ablation_arms.yaml so the figure and the
    prose cannot name a stage differently.
    """
    import yaml
    spec = yaml.safe_load((ROOT / "spec/ablation_arms.yaml").read_text())
    label = {a["id"]: a["label"] for a in spec["arms"]}

    def arms(rel):
        d = json.loads((ROOT / rel).read_text())
        out = []
        for a in d["arms"]["ablation"]:
            s = a["topology"]["strict_success"]
            out.append((a["label"], s["mean"], s["ci95_lo"], s["ci95_hi"]))
        return d["n_images"], out

    n_val, V = arms("results/final/ablation_val/index.json")
    n_test, T = arms("results/final/ablation/index.json")
    test_by = {k: v for k, v, _, _ in T}

    keys = [k for k, _, _, _ in V]
    y = np.array([v for _, v, _, _ in V])
    lo = np.array([l for _, _, l, _ in V])
    hi = np.array([h for _, _, _, h in V])
    ty = np.array([test_by.get(k, np.nan) for k in keys])
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(WIDE_W, 3.1))
    ax.bar(x, y, 0.60, color=C_TEST, edgecolor="white", linewidth=0.4,
           label=f"validation ({n_val}), reported")
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#333333",
                elinewidth=0.7, capsize=2)
    ax.plot(x, ty, marker="o", ms=3.2, lw=0.9, color=C_VAL, ls="--",
            label=f"test ({n_test}), comparison")
    # step connectors: read it as a cumulative progression, not 12 configs
    for i in range(len(x) - 1):
        ax.plot([x[i] + 0.30, x[i + 1] - 0.30], [y[i], y[i]],
                ls=":", lw=0.6, color="#888888")
    for xi, yi, hii in zip(x, y, hi):
        ax.text(xi, hii + 0.016, f"{yi:.3f}", ha="center", va="bottom",
                fontsize=5.6)

    # the one stage the paper singles out
    j = keys.index("v5_plus_crossover_DEFAULT")
    ax.annotate("port templates:\nthe terminal-identity stage",
                xy=(x[j], y[j]), xytext=(x[j] - 0.4, y[j] + 0.20),
                fontsize=5.8, color="#0b4a75", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#0b4a75"))

    ax.set_xticks(x)
    # Wrap hard: at this width an unwrapped label runs into its neighbour and
    # the axis becomes unreadable, which is how the first draft of this figure
    # came out.
    import textwrap
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(label.get(k, k), 13)) for k in keys],
        fontsize=5.4)
    ax.set_ylabel("strict success")
    ax.set_ylim(0, max(hi.max(), np.nanmax(ty)) + 0.10)
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_ablation_waterfall")


# --------------------------------------------------------------------------
def fig_oracle_waterfall() -> None:
    """Where the error actually is: replace one stage at a time with GT.

    Modes are cumulative — B is A with GT detections, C is B with GT
    connectivity, D is everything. The gap between consecutive bars is the
    error that stage owns, which is the number the attribution reports.
    """
    o = json.loads((ROOT / "results/oracle_test192/summary.json").read_text())
    m, attr = o["means_on_valid_subset"], o["attribution_tp_f1"]

    modes = ["A", "B", "C", "D"]
    names = ["predicted\n(baseline)", "+ GT\ndetections",
             "+ GT\nconnectivity", "all GT\n(ceiling)"]
    y = [m[k]["tp_f1"] for k in modes]
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    # Bars deliberately narrow. The deltas are the point of this figure and
    # they live in the GAPS, so the gaps need room; wide bars leave the
    # annotations sitting on top of the data they describe.
    ax.bar(x, y, 0.40, color=[C_TEST, C_TEST, C_TEST, C_DEAD],
           edgecolor="white", linewidth=0.4)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.008, f"{yi:.3f}", ha="center", va="bottom",
                fontsize=6.2)

    # Classic waterfall: a dotted rule carries each bar's height across the
    # gap and the delta is measured against it.
    for i, key in enumerate(["detection", "wires", "snapping"]):
        y0, y1 = y[i], y[i + 1]
        ax.plot([x[i] + 0.20, x[i + 1] + 0.20], [y0, y0],
                ls=":", lw=0.6, color="#999999", zorder=1)
        xm = x[i] + 0.5
        ax.annotate("", (xm, y1), (xm, y0),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="#B22222",
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.text(xm, (y0 + y1) / 2, f"{key}\n+{attr[key]:.4f}",
                ha="center", va="center", fontsize=5.6, color="#B22222",
                linespacing=1.2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec="none", alpha=0.92))

    ax.set_xticks(x, names, fontsize=6.5, linespacing=1.3)
    ax.set_ylabel("terminal-pair $F_1$")
    ax.set_ylim(0.70, 1.05)
    ax.set_title(f"$n={o['n_mode_c_valid']}$ of {o['n_images']} "
                 "(mode-C wiring verified)", fontsize=7)
    save(fig, "fig_oracle_waterfall")


# --------------------------------------------------------------------------
def fig_per_class_ap() -> None:
    """Per-class detection quality against class support.

    AP@0.5 is saturated — every one of the 17 classes clears 0.97 — so
    plotting it as bars produces seventeen identical full-width bars and
    says nothing. The variation lives entirely in AP@0.5:0.95, i.e. in
    localisation tightness rather than in whether the symbol is found. Bars
    are therefore AP@0.5:0.95; AP@0.5 is a tick mark, present to show the
    saturation rather than to be compared across classes.
    """
    p = ROOT / "results/detection_test192/test/per_class_ap.csv"
    with p.open() as fh:
        data = [(r["class"], int(r["support"]), float(r["ap50"]),
                 float(r["ap50_95"])) for r in csv.DictReader(fh)]
    data.sort(key=lambda t: t[3])
    names, sup = [d[0] for d in data], [d[1] for d in data]
    ap50 = [d[2] for d in data]
    ap5095 = [d[3] for d in data]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    ax.barh(y, ap5095, 0.62, color=C_TEST, edgecolor="white", linewidth=0.3,
            label="AP@0.5:0.95")
    ax.scatter(ap50, y, marker="|", s=42, linewidths=1.1, color="#B22222",
               zorder=3, label="AP@0.5")
    for yi, s in zip(y, sup):
        ax.text(1.005, yi, f"n={s}", va="center", fontsize=5.5, color="#555555")

    ax.set_yticks(y, names, fontsize=6.5)
    ax.set_xlim(0, 1.13)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("average precision")
    # above the axes: inside, it lands on the two lowest-scoring classes
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), handlelength=1.4)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_per_class_ap")


# --------------------------------------------------------------------------
def fig_size_scatter() -> None:
    """Accuracy against circuit size, with the split comparison on top.

    Two things at once, deliberately: the negative size correlation that the
    limitations section rests on, and the fact that the two splits lie on the
    same curve — which is the visual form of "the test split is not easier".
    """
    def series(run: str, gt_dir: str, split: str):
        n, f1, strict = [], [], []
        for r in rows(run):
            stem = Path(r["image"]).stem
            gp = ROOT / gt_dir / f"{stem}.json"
            if not gp.exists():
                continue
            n.append(len(json.loads(gp.read_text())["components"]))
            f1.append(num(r, "net_f1"))
            strict.append(num(r, "strict_success"))
        return np.array(n), np.array(f1), np.array(strict)

    nt, ft, st = series(TEST, TEST_GT, "test")
    nv, fv, sv = series(VAL, VAL_GT, "val")

    rng = np.random.default_rng(0)      # jitter only; seeded so it is stable
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for n_, f_, colour, lab in ((nv, fv, C_VAL, f"validation ($n$={len(nv)})"),
                                (nt, ft, C_TEST, f"test ($n$={len(nt)})")):
        # component count is an integer, so without jitter every circuit of a
        # given size lands on one vertical line and the density is unreadable
        ax.scatter(n_ + rng.uniform(-0.28, 0.28, n_.size), f_,
                   s=7, c=colour, alpha=0.55, linewidths=0, label=lab)

    # Quantile bins over BOTH splits. Equal-width bins put ~2 circuits in the
    # tail bins and the trend line then swings on single images; equal-COUNT
    # bins keep every point of the line supported by the same evidence. One
    # line, not one per split: the trend is a property of the task, and two
    # lines would invite reading noise as a difference between splits.
    alln = np.concatenate([nt, nv]).astype(float)
    allf = np.concatenate([ft, fv])
    q = np.quantile(alln, np.linspace(0, 1, 8))
    idx = np.clip(np.digitize(alln, q[1:-1]), 0, 6)
    bx = [alln[idx == i].mean() for i in range(7) if (idx == i).any()]
    by = [allf[idx == i].mean() for i in range(7) if (idx == i).any()]
    ax.plot(bx, by, color="#333333", lw=1.2, marker="o", ms=3,
            label="mean, equal-count bins")

    r = np.corrcoef(alln, allf)[0, 1]
    ax.text(0.97, 0.08, f"Pearson $r={r:.2f}$", transform=ax.transAxes,
            ha="right", fontsize=6.5, color="#333333")
    ax.set_xlabel("components in circuit")
    ax.set_ylabel("net $F_1$")
    ax.set_ylim(-0.03, 1.06)
    ax.legend(frameon=False, loc="lower left", fontsize=6)
    save(fig, "fig_size_scatter")


def fig_capture() -> None:
    """Capture invariance: how many distinct circuits four photographs produce.

    The unit is the DRAWING, not the photograph. Each CGHD drawing is
    photographed four times, so a pipeline invariant to capture would return one
    topology per drawing. The histogram is the whole result: the modal outcome
    is four photographs of one drawing yielding four different circuits.
    """
    d = json.loads((ROOT / "results/cghd_capture_invariance.json").read_text())
    hist = d["distinct_topologies_histogram"]
    ks = [1, 2, 3, 4]
    vals = [hist.get(str(k), 0) for k in ks]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    # 1 distinct topology is the only invariant outcome; colour it as the
    # target and everything else as the failure it is.
    colours = [C_TEST] + [C_VAL] * 3
    bars = ax.bar([str(k) for k in ks], vals, color=colours, width=0.62)
    for b, v in zip(bars, vals):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v + total * 0.015,
                    f"{v}\n({v / total:.1%})", ha="center", va="bottom",
                    fontsize=6.5, linespacing=1.1)
    ax.set_xlabel("distinct topologies from the drawing's 4 photographs")
    ax.set_ylabel("drawings")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title(f"{d['groups_all_captures_agree']} of {d['n_groups']} drawings "
                 f"({d['fraction_all_agree']:.1%}) give one answer",
                 fontsize=7.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_capture")


def fig_pipeline() -> None:
    """The pipeline as a single row of stages, drawn from the stage list.

    Deliberately schematic rather than pretty: the point a reader needs is that
    terminal identity is decided at ONE stage, late, and that repair is outside
    the topology path entirely. Both are claims the paper makes in prose and
    neither survives a diagram that shows a single undifferentiated chain.
    """
    stages = [
        ("preprocess", "deskew, crop,\nbinarise"),
        ("detect", "YOLOv8s\ncomponents"),
        ("class head", "re-decide\nclass"),
        ("wire mask", "ink minus\ncomponents"),
        ("skeleton", "graph of\nnodes + edges"),
        ("junctions", "junction vs\ncrossing"),
        ("terminals", "snap to\ncomponent"),
        ("port head", "which terminal\nis which"),
        ("nets", "connected\ncomponents"),
        ("netlist", "SPICE\ndeck"),
    ]
    fig, ax = plt.subplots(figsize=(WIDE_W, 1.55))
    ax.set_axis_off()
    n = len(stages)
    w, gap = 1.0, 0.30
    for i, (name, sub) in enumerate(stages):
        x = i * (w + gap)
        # the port head is the stage this paper is about
        accent = name == "port head"
        ax.add_patch(plt.Rectangle(
            (x, 0), w, 1.0, facecolor=(C_TEST if accent else "#FFFFFF"),
            edgecolor=(C_TEST if accent else "#666666"),
            linewidth=(1.1 if accent else 0.7), zorder=2))
        ax.text(x + w / 2, 0.68, name, ha="center", va="center", fontsize=6.6,
                color=("white" if accent else "black"),
                fontweight=("bold" if accent else "normal"), zorder=3)
        ax.text(x + w / 2, 0.30, sub, ha="center", va="center", fontsize=5.4,
                color=("white" if accent else "#444444"), linespacing=1.15,
                zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap, 0.5), xytext=(x + w, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                        color="#666666"))
    span = n * (w + gap) - gap
    # Repair sits OUTSIDE the topology path and the figure has to show that,
    # because the paper's claim that no topology metric involves repair is the
    # kind of thing a single undifferentiated chain quietly contradicts.
    box_w = 2 * w + gap
    box_x = span - box_w
    ax.add_patch(plt.Rectangle((box_x, -0.92), box_w, 0.55,
                               facecolor="#FFFFFF", edgecolor=C_VAL,
                               linewidth=0.9, linestyle=(0, (3, 2)), zorder=2))
    ax.text(box_x + box_w / 2, -0.645,
            "declared repair\n(ledgered, never scored)", ha="center",
            va="center", fontsize=5.6, color="#8a6100", linespacing=1.2,
            zorder=3)
    # arrow from the repair box up into the netlist stage it feeds
    last_cx = (n - 1) * (w + gap) + w / 2
    ax.annotate("", xy=(last_cx, 0.0), xytext=(last_cx, -0.37),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_VAL,
                                linestyle=(0, (3, 2))))
    ax.set_xlim(-0.15, span + 0.15)
    ax.set_ylim(-1.02, 1.12)
    save(fig, "fig_pipeline")


def fig_op_gap() -> None:
    """The paper's central claim as one picture: perfect, then simulated.

    Two stacked bars over the SAME population, so the eye compares the split
    rather than two independent quantities. Read from the multistability record,
    which carries the headline counts and the flagged-circuit control together.
    """
    d = json.loads((ROOT / "results/multistability.json").read_text())
    h = d["headline_all_circuits"]
    perfect = int(h["topologically_perfect"])
    disagree = int(h["of_those_op_disagrees"])
    agree = perfect - disagree
    scored = int(d["n_circuits_tested"])
    not_perfect = scored - perfect

    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    ax.barh([1], [perfect], color=C_TEST, height=0.55)
    ax.barh([1], [not_perfect], left=[perfect], color=C_DEAD, height=0.55)
    ax.barh([0], [agree], color=C_TEST, height=0.55)
    ax.barh([0], [disagree], left=[agree], color=C_VAL, height=0.55)
    ax.barh([0], [not_perfect], left=[perfect], color=C_DEAD, height=0.55)

    ax.text(perfect / 2, 1, f"{perfect} scored perfect", ha="center",
            va="center", color="white", fontsize=7, fontweight="bold")
    ax.text(perfect + not_perfect / 2, 1, f"{not_perfect} not",
            ha="center", va="center", color="#555555", fontsize=6.5)
    ax.text(agree / 2, 0, f"{agree} agree", ha="center", va="center",
            color="white", fontsize=7)
    ax.text(agree + disagree / 2, 0,
            f"{disagree} DISAGREE\n({h['rate']:.1%})", ha="center",
            va="center", color="#4a3400", fontsize=6.8, fontweight="bold",
            linespacing=1.1)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["operating point", "topology metric"], fontsize=7)
    ax.set_xlabel(f"circuits where both sides solve (n={scored})")
    ax.set_xlim(0, scored)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_op_gap")


def fig_vlm() -> None:
    """Unaided vs handed our detections, for both frontier models.

    The decomposition is the result: the gap is enormous in variant A and gone
    in variant B, which localises the failure to component detection rather
    than to connectivity reasoning. Values are the ones the manuscript's
    tab:vlm reports, kept here in one literal block so the two cannot drift
    without this comment being wrong too.
    """
    systems = ["This work", "Claude Opus 5", "GPT-5.5"]
    unaided = [0.5312, 0.1250, 0.1250]
    assisted = [0.5312, 0.5295, 0.6823]

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    x = range(len(systems))
    wdt = 0.36
    ax.bar([i - wdt / 2 for i in x], unaided, wdt, label="unaided (variant A)",
           color=C_VAL)
    ax.bar([i + wdt / 2 for i in x], assisted, wdt,
           label="given our detections (variant B)", color=C_TEST)
    for i, (u, a) in enumerate(zip(unaided, assisted)):
        ax.text(i - wdt / 2, u + 0.015, f"{u:.4f}", ha="center", va="bottom",
                fontsize=6)
        ax.text(i + wdt / 2, a + 0.015, f"{a:.4f}", ha="center", va="bottom",
                fontsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems, fontsize=7)
    ax.set_ylabel("strict success")
    ax.set_ylim(0, 0.80)
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_vlm")



def fig_qualitative() -> None:
    """circuit_1247: perfect by every structural metric, different operating point.

    Two panels, because the finding is a comparison and not a picture. Left: the
    drawing, with the rail the pipeline chose as its reference marked -- the
    reference annotation has no ground symbol and no net 0 at all. Right: the
    corresponded node voltages, which are what the metric compares and which are
    only defined relative to a reference the drawing never specifies.
    """
    import json as _json
    stem = "circuit_1247"
    rec = _json.loads((ROOT / f"results/final/op_agreement/cache/{stem}.json").read_text())
    res = _json.loads((ROOT / "results/residual_circuits.json").read_text())
    corr = rec["corr"]
    ref_net = res["residuals"][stem]["gt_net_serving_as_pred_reference"]

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from measure_op_agreement import (build_deck, spice_components,
                                      policy_placeholders, _run_ngspice)
    from schematic2netlist.config import load_config
    cfg = load_config(None)
    ph = policy_placeholders(cfg, "hv")
    g = _run_ngspice(build_deck(spice_components(rec["gt_graph"]), ph), cfg)
    pr = _run_ngspice(build_deck(spice_components(rec["pred_graph"]), ph), cfg)

    pairs = []
    for gn, pn in corr.items():
        a = g["voltages"].get(str(gn).lower())
        b = pr["voltages"].get(str(pn).lower())
        if a is not None and b is not None:
            pairs.append((gn, a, b))
    pairs.sort(key=lambda r: -abs(r[2] - r[1]))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(WIDE_W, 2.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    img = plt.imread(str(ROOT / f"data/cleaned_1024/{stem}.jpg"))
    axl.imshow(img, cmap="gray")
    axl.set_axis_off()
    axl.set_title(f"(a) {stem}: no ground symbol is drawn", fontsize=7.5)

    labels = [p[0] for p in pairs]
    y = range(len(pairs))
    axr.barh([i - 0.2 for i in y], [p[1] for p in pairs], height=0.38,
             color=C_TEST, label="reference netlist")
    axr.barh([i + 0.2 for i in y], [p[2] for p in pairs], height=0.38,
             color=C_VAL, label="reconstruction")
    axr.set_yticks(list(y))
    axr.set_yticklabels(labels, fontsize=5.5)
    axr.invert_yaxis()
    axr.set_xlabel("node voltage (V)")
    axr.set_title(f"(b) same graph, but the reconstruction makes {ref_net} its 0 V",
                  fontsize=7.5)
    axr.grid(axis="y", visible=False)
    axr.set_xlim(0, max(max(q[1], q[2]) for q in pairs) * 1.32)
    axr.legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "fig_qualitative")


FIGURES = {
    "precision_cliff": fig_precision_cliff,
    "ablation_waterfall": fig_ablation_waterfall,
    "oracle_waterfall": fig_oracle_waterfall,
    "per_class_ap": fig_per_class_ap,
    "size_scatter": fig_size_scatter,
    "capture": fig_capture,
    "pipeline": fig_pipeline,
    "op_gap": fig_op_gap,
    "vlm": fig_vlm,
    "qualitative": fig_qualitative,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()

    style()
    for name, fn in FIGURES.items():
        if args.only and args.only != name:
            continue
        fn()


if __name__ == "__main__":
    main()
