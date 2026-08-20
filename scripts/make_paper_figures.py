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

# Validated with the dataviz skill's palette checker (light surface):
# lightness band, chroma floor, CVD separation, normal-vision floor and
# contrast-vs-surface all PASS. Vermillion rather than amber because amber
# scored 2.19:1 against the surface and would have needed a relief clause.
C_MAIN = "#0072B2"    # the series the figure is about
C_ALT  = "#D55E00"    # the comparison series
C_MUTE = "#B9BEC7"    # context bars: present, not competing
C_INK  = "#1a1d21"    # text
C_SUB  = "#5b616b"    # secondary text
C_GRID = "#E3E5E8"    # hairline grid, one step off surface


def style() -> None:
    """Print-figure chrome: the data is the only thing allowed to be loud."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "text.color": C_INK,
        "axes.labelcolor": C_INK,
        "axes.edgecolor": C_SUB,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "axes.titlepad": 7,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "xtick.color": C_SUB,
        "ytick.color": C_SUB,
        "xtick.major.size": 0,
        "ytick.major.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


C_TEST, C_VAL, C_DEAD = C_MAIN, C_ALT, C_MUTE


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
    """Cumulative strict success as each stage is added, reported on VALIDATION.

    Emphasis rather than twelve saturated blocks: the story is one stage, so
    eleven bars are muted context and the port-template stage carries the only
    colour. Three direct labels, not twelve -- a number on every bar is the
    anti-pattern that makes a chart go unread.
    """
    import textwrap
    import yaml
    spec = yaml.safe_load((ROOT / "spec/ablation_arms.yaml").read_text())
    label = {a["id"]: a["label"] for a in spec["arms"]}

    def arms(rel):
        d = json.loads((ROOT / rel).read_text())
        return d["n_images"], [(a["label"], a["topology"]["strict_success"])
                               for a in d["arms"]["ablation"]]

    n_val, V = arms("results/final/ablation_val/index.json")
    n_test, T = arms("results/final/ablation/index.json")
    test_by = {k: s["mean"] for k, s in T}

    keys = [k for k, _ in V]
    y = np.array([s["mean"] for _, s in V])
    lo = np.array([s["ci95_lo"] for _, s in V])
    hi = np.array([s["ci95_hi"] for _, s in V])
    ty = np.array([test_by.get(k, np.nan) for k in keys])
    x = np.arange(len(keys))
    star = keys.index("v5_plus_crossover_DEFAULT")

    fig, ax = plt.subplots(figsize=(WIDE_W, 2.7))
    colours = [C_MAIN if i == star else C_MUTE for i in range(len(keys))]
    ax.bar(x, y, 0.58, color=colours, linewidth=0)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none",
                ecolor=C_SUB, elinewidth=0.55, capsize=1.6, alpha=0.85)
    # The comparison must not out-shout the series being reported: thin line,
    # small open markers, drawn UNDER the emphasised bar.
    ax.plot(x, ty, lw=0.9, color=C_ALT, marker="o", ms=2.6,
            markerfacecolor="white", markeredgewidth=0.8,
            markeredgecolor=C_ALT, zorder=2, alpha=0.95)

    # Two labels, not twelve. The emphasised bar is labelled inside it, where
    # nothing else competes; the endpoint sits above its error bar.
    ax.text(x[star], y[star] - 0.035, f"{y[star]:.3f}", ha="center", va="top",
            fontsize=7.2, color="white", fontweight="bold")
    ax.text(x[-1], hi[-1] + 0.016, f"{y[-1]:.3f}", ha="center", va="bottom",
            fontsize=6.6, color=C_SUB)
    # Annotation in the empty lower-right, clear of every mark.
    ax.annotate("the terminal-identity stage:\nthe largest single gain",
                xy=(x[star] + 0.32, 0.20), xytext=(x[star] + 1.05, 0.115),
                fontsize=6.6, color=C_MAIN, va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_MAIN,
                                shrinkA=1, shrinkB=1))

    ax.set_xticks(x)
    # break_long_words=False: at width 11 the default splits "connectivity"
    # into "connectiv / ity", which looks like a typo rather than a wrap.
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(label.get(k, k), 12, break_long_words=False))
         for k in keys], fontsize=6.0, color=C_SUB)
    for i, lab in enumerate(ax.get_xticklabels()):
        if i == star:
            lab.set_color(C_MAIN); lab.set_fontweight("bold")
    ax.set_ylabel("strict success")
    ax.set_ylim(0, 0.62)
    ax.set_xlim(-0.7, len(keys) - 0.3)
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_MUTE, label=f"validation ({n_val}) — reported"),
                       Line2D([], [], color=C_ALT, lw=1.4, marker="o", ms=3.4,
                              markerfacecolor="white",
                              label=f"test ({n_test}) — comparison")],
              frameon=False, loc="upper left", fontsize=6.6,
              handlelength=1.6, borderpad=0.1, labelspacing=0.35,
              bbox_to_anchor=(0.005, 1.02))
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
    """The pipeline as five phases, not ten boxes.

    Ten equal boxes across a two-column width gave every stage the same weight
    and 0.7in of room, which is neither readable nor true: the paper's claim is
    that ONE phase decides terminal identity. Grouping into five phases with
    their stages listed beneath buys the space to say so, and lets the
    identity phase carry the only colour.
    """
    from matplotlib.patches import FancyBboxPatch
    phases = [
        ("Preprocess", "deskew, crop,\nshadow-normalise"),
        ("Detect", "YOLOv8s +\nclass head"),
        ("Trace", "wire mask, skeleton,\njunction vs crossing"),
        ("Identify", "terminal snapping,\nport head"),
        ("Export", "nets,\nSPICE deck"),
    ]
    star = 3

    fig, ax = plt.subplots(figsize=(WIDE_W, 1.62))
    ax.set_axis_off()
    w, gap, h = 1.0, 0.34, 0.72
    for i, (name, sub) in enumerate(phases):
        x = i * (w + gap)
        on = i == star
        ax.add_patch(FancyBboxPatch(
            (x, 0), w, h, boxstyle="round,pad=0,rounding_size=0.045",
            facecolor=(C_MAIN if on else "#F4F5F7"),
            edgecolor=(C_MAIN if on else "#D5D8DD"),
            linewidth=(0 if on else 0.7), zorder=2))
        ax.text(x + w / 2, h * 0.63, name, ha="center", va="center",
                fontsize=8.2, color=("white" if on else C_INK),
                fontweight="bold", zorder=3)
        ax.text(x + w / 2, h * 0.27, sub, ha="center", va="center",
                fontsize=5.9, color=("#D8E7F2" if on else C_SUB),
                linespacing=1.3, zorder=3)
        if i < len(phases) - 1:
            ax.annotate("", xy=(x + w + gap - 0.055, h / 2),
                        xytext=(x + w + 0.055, h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                        color="#9AA0A8",
                                        mutation_scale=7))
    span = len(phases) * (w + gap) - gap

    ax.annotate("decides which terminal is which —\n"
                "the quantity every graph metric discards",
                xy=(star * (w + gap) + w / 2, -0.045),
                xytext=(star * (w + gap) + w / 2, -0.40),
                ha="center", va="center", fontsize=6.4, color=C_MAIN,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_MAIN,
                                shrinkA=1, shrinkB=1))

    # repair: outside the topology path, and drawn that way
    rx, rw = span - w, w
    ax.add_patch(FancyBboxPatch(
        (rx, h + 0.30), rw, 0.30,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor="white", edgecolor=C_ALT, linewidth=0.8,
        linestyle=(0, (2.6, 1.8)), zorder=2))
    ax.text(rx + rw / 2, h + 0.45, "declared repair", ha="center",
            va="center", fontsize=6.4, color=C_ALT, zorder=3)
    ax.annotate("", xy=(rx + rw / 2, h + 0.04),
                xytext=(rx + rw / 2, h + 0.28),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C_ALT,
                                linestyle=(0, (2.2, 1.6)), mutation_scale=6))
    ax.text(rx + rw / 2, h + 0.70, "outside the topology path",
            ha="center", va="center", fontsize=6.0, color=C_SUB)

    ax.set_xlim(-0.10, span + 0.10)
    ax.set_ylim(-0.62, h + 0.84)
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


def fig_qualitative() -> None:
    """circuit_1247: perfect by every structural metric, different operating point.

    A dumbbell rather than paired bars. The quantity that matters is the GAP
    between the two netlists at each node, and a dumbbell encodes a difference
    as a length you read directly; forty paired bars made the reader compute it.
    Nodes are sorted by that gap, so the shape of the failure is the shape of
    the chart.
    """
    import json as _json
    import sys as _sys
    stem = "circuit_1247"
    rec = _json.loads((ROOT / f"results/final/op_agreement/cache/{stem}.json").read_text())
    res = _json.loads((ROOT / "results/residual_circuits.json").read_text())
    ref_net = res["residuals"][stem]["gt_net_serving_as_pred_reference"]

    _sys.path.insert(0, str(ROOT / "scripts"))
    from measure_op_agreement import (build_deck, spice_components,
                                      policy_placeholders, _run_ngspice)
    from schematic2netlist.config import load_config
    cfg = load_config(None)
    ph = policy_placeholders(cfg, "hv")
    g = _run_ngspice(build_deck(spice_components(rec["gt_graph"]), ph), cfg)
    pr = _run_ngspice(build_deck(spice_components(rec["pred_graph"]), ph), cfg)

    pairs = []
    for gn, pn in rec["corr"].items():
        a_ = g["voltages"].get(str(gn).lower())
        b_ = pr["voltages"].get(str(pn).lower())
        if a_ is not None and b_ is not None:
            pairs.append((gn, a_, b_))
    pairs.sort(key=lambda r: abs(r[2] - r[1]))          # gap ascending

    fig = plt.figure(figsize=(WIDE_W, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.13)
    axl, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    img = plt.imread(str(ROOT / f"data/cleaned_1024/{stem}.jpg"))
    axl.imshow(img, cmap="gray", interpolation="bilinear")
    axl.set_axis_off()
    axl.set_title("(a)  no ground symbol is drawn", fontsize=7.4,
                  color=C_INK, loc="left", pad=5)

    y = np.arange(len(pairs))
    gt = np.array([r[1] for r in pairs])
    pd_ = np.array([r[2] for r in pairs])
    axr.hlines(y, gt, pd_, color=C_MUTE, lw=1.6, zorder=1)
    axr.scatter(gt, y, s=15, color=C_MAIN, zorder=3,
                edgecolor="white", linewidth=0.7, label="reference netlist")
    axr.scatter(pd_, y, s=15, color=C_ALT, zorder=3,
                edgecolor="white", linewidth=0.7, label="reconstruction")

    axr.set_yticks(y)
    axr.set_yticklabels([r[0] for r in pairs], fontsize=6.0, color=C_SUB)
    axr.set_xlabel("node voltage (V)")
    axr.set_xlim(-1.2, 17.4)
    axr.set_ylim(-0.9, len(pairs) - 0.1)
    axr.grid(axis="y", visible=False)
    axr.set_title("(b)  same graph, measured from a different origin",
                  fontsize=7.4, color=C_INK, loc="left", pad=5)
    # Bottom-left is genuinely empty: the last five nodes sit at ~15 V in both
    # netlists, so nothing is drawn there. Legend and annotation both go there
    # rather than on top of the marks.
    axr.legend(frameon=False, loc="lower left", fontsize=6.6,
               handletextpad=0.3, borderpad=0.2, labelspacing=0.3,
               bbox_to_anchor=(0.01, 0.02))

    # No leader line: the text sits in empty space directly under the bars it
    # describes, and any arrow long enough to reach them crosses every one.
    axr.text(1.2, 3.9, f"the reconstruction makes {ref_net} its 0 V,\n"
             "so every node is displaced",
             fontsize=6.6, color=C_SUB, va="center", ha="left",
             linespacing=1.35)
    save(fig, "fig_qualitative")


# --------------------------------------------------------------------------
def fig_runtime() -> None:
    """Where the wall clock goes, in the two scopes the summary distinguishes.

    Medians, not means. The run's own caveat says to prefer them (other agents
    were on the machine), and they are the numbers the manuscript quotes. That
    rules out a stacked bar: stage medians do not sum to the total median, and
    a stack silently asserts that they do. Grouped bars make no such claim.

    Eight of the fourteen stages are below 6 ms and fold into "all other" --
    plotting them would add eight rows of nothing to read.
    """
    d = json.loads((ROOT / "results/final/runtime/summary.json").read_text())
    e2e = d["scopes"]["e2e"]["stages"]
    cac = d["scopes"]["cached"]["stages"]

    SHOWN = ["detect", "class_head", "stitch", "wires", "textmask"]
    LABEL = {"detect": "symbol detection", "class_head": "class head",
             "stitch": "wire stitching", "wires": "wire extraction",
             "textmask": "text masking"}

    def rest(st):
        return sum(v["median_ms"] for k, v in st.items()
                   if k not in SHOWN and k != "total")

    order = sorted(SHOWN, key=lambda k: -e2e[k]["median_ms"])
    cats = [LABEL[k] for k in order] + ["all other stages"]
    ee = [e2e[k]["median_ms"] for k in order] + [rest(e2e)]
    cc = [cac[k]["median_ms"] for k in order] + [rest(cac)]

    tot_e, tot_c = e2e["total"]["median_ms"], cac["total"]["median_ms"]
    share = e2e["detect"]["share_of_total_median"]
    ratio = d["e2e_over_cached_total"]["median_x"]

    y = np.arange(len(cats))
    h = 0.36
    fig, ax = plt.subplots(figsize=(COL_W, 2.65))
    ax.barh(y - h / 2, ee, h, color=C_MAIN, linewidth=0,
            label=f"end-to-end \u2014 {tot_e:.0f} ms")
    ax.barh(y + h / 2, cc, h, color=C_ALT, linewidth=0,
            label=f"detections cached \u2014 {tot_c:.0f} ms")

    # Two labels, not twelve: the stage that is the whole difference between
    # the scopes, and its vanished counterpart.
    ax.text(ee[0] + 4, y[0] - h / 2, f"{ee[0]:.0f}", va="center", ha="left",
            fontsize=6.6, color=C_MAIN, fontweight="bold")
    ax.text(cc[0] + 4, y[0] + h / 2, f"{cc[0]:.1f}", va="center", ha="left",
            fontsize=6.4, color=C_ALT)

    # No leader line. The text sits in the white space directly beneath the row
    # it describes and names that row in its first two words; an arrow long
    # enough to reach the bar would have to cross the two below it.
    # One line, not two: the clear band between the first two rows is 0.64 of a
    # row high and a two-line block is 0.75, so the second line landed on the
    # class-head bar and was unreadable.
    ax.text(44, y[0] + 0.5,
            f"detection: {share * 100:.0f}% of end-to-end, "
            f"and all of the {ratio:.2f}$\\times$ gap",
            fontsize=6.5, color=C_SUB, va="center", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=6.6, color=C_SUB)
    ax.get_yticklabels()[0].set_color(C_INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("median time per image (ms)")
    ax.set_xlim(0, 200)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5,
              handlelength=1.5, borderpad=0.15, labelspacing=0.35)
    save(fig, "fig_runtime")


# --------------------------------------------------------------------------
def fig_vlm() -> None:
    """What handing the models our detections does to them.

    A slope chart, because the finding IS the change: each model is one line
    from unaided to assisted, and its steepness is the effect that detection
    alone accounts for. Paired bars made the reader subtract.

    This work is a reference rule rather than a third series. It does not move
    between the variants -- it is the system supplying the detections -- so a
    line implying change would be wrong, and a grey categorical hue fails the
    palette checks anyway.
    """
    sig = json.loads((ROOT / "results/table5_significance.json").read_text())
    cmp_ = sig["comparisons"]
    ours = sig["pipeline_strict_success"]

    cb = json.loads(
        (ROOT / "results/vlm/claude_b_test/scored/summary.json").read_text())
    cb_mean = cb["across_repeats"]["strict_success"]["mean"]
    cb_sd = cb["across_repeats"]["strict_success"]["sd"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    x0, x1 = 0.0, 1.0

    ax.axhline(ours, color=C_SUB, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    ax.text(x0 + 0.02, ours + 0.022,
            f"this work \u2014 {ours:.4f}\n(it supplies the boxes in B)",
            fontsize=6.3, color=C_SUB, va="bottom", ha="left", linespacing=1.3)

    # (name, A, B, sd, colour, p-note, vertical side to hang the label on)
    series = [
        ("GPT-5.5", cmp_["gpt_A"]["vlm_strict_success"],
         cmp_["gpt_B"]["vlm_strict_success"], 0.0, C_ALT,
         r"$p=1.1\times10^{-4}$", +1),
        ("Claude Opus 5", cmp_["claude_A"]["vlm_strict_success"],
         cb_mean, cb_sd, C_MAIN,
         f"$p={cmp_['claude_B']['mcnemar']['p_exact']:.2f}$ \u2014 not distinguishable",
         -1),
    ]
    for name, a, b, sd, col, note, side in series:
        ax.plot([x0, x1], [a, b], color=col, lw=1.6, zorder=3,
                solid_capstyle="round")
        ax.scatter([x0, x1], [a, b], s=26, color=col, zorder=4,
                   edgecolor="white", linewidth=0.9)
        if sd:
            ax.errorbar([x1], [b], yerr=[sd], fmt="none", ecolor=col,
                        elinewidth=0.9, capsize=2.0, zorder=3)
        # Claude lands within 0.002 of the reference rule, so its label hangs
        # BELOW its marker and the rule's label sits at the far end. Nothing is
        # nudged: the collision is the finding.
        top = b + 0.020 if side > 0 else b - 0.026
        ax.text(x1 + 0.05, top, f"{name}\n{b:.4f}", fontsize=6.5, color=col,
                va="bottom" if side > 0 else "top", ha="left",
                linespacing=1.3, fontweight="bold")
        ax.text(x1 + 0.05, top - (0.0 if side > 0 else 0.075) + (
            -0.030 if side > 0 else 0.0),
            note, fontsize=5.9, color=C_SUB,
            va="top", ha="left")

    # Both models land on the same value unaided; one label, not two.
    ax.text(x0 - 0.05, cmp_["gpt_A"]["vlm_strict_success"],
            f"both models\n{cmp_['gpt_A']['vlm_strict_success']:.4f}",
            fontsize=6.4, color=C_SUB, va="center", ha="right", linespacing=1.3)
    ax.text(0.30, 0.155, "unaided, they must also\nfind the components",
            fontsize=6.3, color=C_SUB, va="center", ha="left", linespacing=1.35)

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["A: raw scan\n(unaided)",
                        "B: given our\ndetected boxes"], fontsize=6.8,
                       color=C_INK, linespacing=1.4)
    ax.set_ylabel("strict success")
    ax.set_ylim(0.05, 0.80)
    ax.set_xlim(-0.34, 1.72)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    save(fig, "fig_vlm")


# --------------------------------------------------------------------------
def fig_pin_order() -> None:
    """Templates vs the learned port head, per class, as signed movement.

    An arrow per class rather than paired bars: the quantity being argued about
    is the CHANGE, and one class moves backwards. Paired bars render a
    regression as "the right bar is shorter", which reads as a smaller number
    rather than as the anomaly it is; an arrow pointing the other way cannot be
    misread. Colour follows the sign -- a polarity job, not identity.
    """
    d = json.loads((ROOT / "results/final/pin_order/summary.json").read_text())
    t, h = d["templates_only"], d["port_head"]

    rows = [(k, t["by_class"][k]["accuracy"], h["by_class"][k]["accuracy"],
             t["by_class"][k]["decidable"]) for k in t["by_class"]]
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)
    rows.append(("Overall", t["accuracy"], h["accuracy"], t["decidable"]))

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    y = np.arange(len(rows))[::-1]

    for yi, (name, a, b, n) in zip(y, rows):
        col = C_MAIN if b >= a else C_ALT
        ax.annotate("", xy=(b, yi), xytext=(a, yi),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color=col,
                                    shrinkA=0, shrinkB=0, mutation_scale=7.5))
        ax.scatter([a], [yi], s=13, color="white", zorder=4,
                   edgecolor=col, linewidth=1.1)
        ax.text(min(a, b) - 0.022, yi, f"{min(a, b):.3f}", fontsize=6.0,
                color=C_SUB, va="center", ha="right")
        ax.text(max(a, b) + 0.022, yi, f"{max(a, b):.3f}", fontsize=6.2,
                color=col, va="center", ha="left", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  (n={c})" for n, _, _, c in rows], fontsize=6.6,
                       color=C_SUB)
    ax.get_yticklabels()[-1].set_color(C_INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(0.5, color=C_GRID, lw=0.8)   # aggregate, not a sixth class

    # The op-amp row's right half is the only genuinely empty region, and it is
    # the row being described, so the note goes there without a leader.
    oi = [i for i, r in enumerate(rows) if r[0] == "Op-Amp"][0]
    ax.text(0.75, y[oi], "the head makes this one worse:\n"
            "an op-amp's inputs differ by a\n$+$/$-$ glyph, not by geometry",
            fontsize=6.0, color=C_ALT, va="center", ha="left", linespacing=1.35)

    ax.set_xlabel("pin-order accuracy on decidable devices")
    ax.set_xlim(0.30, 1.30)
    ax.set_ylim(-0.75, len(rows) - 0.45)
    ax.grid(axis="y", visible=False)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    save(fig, "fig_pin_order")


# --------------------------------------------------------------------------
def fig_determinism() -> None:
    """Same input, asked five times (us) and three times (a hosted model).

    Both agreements are fractions of the same 192 circuits, so they share an
    axis, with counts as the direct labels -- the count is what a reader wants
    and the fraction is what makes the two systems comparable.

    Directly labelled instead of carrying a legend: with four bars, naming the
    two systems on the first pair costs less space than a legend box and puts
    the name where the eye already is.
    """
    ours = json.loads(
        (ROOT / "results/final/determinism/summary.json").read_text())
    them = json.loads(
        (ROOT / "results/final/vlm_determinism/summary.json").read_text())

    n = ours["n_circuits"]
    cats = ["exact output\n(byte-identical)", "topology\n(naming-invariant)"]
    a = [ours["exact_output_agreement"]["netlist_base_byte_identical_fraction"],
         ours["topology_changes"]["fraction_stable"]]
    b = [them["exact_output_agreement"]["fraction_all_repeats_identical"],
         them["topology_agreement"]["fraction_all_repeats_identical"]]
    b_n = [them["exact_output_agreement"]["n_identical"],
           them["topology_agreement"]["n_identical"]]

    y = np.arange(len(cats))
    h = 0.32
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    ax.barh(y - h / 2, a, h, color=C_MAIN, linewidth=0)
    ax.barh(y + h / 2, b, h, color=C_ALT, linewidth=0)

    for i in range(len(cats)):
        ax.text(a[i] - 0.02, y[i] - h / 2, f"{n}/{n}", va="center", ha="right",
                fontsize=6.3, color="white", fontweight="bold")
        ax.text(b[i] + 0.018, y[i] + h / 2, f"{b_n[i]}/{n}", va="center",
                ha="left", fontsize=6.3, color=C_ALT)

    # Direct labels on the first pair only; the second pair inherits them.
    # The blue label rides inside its bar (there is room, and nothing else is
    # there); the vermillion one sits clear of its count rather than under it.
    ax.text(0.02, y[0] - h / 2, f"this work \u2014 {ours['runs']} fresh interpreters",
            fontsize=6.3, color="white", va="center", ha="left",
            fontweight="bold")
    ax.text(0.42, y[0] + h / 2,
            f"{them['model']} \u2014 {them['n_repeats']} identical requests",
            fontsize=6.3, color=C_ALT, va="center", ha="left",
            fontweight="bold")

    # The band between the two groups is the only region no mark occupies.
    chg = them["topology_agreement"]["n_changed"]
    ax.text(0.02, 0.5,
            f"{chg} of {n} drawings come back as a different circuit;\n"
            f"pairwise agreement "
            f"{them['pairwise_min']:.4f}\u2013{them['pairwise_max']:.4f}",
            fontsize=6.2, color=C_SUB, va="center", ha="left", linespacing=1.35)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=6.6, color=C_SUB, linespacing=1.4)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of circuits identical across runs")
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", visible=False)
    save(fig, "fig_determinism")


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
    "runtime": fig_runtime,
    "pin_order": fig_pin_order,
    "determinism": fig_determinism,
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
