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
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):        # PDF for LaTeX, PNG to eyeball quickly
        fig.savefig(FIG / f"{name}.{ext}")
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
    """Cumulative strict success as each mechanism is added, v1 -> v12."""
    with (ROOT / "results/ablations_test192/wire_method.csv").open() as fh:
        abl = list(csv.DictReader(fh))

    # the csv label carries the arm index and a slug; the paper wants the
    # mechanism, not the slug
    # every line kept under ~12 characters: at this figure width anything
    # longer runs into the neighbouring tick and the axis becomes unreadable
    short = {
        "v1": "classical\n(canny)", "v2": "ink wires\n+ boundary",
        "v3": "+ hole\nstitching", "v4": "+ crossover\nnets",
        "v5": "+ port\ntemplates", "v6": "+ bridge\nspan 7",
        "v7": "+ connect.\nrepair", "v8": "+ snap\nexpand 80",
        "v9": "+ blob\nthresholds", "v10": "+ Sauvola\nbinarize",
        "v11": "+ class\nhead", "v12": "+ head\nensemble",
    }
    keys = [r["label"].split("_")[0] for r in abl]
    y = np.array([float(r["strict_success"]) for r in abl])
    lo = np.array([float(r["strict_success_ci95_lo"]) for r in abl])
    hi = np.array([float(r["strict_success_ci95_hi"]) for r in abl])
    x = np.arange(len(abl))

    fig, ax = plt.subplots(figsize=(WIDE_W, 2.7))
    ax.bar(x, y, 0.62, color=C_TEST, edgecolor="white", linewidth=0.4)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#333333",
                elinewidth=0.7, capsize=2)
    # step connectors make it read as a cumulative progression rather than
    # twelve unrelated configurations
    for i in range(len(x) - 1):
        ax.plot([x[i] + 0.31, x[i + 1] - 0.31], [y[i], y[i]],
                ls=":", lw=0.6, color="#888888")
    for xi, yi, hii in zip(x, y, hi):
        ax.text(xi, hii + 0.018, f"{yi:.3f}", ha="center", va="bottom",
                fontsize=6.2)

    ax.set_xticks(x, [f"{k}\n{short.get(k, '')}" for k in keys],
                  fontsize=5.8, linespacing=1.25)
    ax.set_ylabel("strict end-to-end success")
    ax.set_ylim(0, hi.max() * 1.16)
    ax.margins(x=0.012)
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


FIGURES = {
    "precision_cliff": fig_precision_cliff,
    "ablation_waterfall": fig_ablation_waterfall,
    "oracle_waterfall": fig_oracle_waterfall,
    "per_class_ap": fig_per_class_ap,
    "size_scatter": fig_size_scatter,
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
