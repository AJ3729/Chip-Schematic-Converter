#!/usr/bin/env python3
"""The two manuscript figures for the robustness study.

fig_robustness_cam  a degraded scan beside the detector's attention on it, so a
                    reader can see that the evidence the detector uses survives
                    the degradation rather than take the table's word for it.
fig_robustness      strict success against severity for every condition family,
                    with the clean control as the reference line.

Same chrome and the same validated two-hue palette as the other figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_paper_figures import (C_MAIN, C_ALT, C_MUTE, C_INK, C_SUB,   # noqa: E402
                                COL_W, WIDE_W, style, save)


def fig_cam(stem: str, cond: str) -> None:
    """Clean detail, the same detail degraded, and the attention over it.

    NATIVE RESOLUTION, CROPPED. The first version of this figure showed the
    whole page scaled to the column width, and at that scale the degradation was
    invisible: averaging 3x3 blocks halves the noise, so a delivered sigma of 7.4
    reached the reader as about 3.2 and the "degraded" panel looked identical to
    a clean scan. A crop at 1:1 shows what the pipeline actually received.

    The clean panel is not decoration either. Noise is only judgeable against a
    reference, and without one a reader has no way to tell a degraded scan from
    a slightly grubby original.
    """
    def _read(d, st):
        for e in (".jpg", ".png"):
            im = cv2.imread(str(d / f"{st}{e}"))
            if im is not None:
                return im
        return None
    raw_c = _read(ROOT / "data/robustness/raw/clean", stem)
    raw_d = _read(ROOT / "data/robustness/raw" / cond, stem)
    cam = cv2.imread(str(ROOT / "results/robustness/gradcam/raw" / cond /
                         f"{stem}_cam.jpg"))
    if raw_c is None or raw_d is None or cam is None:
        raise SystemExit(f"missing inputs for {cond}/{stem}")
    H, W = raw_c.shape[:2]
    # a window over real ink, sized so 1:1 pixels survive to print
    cw, ch = min(430, W), min(300, H)
    x0 = int(W * 0.10); y0 = int(H * 0.30)
    x0 = max(0, min(x0, W - cw)); y0 = max(0, min(y0, H - ch))

    def crop(im):
        sy, sx = im.shape[0] / H, im.shape[1] / W      # cam may differ slightly
        c = im[int(y0 * sy):int((y0 + ch) * sy), int(x0 * sx):int((x0 + cw) * sx)]
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    panels = [(crop(raw_c), "(a)  clean scan"),
              (crop(raw_d), "(b)  degraded input"),
              (crop(cam),   "(c)  detector attention")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE_W, WIDE_W * ch / (cw * 3) + 0.34))
    for ax, (img, t) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")     # nearest: do not re-smooth
        ax.set_axis_off()
        ax.set_title(t, fontsize=7.4, color=C_INK, loc="left", pad=4)
    fig.subplots_adjust(wspace=0.025)
    save(fig, "fig_robustness_cam")


def fig_curve() -> None:
    s = json.loads((ROOT / "results/robustness/SUMMARY.json").read_text())
    conds = s["conditions"]
    base = conds["clean"]["strict_success"]

    fams: dict[str, list[tuple[int, float]]] = {}
    for name, r in conds.items():
        if r["family"] not in ("photometric", "geometric"):
            continue
        fam = name.rsplit("_s", 1)[0]
        fams.setdefault(fam, []).append((r["severity"], r["strict_success"]))
    for v in fams.values():
        v.sort()

    # the two conditions the text calls out; everything else is context
    LOUD = {"speckle": C_ALT, "perspective": C_MAIN}
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.axhline(base, color=C_SUB, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    # away from the y-axis: at the left edge this label sat on the 0.55 tick
    ax.text(2.4, base + 0.010, f"clean control {base:.4f}", fontsize=6.2,
            color=C_SUB, va="bottom", ha="center")

    for fam, pts in sorted(fams.items()):
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        col = LOUD.get(fam)
        ax.plot(x, y, lw=1.5 if col else 0.9, color=col or C_MUTE,
                marker="o", ms=3.2 if col else 2.2,
                markerfacecolor=col or C_MUTE, zorder=4 if col else 2)
        if col:
            ax.text(x[-1] + 0.08, y[-1], fam, fontsize=6.4, color=col,
                    va="center", ha="left", fontweight="bold")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["mild", "moderate", "severe"], fontsize=6.8)
    ax.set_xlim(0.8, 3.9)
    ax.set_ylim(0.18, 0.60)
    ax.set_xlabel("degradation severity")
    ax.set_ylabel("strict success")
    ax.grid(axis="x", visible=False)
    ax.text(0.92, 0.235, "seven other\nconditions", fontsize=6.2, color=C_MUTE,
            va="center", ha="left", linespacing=1.35, transform=ax.transData)
    save(fig, "fig_robustness")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="circuit_1013")
    ap.add_argument("--cond", default="gauss_noise_s2")
    a = ap.parse_args()
    style()
    fig_cam(a.stem, a.cond)
    fig_curve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
